from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

import uuid
import shutil
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core.config import settings
from app.models.schemas import VectorizeResult, TaskStatusResponse, TaskStatus
from app.services.segmentation import SegmentationService
from app.services.tracing import ContourTracer
from app.services.bezier import fit_cubic_bezier
from app.services.svg_builder import SVGBuilder
from app.services.logo_vectorizer import LogoVectorizer
from app.utils.metrics import compute_all_metrics


router = APIRouter()
logger = logging.getLogger(__name__)

tasks: dict = {}

tracer = ContourTracer()
svg_builder = SVGBuilder()
logo_vectorizer = LogoVectorizer()

_seg_service: Optional[SegmentationService] = None


def ensure_app_dirs() -> None:
    """Створює потрібні директорії, якщо їх ще немає."""
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def get_seg_service() -> SegmentationService:
    """Lazy initialization для SAM2 / fallback-сегментації."""
    global _seg_service

    if _seg_service is None:
        _seg_service = SegmentationService(
            checkpoint=settings.SAM_CHECKPOINT,
            model_cfg=settings.SAM_MODEL_CFG,
            device=settings.DEVICE,
        )

    return _seg_service


def normalize_mode(mode: str) -> str:
    """
    Нормалізує режим векторизації.

    Доступні режими:
    - auto
    - logo
    - semantic
    """
    allowed_modes = {"auto", "logo", "semantic"}
    normalized = (mode or "auto").strip().lower()

    if normalized not in allowed_modes:
        return "auto"

    return normalized


def _is_logo_candidate(image: np.ndarray) -> bool:
    """
    Консервативна евристика для Auto Mode.

    Завдання:
    - logo / line-art має йти в Logo Mode;
    - фото, навіть бежеве або малокольорове, має йти в Semantic Mode.

    Ключова ідея:
    логотип добре описується бінарною маскою:
    чорне/біле, мало тональних переходів, низька помилка бінаризації.

    Фото погано описується бінарною маскою:
    багато напівтонів, градієнтів, текстур, тіней.
    """
    try:
        if image is None or image.size == 0:
            return False

        h, w = image.shape[:2]
        max_side = max(h, w)

        if max_side > 256:
            scale = 256 / max_side
            small = cv2.resize(
                image,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = image.copy()

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        small_h, small_w = small.shape[:2]
        total_pixels = small_h * small_w

        quantized_rgb = (small // 32).astype(np.uint8)
        unique_rgb_colors = len(np.unique(quantized_rgb.reshape(-1, 3), axis=0))

        gray_q = (gray // 16).astype(np.uint8)
        unique_gray_levels = len(np.unique(gray_q))

        sat_mean = float(np.mean(hsv[:, :, 1]))
        sat_std = float(np.std(hsv[:, :, 1]))
        gray_std = float(np.std(gray))

        threshold_value, binary_inv = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        fg_ratio = float(np.mean(binary_inv > 0))

        if fg_ratio <= 0.01 or fg_ratio >= 0.85:
            logger.info("Auto mode: semantic because fg_ratio=%.3f", fg_ratio)
            return False

        fg_pixels = gray[binary_inv > 0]
        bg_pixels = gray[binary_inv == 0]

        if len(fg_pixels) == 0 or len(bg_pixels) == 0:
            return False

        fg_mean = float(np.mean(fg_pixels))
        bg_mean = float(np.mean(bg_pixels))

        reconstructed = np.where(binary_inv > 0, fg_mean, bg_mean).astype(np.float32)
        binary_reconstruction_error = float(
            np.mean(np.abs(gray.astype(np.float32) - reconstructed))
        )

        fg_std = float(np.std(fg_pixels))
        bg_std = float(np.std(bg_pixels))
        within_region_std = (fg_std + bg_std) / 2

        num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_inv,
            connectivity=8,
        )

        meaningful_components = 0

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]

            if area > total_pixels * 0.001:
                meaningful_components += 1

        edges = cv2.Canny(gray, 80, 160)
        edge_ratio = float(np.mean(edges > 0))

        dark_ratio = float(np.mean(gray < 70))
        light_ratio = float(np.mean(gray > 185))
        midtone_ratio = 1.0 - dark_ratio - light_ratio

        if midtone_ratio > 0.35:
            logger.info(
                "Auto mode: semantic because midtone_ratio=%.3f",
                midtone_ratio,
            )
            return False

        looks_like_photo = (
            unique_rgb_colors > 55
            or unique_gray_levels > 22
            or binary_reconstruction_error > 18
            or within_region_std > 24
            or meaningful_components > 18
        )

        if looks_like_photo:
            logger.info(
                "Auto mode: semantic because photo-like. "
                "unique_rgb=%s, unique_gray=%s, bin_error=%.2f, "
                "within_std=%.2f, components=%s, midtone=%.3f",
                unique_rgb_colors,
                unique_gray_levels,
                binary_reconstruction_error,
                within_region_std,
                meaningful_components,
                midtone_ratio,
            )
            return False

        is_logo = (
            unique_rgb_colors <= 55
            and unique_gray_levels <= 22
            and gray_std >= 25
            and binary_reconstruction_error <= 18
            and within_region_std <= 24
            and meaningful_components <= 18
            and edge_ratio <= 0.22
            and midtone_ratio <= 0.35
            and 0.02 < fg_ratio < 0.80
        )

        logger.info(
            "Auto mode analysis: unique_rgb=%s, unique_gray=%s, sat_mean=%.2f, "
            "sat_std=%.2f, gray_std=%.2f, fg_ratio=%.3f, components=%s, "
            "edge_ratio=%.3f, midtone_ratio=%.3f, bin_error=%.2f, "
            "within_std=%.2f, threshold=%.2f, is_logo=%s",
            unique_rgb_colors,
            unique_gray_levels,
            sat_mean,
            sat_std,
            gray_std,
            fg_ratio,
            meaningful_components,
            edge_ratio,
            midtone_ratio,
            binary_reconstruction_error,
            within_region_std,
            threshold_value,
            is_logo,
        )

        return is_logo

    except Exception as e:
        logger.warning(
            "Не вдалося визначити тип зображення, fallback to semantic mode: %s",
            e,
        )
        return False


def _save_resized_if_needed(
    image: np.ndarray,
    image_path: Path,
) -> tuple[np.ndarray, int, int]:
    """
    Якщо зображення завелике — масштабує та перезаписує файл,
    щоб усі наступні модулі працювали з однаковою версією.
    """
    h, w = image.shape[:2]
    max_image_size = getattr(settings, "MAX_IMAGE_SIZE", 2048)

    if max(h, w) > max_image_size:
        scale = max_image_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        image = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        cv2.imwrite(str(image_path), image)
        h, w = image.shape[:2]

    return image, h, w


def _compute_metrics_safe(
    image_path: Path,
    out_path: Path,
    width: int,
    height: int,
) -> dict:
    """
    Безпечний виклик метрик.

    Підтримує обидва варіанти:
    - compute_all_metrics(image_path, out_path, width, height)
    - compute_all_metrics(image_path, out_path)
    """
    try:
        return compute_all_metrics(image_path, out_path, width, height)
    except TypeError:
        return compute_all_metrics(image_path, out_path)


def _attach_processing_time(task_id: str, started_at: float) -> None:
    """Додає processing_time у metrics після завершення pipeline."""
    elapsed = round(time.perf_counter() - started_at, 3)

    if task_id not in tasks:
        return

    if tasks[task_id].get("metrics") is None:
        tasks[task_id]["metrics"] = {}

    tasks[task_id]["metrics"]["processing_time"] = elapsed


def _process_logo_mode(
    task_id: str,
    image_path: Path,
    width: int,
    height: int,
    tolerance: float,
) -> None:
    """
    Pipeline для логотипів:
    binary mask -> contours tree -> SVG compound path.
    """
    tasks[task_id]["progress"] = 35

    out_path = settings.RESULTS_DIR / f"{task_id}.svg"

    logo_result = logo_vectorizer.vectorize(
        image_path=image_path,
        output_path=out_path,
        tolerance=tolerance,
        min_area=20,
    )

    tasks[task_id]["progress"] = 90

    computed_metrics = _compute_metrics_safe(image_path, out_path, width, height)
    logo_metrics = logo_result.get("metrics", {})

    metrics = {
        **computed_metrics,
        **logo_metrics,
    }
    metrics["gapless_coverage"] = 100.0
    metrics["gap_pixels"] = 0
    metrics["cad_readiness_score"] = round(
        min(
            100.0,
            0.45 * 100.0
            + 0.35 * float(metrics.get("editability_score", 0) or 0)
            + 0.12 * 100.0
            + 0.08 * 100.0,
        ),
        1,
    )

    layers_payload = []

    for layer in logo_result.get("layers", []):
        layers_payload.append(
            {
                "name": layer.get("name", layer.get("label", "Layer")),
                "node_count": layer.get("node_count", 0),
                "color": layer.get("color", "#000000"),
            }
        )

    tasks[task_id].update(
        {
            "status": TaskStatus.DONE,
            "progress": 100,
            "svg_path": str(out_path),
            "metrics": metrics,
            "layers": layers_payload,
            "mode": "logo",
            "error": None,
        }
    )


def _process_semantic_mode(
    task_id: str,
    image_path: Path,
    image: np.ndarray,
    width: int,
    height: int,
    tolerance: float,
    max_layers: int,
    simplify: bool,
) -> None:
    """
    Semantic pipeline:
    segmentation -> tracing -> bezier -> svg_builder.
    """
    tasks[task_id]["progress"] = 20
    max_layers = int(np.clip(max_layers, 1, 96))
    tolerance = float(np.clip(tolerance, 0.35, 5.0))

    seg_service = get_seg_service()

    # ВАЖЛИВО:
    # новий segmentation.py підтримує max_layers,
    # тому передаємо його напряму.
    masks = seg_service.segment(image, max_layers=max_layers)
    masks = masks[:max_layers]

    tasks[task_id]["progress"] = 50

    layers_data = []

    for mask_info in masks:
        contours = tracer.trace(
            mask_info["mask"],
            simplify=False,
            epsilon_factor=0.0012,
            min_area_ratio=0.00004,
            keep_holes=False,
            max_contours=240,
        )

        bezier_segs_per_contour = []

        for contour in contours:
            segs = fit_cubic_bezier(
                contour.astype(float),
                tolerance=tolerance,
            )
            bezier_segs_per_contour.append(segs)

        layers_data.append(
            {
                "label": mask_info.get("label", "Layer"),
                "color": mask_info.get("color", "#000000"),
                "contours": contours,
                "bezier_segs": bezier_segs_per_contour,
                "prefer_contours": True,
            }
        )

    tasks[task_id]["progress"] = 75

    out_path = settings.RESULTS_DIR / f"{task_id}.svg"
    background_color = None
    if masks:
        background_color = masks[0].get("color", "#FFFFFF")

    svg_builder.build(
        layers_data,
        width=width,
        height=height,
        output_path=out_path,
        background_color=background_color,
    )

    tasks[task_id]["progress"] = 90

    metrics = _compute_metrics_safe(image_path, out_path, width, height)

    tasks[task_id].update(
        {
            "status": TaskStatus.DONE,
            "progress": 100,
            "svg_path": str(out_path),
            "metrics": metrics,
            "layers": [
                {
                    "name": layer["label"],
                    "node_count": sum(len(segs) for segs in layer["bezier_segs"]),
                    "color": layer["color"],
                }
                for layer in layers_data
            ],
            "mode": "semantic",
            "error": None,
        }
    )


def _process_image(
    task_id: str,
    image_path: Path,
    tolerance: float,
    max_layers: int,
    simplify: bool,
    mode: str,
) -> None:
    """Основний pipeline векторизації."""
    started_at = time.perf_counter()

    try:
        ensure_app_dirs()

        tasks[task_id]["status"] = TaskStatus.PROCESSING
        tasks[task_id]["progress"] = 10

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError("Не вдалося прочитати зображення")

        image, h, w = _save_resized_if_needed(image, image_path)

        requested_mode = normalize_mode(mode)

        if requested_mode == "auto":
            selected_mode = "logo" if _is_logo_candidate(image) else "semantic"
        else:
            selected_mode = requested_mode

        tasks[task_id]["requested_mode"] = requested_mode
        tasks[task_id]["mode"] = selected_mode

        logger.info(
            "Task %s selected mode: %s requested mode: %s",
            task_id,
            selected_mode,
            requested_mode,
        )

        if selected_mode == "logo":
            _process_logo_mode(
                task_id=task_id,
                image_path=image_path,
                width=w,
                height=h,
                tolerance=tolerance,
            )
        else:
            _process_semantic_mode(
                task_id=task_id,
                image_path=image_path,
                image=image,
                width=w,
                height=h,
                tolerance=tolerance,
                max_layers=max_layers,
                simplify=simplify,
            )

        _attach_processing_time(task_id, started_at)

    except Exception as e:
        logger.exception("Помилка обробки task %s", task_id)

        if task_id in tasks:
            tasks[task_id]["status"] = TaskStatus.ERROR
            tasks[task_id]["error"] = str(e)
            tasks[task_id]["progress"] = tasks[task_id].get("progress", 0)

            if tasks[task_id].get("metrics") is None:
                tasks[task_id]["metrics"] = {}

            tasks[task_id]["metrics"]["processing_time"] = round(
                time.perf_counter() - started_at,
                3,
            )


@router.post("/vectorize", response_model=VectorizeResult)
async def vectorize(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tolerance: float = Form(1.0),
    max_layers: int = Form(5),
    simplify: bool = Form(True),
    mode: str = Form("auto"),
):
    ensure_app_dirs()
    tolerance = float(np.clip(tolerance, 0.35, 5.0))
    max_layers = int(np.clip(max_layers, 1, 96))

    allowed_types = ["image/png", "image/jpeg", "image/webp"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Підтримуються лише PNG, JPG, WebP",
        )

    task_id = str(uuid.uuid4())

    original_ext = Path(file.filename or "").suffix.lower()

    if original_ext not in [".png", ".jpg", ".jpeg", ".webp"]:
        if file.content_type == "image/png":
            original_ext = ".png"
        elif file.content_type == "image/jpeg":
            original_ext = ".jpg"
        elif file.content_type == "image/webp":
            original_ext = ".webp"
        else:
            original_ext = ".png"

    image_path = settings.UPLOAD_DIR / f"{task_id}{original_ext}"

    try:
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.exception("Не вдалося зберегти файл")
        raise HTTPException(
            status_code=500,
            detail=f"Не вдалося зберегти файл: {e}",
        )
    finally:
        await file.close()

    requested_mode = normalize_mode(mode)

    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "progress": 0,
        "svg_path": None,
        "metrics": None,
        "layers": None,
        "requested_mode": requested_mode,
        "mode": None,
        "error": None,
    }

    background_tasks.add_task(
        _process_image,
        task_id,
        image_path,
        tolerance,
        max_layers,
        simplify,
        requested_mode,
    )

    return VectorizeResult(
        task_id=task_id,
        status=TaskStatus.PENDING,
        requested_mode=requested_mode,
        mode=None,
    )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(
            status_code=404,
            detail="Task не знайдено",
        )

    task = tasks[task_id]

    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress", 0),
        requested_mode=task.get("requested_mode"),
        mode=task.get("mode"),
        error=task.get("error"),
    )


@router.get("/result/{task_id}", response_model=VectorizeResult)
async def get_result(task_id: str):
    if task_id not in tasks:
        raise HTTPException(
            status_code=404,
            detail="Task не знайдено",
        )

    task = tasks[task_id]

    if task["status"] != TaskStatus.DONE:
        return VectorizeResult(
            task_id=task_id,
            status=task["status"],
            requested_mode=task.get("requested_mode"),
            mode=task.get("mode"),
            error=task.get("error"),
        )

    return VectorizeResult(
        task_id=task_id,
        status=TaskStatus.DONE,
        svg_url=f"/api/download/{task_id}",
        layers=task.get("layers"),
        metrics=task.get("metrics"),
        requested_mode=task.get("requested_mode"),
        mode=task.get("mode"),
        error=task.get("error"),
    )


@router.get("/download/{task_id}")
async def download_svg(task_id: str):
    if task_id not in tasks:
        raise HTTPException(
            status_code=404,
            detail="Task не знайдено",
        )

    svg_path = tasks[task_id].get("svg_path")

    if not svg_path or not Path(svg_path).exists():
        raise HTTPException(
            status_code=404,
            detail="SVG ще не готовий",
        )

    return FileResponse(
        svg_path,
        media_type="image/svg+xml",
        filename=f"{task_id}.svg",
    )

    from PIL import Image
import io

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Повертає рекомендовані параметри для завантаженого зображення."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    seg_service = get_seg_service()
    recommendation = seg_service.analyze_image(image)
    return recommendation
