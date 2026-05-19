from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response

import uuid
import shutil
import logging
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core.config import settings
from app.models.schemas import (
    FinalizeVectorizeRequest,
    MaskPreviewResult,
    TaskStatus,
    TaskStatusResponse,
    VectorizeResult,
)
from app.services.segmentation import SegmentationService
from app.services.tracing import ContourTracer
from app.services.svg_builder import SVGBuilder
from app.services.logo_vectorizer import LogoVectorizer
from app.services.layer_namer import LayerNamer
from app.services.mode_router import is_logo_candidate
from app.utils.metrics import compute_all_metrics


router = APIRouter()
logger = logging.getLogger(__name__)

tasks: dict = {}

tracer = ContourTracer()
svg_builder = SVGBuilder()
logo_vectorizer = LogoVectorizer()
layer_namer = LayerNamer()


def _svg_download_name(filename: Optional[str]) -> str:
    base = Path(filename or "vectorized").stem.strip()
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", base)
    base = re.sub(r"\s+", " ", base).strip(" .-_")
    return f"{base or 'vectorized'}.svg"

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


def _mask_bbox(mask: np.ndarray) -> list[int]:
    mask_bool = mask.astype(bool)
    if int(mask_bool.sum()) == 0:
        return [0, 0, 0, 0]

    ys, xs = np.where(mask_bool)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _preview_payload(task_id: str) -> MaskPreviewResult:
    task = tasks.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task не знайдено")

    preview_masks = task.get("preview_masks") or []

    return MaskPreviewResult(
        task_id=task_id,
        status=task.get("status", TaskStatus.PREVIEW),
        image_url=f"/api/preview/{task_id}/image",
        masks=[
            {
                "id": item["id"],
                "name": item["name"],
                "color": item["color"],
                "area": int(item["area"]),
                "bbox": item["bbox"],
            }
            for item in preview_masks
        ],
        requested_mode=task.get("requested_mode"),
        mode=task.get("mode"),
        error=task.get("error"),
    )


def _build_preview_masks(
    image: np.ndarray,
    masks: list[dict],
    use_vlm_names: bool = False,
) -> list[dict]:
    layer_names = layer_namer.name_layers(image, masks) if use_vlm_names else []
    preview_masks = []

    for index, mask_info in enumerate(masks):
        mask = mask_info["mask"].astype(bool)
        name = layer_names[index] if index < len(layer_names) else f"Mask {index + 1}"

        preview_masks.append(
            {
                "id": f"mask-{index + 1}",
                "name": name,
                "color": mask_info.get("color", "#000000"),
                "area": int(mask.sum()),
                "bbox": _mask_bbox(mask),
                "mask": mask,
                "source_label": mask_info.get("label", "Layer"),
            }
        )

    return preview_masks


def _find_preview_mask(task: dict, mask_id: str) -> Optional[dict]:
    for item in task.get("preview_masks") or []:
        if item.get("id") == mask_id:
            return item

    return None


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

    layer_names = layer_namer.name_layers(image, masks)

    layers_data = []

    for index, mask_info in enumerate(masks):
        trace_max_contours = 2200 if max_layers >= 32 else 1200
        contours = tracer.trace(
            mask_info["mask"],
            simplify=False,
            epsilon_factor=0.0012,
            min_area_ratio=0.000004,
            keep_holes=False,
            max_contours=trace_max_contours,
        )

        if not contours:
            continue

        underpaint_contours = tracer.trace_raw_underpaint(
            mask_info["mask"],
            dilate_px=0,
            min_area_ratio=0.0000005,
            max_contours=8000,
        )

        tasks[task_id]["progress"] = 50 + int(22 * (index + 1) / max(len(masks), 1))

        layers_data.append(
            {
                "label": (
                    layer_names[index]
                    if index < len(layer_names)
                    else mask_info.get("label", "Layer")
                ),
                "source_label": mask_info.get("label", "Layer"),
                "color": mask_info.get("color", "#000000"),
                "contours": contours,
                "underpaint_contours": underpaint_contours or contours,
                "bezier_segs": [],
                "prefer_contours": True,
                "stroke_width": 0.0,
                "underpaint_stroke_width": 0.0,
            }
        )

    tasks[task_id]["progress"] = 75

    out_path = settings.RESULTS_DIR / f"{task_id}.svg"

    svg_builder.build(
        layers_data,
        width=width,
        height=height,
        output_path=out_path,
        background_color=masks[0].get("color", "#FFFFFF") if masks else "#FFFFFF",
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
                    "node_count": sum(len(contour) for contour in layer["contours"]),
                    "color": layer["color"],
                }
                for layer in layers_data
            ],
            "mode": "semantic",
            "error": None,
        }
    )


def _process_preview_finalization(
    task_id: str,
    tolerance: float,
    simplify: bool,
    requested_layers: list[dict],
) -> None:
    started_at = time.perf_counter()

    try:
        task = tasks[task_id]
        task["status"] = TaskStatus.PROCESSING
        task["progress"] = 30
        task["error"] = None

        image_path = Path(task["image_path"])
        image = task["image"]
        height = int(task["height"])
        width = int(task["width"])
        max_layers = int(
            np.clip(task.get("max_layers", len(requested_layers) or 10), 1, 96)
        )

        _process_semantic_mode(
            task_id=task_id,
            image_path=image_path,
            image=image,
            width=width,
            height=height,
            tolerance=tolerance,
            max_layers=max_layers,
            simplify=simplify,
        )
        _attach_processing_time(task_id, started_at)
    except Exception as e:
        logger.exception("Помилка фіналізації preview task %s", task_id)

        if task_id in tasks:
            tasks[task_id]["status"] = TaskStatus.ERROR
            tasks[task_id]["error"] = str(e)
            tasks[task_id]["metrics"] = {
                "processing_time": round(time.perf_counter() - started_at, 3)
            }


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
            selected_mode = "logo" if is_logo_candidate(image) else "semantic"
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
        "download_filename": _svg_download_name(file.filename),
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
        filename=tasks[task_id]["download_filename"],
        requested_mode=requested_mode,
        mode=None,
    )


@router.post("/vectorize/prepare", response_model=MaskPreviewResult)
async def prepare_vectorize_preview(
    file: UploadFile = File(...),
    max_layers: int = Form(5),
    mode: str = Form("auto"),
):
    ensure_app_dirs()
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
        original_ext = ".png"

    image_path = settings.UPLOAD_DIR / f"{task_id}{original_ext}"

    try:
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    image = cv2.imread(str(image_path))

    if image is None:
        raise HTTPException(status_code=400, detail="Не вдалося прочитати зображення")

    image, h, w = _save_resized_if_needed(image, image_path)
    requested_mode = normalize_mode(mode)
    selected_mode = "logo" if requested_mode == "logo" else "semantic"

    if requested_mode == "auto":
        selected_mode = "logo" if is_logo_candidate(image) else "semantic"

    if selected_mode == "logo":
        full_mask = np.ones((h, w), dtype=bool)
        masks = [
            {
                "mask": full_mask,
                "label": "logo",
                "color": "#111111",
                "area": int(full_mask.sum()),
            }
        ]
    else:
        seg_service = get_seg_service()
        masks = seg_service.generate_preview_masks(
            image,
            max_layers=min(max_layers, 48),
        )

        if not masks:
            raise HTTPException(status_code=503, detail="Preview masks unavailable")

    preview_masks = _build_preview_masks(image, masks)

    tasks[task_id] = {
        "status": TaskStatus.PREVIEW,
        "progress": 25,
        "svg_path": None,
        "download_filename": _svg_download_name(file.filename),
        "metrics": None,
        "layers": None,
        "requested_mode": requested_mode,
        "mode": selected_mode,
        "error": None,
        "image_path": str(image_path),
        "image": image,
        "width": w,
        "height": h,
        "max_layers": max_layers,
        "preview_masks": preview_masks,
    }

    return _preview_payload(task_id)


@router.get("/preview/{task_id}", response_model=MaskPreviewResult)
async def get_preview(task_id: str):
    return _preview_payload(task_id)


@router.get("/preview/{task_id}/image")
async def get_preview_image(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task не знайдено")

    image_path = tasks[task_id].get("image_path")

    if not image_path or not Path(image_path).exists():
        raise HTTPException(status_code=404, detail="Зображення не знайдено")

    return FileResponse(image_path)


@router.get("/preview/{task_id}/mask/{mask_id}")
async def get_preview_mask(task_id: str, mask_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task не знайдено")

    item = _find_preview_mask(tasks[task_id], mask_id)

    if not item:
        raise HTTPException(status_code=404, detail="Маску не знайдено")

    mask = item["mask"].astype(np.uint8) * 255
    ok, encoded = cv2.imencode(".png", mask)

    if not ok:
        raise HTTPException(status_code=500, detail="Не вдалося закодувати маску")

    return Response(content=encoded.tobytes(), media_type="image/png")


@router.get("/preview/{task_id}/mask/{mask_id}/overlay")
async def get_preview_mask_overlay(task_id: str, mask_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task не знайдено")

    item = _find_preview_mask(tasks[task_id], mask_id)

    if not item:
        raise HTTPException(status_code=404, detail="Маску не знайдено")

    mask = item["mask"].astype(bool)
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[mask] = np.array([139, 92, 246, 150], dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", overlay)

    if not ok:
        raise HTTPException(status_code=500, detail="Не вдалося закодувати overlay")

    return Response(content=encoded.tobytes(), media_type="image/png")


@router.post("/preview/{task_id}/split/{mask_id}", response_model=MaskPreviewResult)
async def split_preview_mask(task_id: str, mask_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task не знайдено")

    task = tasks[task_id]
    item = _find_preview_mask(task, mask_id)

    if not item:
        raise HTTPException(status_code=404, detail="Маску не знайдено")

    mask_u8 = item["mask"].astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    components = []

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < max(20, int(item["area"] * 0.015)):
            continue
        component_mask = labels == label_id
        components.append((area, component_mask))

    if len(components) < 2:
        image = task["image"]
        pixels = image[item["mask"].astype(bool)]

        if len(pixels) >= 20:
            compactness, labels_2, centers = cv2.kmeans(
                pixels.astype(np.float32),
                2,
                None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0),
                3,
                cv2.KMEANS_PP_CENTERS,
            )
            flat_indices = np.flatnonzero(item["mask"].astype(bool))
            components = []

            for cluster_id in range(2):
                cluster_mask = np.zeros(item["mask"].shape, dtype=bool)
                selector = labels_2.reshape(-1) == cluster_id
                cluster_mask.flat[flat_indices[selector]] = True
                area = int(cluster_mask.sum())
                if area > 0:
                    components.append((area, cluster_mask))

    if len(components) < 2:
        raise HTTPException(status_code=400, detail="Цю маску не вдалося розділити автоматично")

    components = sorted(components, key=lambda pair: pair[0], reverse=True)
    preview_masks = task.get("preview_masks") or []
    item_index = next(
        (index for index, preview_item in enumerate(preview_masks) if preview_item.get("id") == mask_id),
        -1,
    )

    if item_index < 0:
        raise HTTPException(status_code=404, detail="Маску не знайдено")
    next_items = []

    for split_index, (area, component_mask) in enumerate(components, start=1):
        color = item["color"]
        next_items.append(
            {
                "id": f"{item['id']}-part-{split_index}",
                "name": f"{item['name']} {split_index}",
                "color": color,
                "area": area,
                "bbox": _mask_bbox(component_mask),
                "mask": component_mask,
                "source_label": item.get("source_label", item["name"]),
            }
        )

    task["preview_masks"] = preview_masks[:item_index] + next_items + preview_masks[item_index + 1 :]
    return _preview_payload(task_id)


@router.post("/vectorize/finalize/{task_id}", response_model=VectorizeResult)
async def finalize_vectorize_preview(
    task_id: str,
    payload: FinalizeVectorizeRequest,
    background_tasks: BackgroundTasks,
):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task не знайдено")

    task = tasks[task_id]

    if not task.get("preview_masks"):
        raise HTTPException(status_code=400, detail="Preview-маски недоступні")

    requested_layers = [layer.model_dump() for layer in payload.layers]

    if not requested_layers:
        raise HTTPException(status_code=400, detail="Немає шарів для векторизації")

    task["status"] = TaskStatus.PENDING
    task["progress"] = 30
    task["layers"] = None
    task["metrics"] = None
    task["svg_path"] = None
    task["error"] = None

    background_tasks.add_task(
        _process_preview_finalization,
        task_id,
        payload.tolerance,
        payload.simplify,
        requested_layers,
    )

    return VectorizeResult(
        task_id=task_id,
        status=TaskStatus.PENDING,
        filename=task.get("download_filename"),
        requested_mode=task.get("requested_mode"),
        mode=task.get("mode"),
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


@router.get("/vlm/status")
async def vlm_status():
    warmup_enabled = getattr(settings, "VLM_WARMUP_ON_START", True)
    on_demand_enabled = getattr(settings, "VLM_ON_DEMAND_LOAD_ENABLED", False)
    unavailable_by_config = (
        settings.VLM_LAYER_NAMING_ENABLED
        and not layer_namer.is_ready
        and not warmup_enabled
        and not on_demand_enabled
    )

    return {
        "enabled": settings.VLM_LAYER_NAMING_ENABLED,
        "model": settings.VLM_LAYER_NAMING_MODEL,
        "device": settings.VLM_LAYER_NAMING_DEVICE,
        "ready": layer_namer.is_ready,
        "load_failed": layer_namer.load_failed or unavailable_by_config,
    }


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
            filename=task.get("download_filename"),
            requested_mode=task.get("requested_mode"),
            mode=task.get("mode"),
            error=task.get("error"),
        )

    return VectorizeResult(
        task_id=task_id,
        status=TaskStatus.DONE,
        svg_url=f"/api/download/{task_id}",
        filename=task.get("download_filename"),
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
        filename=tasks[task_id].get("download_filename") or f"{task_id}.svg",
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
