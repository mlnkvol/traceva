import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from eval.baselines import result_record
from eval.config import BACKEND_DIR, BenchmarkConfig


if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _run_logo(image_path: Path, output_svg: Path, config: BenchmarkConfig) -> None:
    from app.services.logo_vectorizer import LogoVectorizer

    LogoVectorizer().vectorize(
        image_path=image_path,
        output_path=output_svg,
        tolerance=config.traceva_tolerance,
        min_area=20,
    )


def _run_semantic(image_path: Path, output_svg: Path, config: BenchmarkConfig) -> None:
    from app.core.config import settings
    from app.services.bezier import fit_cubic_bezier
    from app.services.layer_namer import LayerNamer
    from app.services.segmentation import SegmentationService
    from app.services.svg_builder import SVGBuilder
    from app.services.tracing import ContourTracer

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    settings.VLM_LAYER_NAMING_ENABLED = bool(config.vlm_enabled)
    seg_service = SegmentationService(
        checkpoint=settings.SAM_CHECKPOINT,
        model_cfg=settings.SAM_MODEL_CFG,
        device=settings.DEVICE,
    )
    masks = seg_service.segment(image, max_layers=config.traceva_max_layers)
    masks = masks[: config.traceva_max_layers]

    tracer = ContourTracer()
    layer_names = LayerNamer().name_layers(image, masks)
    layers = []

    for index, mask_info in enumerate(masks):
        contours = tracer.trace(
            mask_info["mask"],
            simplify=False,
            epsilon_factor=0.0012,
            min_area_ratio=0.000004,
            keep_holes=False,
            max_contours=1800,
        )
        if not contours:
            continue

        underpaint = tracer.trace_raw_underpaint(
            mask_info["mask"],
            dilate_px=0,
            min_area_ratio=0.0000005,
            max_contours=6000,
        )
        bezier_segs = [
            fit_cubic_bezier(contour.astype(float), tolerance=config.traceva_tolerance)
            for contour in contours
        ]
        layers.append(
            {
                "label": layer_names[index] if index < len(layer_names) else mask_info.get("label", "Layer"),
                "color": mask_info.get("color", "#000000"),
                "contours": contours,
                "underpaint_contours": underpaint or contours,
                "bezier_segs": bezier_segs,
                "prefer_contours": False,
                "stroke_width": 0.25,
                "underpaint_stroke_width": 0.0,
            }
        )

    h, w = image.shape[:2]
    SVGBuilder().build(
        layers,
        width=w,
        height=h,
        output_path=output_svg,
        background_color=masks[0].get("color", "#FFFFFF") if masks else "#FFFFFF",
    )


def run(image_path: Path, output_svg: Path, mode: str, config: BenchmarkConfig) -> dict:
    method = f"traceva_{mode}"
    started = time.perf_counter()
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    os.environ["VLM_LAYER_NAMING_ENABLED"] = "true" if config.vlm_enabled else "false"

    try:
        if mode not in {"auto", "logo", "semantic"}:
            raise ValueError(f"Unsupported traceva mode: {mode}")

        actual_mode = mode
        if mode == "auto":
            from app.services.mode_router import is_logo_candidate

            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            actual_mode = "logo" if is_logo_candidate(image) else "semantic"

        if actual_mode == "logo":
            _run_logo(image_path, output_svg, config)
        else:
            _run_semantic(image_path, output_svg, config)

        return result_record(
            method=method,
            image_path=image_path,
            svg_path=output_svg,
            actual_mode=actual_mode,
            elapsed_sec=round(time.perf_counter() - started, 3),
            status="ok",
        )
    except Exception as exc:
        return result_record(
            method=method,
            image_path=image_path,
            svg_path=output_svg if output_svg.exists() else None,
            actual_mode=None,
            elapsed_sec=round(time.perf_counter() - started, 3),
            status="error",
            error_message=str(exc),
        )
