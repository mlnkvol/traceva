import time
from pathlib import Path

import cv2
import numpy as np
import svgwrite

from eval.baselines import result_record
from eval.config import BenchmarkConfig


def _hex_color(bgr: np.ndarray) -> str:
    b, g, r = [int(np.clip(round(float(value)), 0, 255)) for value in bgr]
    return f"#{r:02X}{g:02X}{b:02X}"


def _contour_path(contour: np.ndarray) -> str:
    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return ""
    parts = [f"M {float(pts[0][0]):.2f},{float(pts[0][1]):.2f}"]
    for point in pts[1:]:
        parts.append(f"L {float(point[0]):.2f},{float(point[1]):.2f}")
    parts.append("Z")
    return " ".join(parts)


def run(image_path: Path, output_svg: Path, config: BenchmarkConfig) -> dict:
    method = "opencv_kmeans"
    started = time.perf_counter()
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3).astype(np.float32)
        k = int(np.clip(config.opencv_kmeans_k, 2, 32))

        _compactness, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.8),
            2,
            cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.reshape(h, w)
        centers = centers.astype(np.uint8)

        dwg = svgwrite.Drawing(
            filename=str(output_svg),
            size=(f"{w}px", f"{h}px"),
            viewBox=f"0 0 {w} {h}",
            profile="full",
        )
        bg_label = int(np.bincount(labels.reshape(-1)).argmax())
        dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill=_hex_color(centers[bg_label])))

        areas = [(label_id, int(np.count_nonzero(labels == label_id))) for label_id in range(k)]
        for label_id, _area in sorted(areas, key=lambda item: item[1], reverse=True):
            mask = (labels == label_id).astype(np.uint8) * 255
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            group = dwg.g(id=f"kmeans-layer-{label_id + 1}")
            color = _hex_color(centers[label_id])
            for contour in contours:
                if cv2.contourArea(contour) < max(12, h * w * 0.00008):
                    continue
                epsilon = 0.004 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                path_d = _contour_path(approx)
                if path_d:
                    group.add(dwg.path(d=path_d, fill=color, stroke="none"))
            if group.elements:
                dwg.add(group)

        dwg.save(pretty=True)
        return result_record(
            method=method,
            image_path=image_path,
            svg_path=output_svg,
            actual_mode="kmeans",
            elapsed_sec=round(time.perf_counter() - started, 3),
            status="ok",
        )
    except Exception as exc:
        return result_record(
            method=method,
            image_path=image_path,
            svg_path=output_svg if output_svg.exists() else None,
            actual_mode="kmeans",
            elapsed_sec=round(time.perf_counter() - started, 3),
            status="error",
            error_message=str(exc),
        )
