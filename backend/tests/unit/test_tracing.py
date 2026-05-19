import cv2
import numpy as np

from app.services.tracing import ContourTracer


def test_trace_empty_mask_returns_empty():
    contours = ContourTracer().trace(np.zeros((100, 100), dtype=bool))

    assert contours == []


def test_trace_simple_square(simple_binary_mask):
    contours = ContourTracer().trace(simple_binary_mask)

    assert len(contours) >= 1
    assert len(contours[0]) >= 4


def test_trace_with_holes_default(mask_with_hole):
    contours = ContourTracer().trace(mask_with_hole, keep_holes=True, min_area_ratio=0.0001)

    assert len(contours) >= 2


def test_trace_with_holes_disabled(mask_with_hole):
    contours = ContourTracer().trace(mask_with_hole, keep_holes=False, min_area_ratio=0.0001)

    assert len(contours) == 1


def test_trace_respects_max_contours():
    mask = np.zeros((120, 120), dtype=bool)
    for index in range(5):
        x = 6 + index * 22
        mask[20:35, x : x + 15] = True

    contours = ContourTracer().trace(mask, keep_holes=False, max_contours=3, min_area_ratio=0.00001)

    assert len(contours) <= 3


def test_trace_simplify_reduces_points():
    mask = np.zeros((160, 160), dtype=np.uint8)
    cv2.circle(mask, (80, 80), 55, 255, -1)

    raw = ContourTracer().trace(mask, simplify=False, keep_holes=False, min_area_ratio=0.0001)
    simplified = ContourTracer().trace(
        mask,
        simplify=True,
        epsilon_factor=0.01,
        keep_holes=False,
        min_area_ratio=0.0001,
    )

    assert raw and simplified
    assert len(simplified[0]) < len(raw[0])
