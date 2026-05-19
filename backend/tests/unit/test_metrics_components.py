import pytest

from app.utils.metrics import (
    compute_closed_path_score,
    compute_color_consistency_score,
    compute_file_size_score,
    compute_layer_structure_score,
    compute_named_layer_score,
    compute_node_efficiency_score,
    compute_path_complexity_score,
)


def test_compute_named_layer_score_all_named():
    assert compute_named_layer_score(5, 5) == 1.0


def test_compute_named_layer_score_half_named():
    assert compute_named_layer_score(2, 4) == 0.5


def test_compute_named_layer_score_zero_layers():
    assert compute_named_layer_score(0, 0) == 0.0


def test_compute_closed_path_score():
    assert compute_closed_path_score(3, 3) == 1.0
    assert compute_closed_path_score(1, 4) == 0.25
    assert compute_closed_path_score(0, 0) == 0.0


@pytest.mark.parametrize(
    ("node_count", "expected"),
    [
        # Brackets: <=300, <=1000, <=2500, <=5000, <=10000, >10000.
        (100, 1.0),
        (500, 0.90),
        (2000, 0.72),
        (4000, 0.50),
        (8000, 0.35),
        (15000, 0.20),
    ],
)
def test_compute_node_efficiency_score_brackets(node_count, expected):
    assert compute_node_efficiency_score(node_count) == expected


@pytest.mark.parametrize(
    ("layer_count", "expected"),
    [(0, 0.0), (1, 0.75), (5, 1.0), (10, 0.80), (15, 0.55), (25, 0.35)],
)
def test_compute_layer_structure_score_brackets(layer_count, expected):
    assert compute_layer_structure_score(layer_count) == expected


@pytest.mark.parametrize(
    ("file_size", "expected"),
    [
        # Brackets are <=50KB, <=150KB, <=500KB, <=1000KB, <=2500KB, >2500KB.
        (1024, 1.0),
        (30 * 1024, 1.0),
        (200 * 1024, 0.68),
        (800 * 1024, 0.45),
        (3_000_000, 0.18),
    ],
)
def test_compute_file_size_score_brackets(file_size, expected):
    assert compute_file_size_score(file_size) == expected


def test_compute_color_consistency_score_empty():
    assert compute_color_consistency_score([]) == 0.0


def test_compute_color_consistency_score_mono():
    assert compute_color_consistency_score([1, 1, 1]) == 1.0


def test_compute_path_complexity_score_zero_paths():
    assert compute_path_complexity_score(0, 0, []) == 0.0
