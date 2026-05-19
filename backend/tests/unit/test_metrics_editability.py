import builtins

import numpy as np
from PIL import Image

from app.services.svg_builder import SVGBuilder
from app.utils.metrics import (
    compute_all_metrics,
    compute_editability_score,
    count_named_layers,
    count_svg_layers,
    count_svg_nodes,
    count_svg_paths,
    get_file_size,
)


def test_count_svg_nodes_on_fixture(sample_named_svg):
    assert count_svg_nodes(sample_named_svg) == 12


def test_count_svg_layers_on_named(sample_named_svg):
    assert count_svg_layers(sample_named_svg) == 3


def test_count_named_layers(sample_named_svg, sample_unnamed_svg):
    assert count_named_layers(sample_named_svg) == count_svg_layers(sample_named_svg)
    assert count_named_layers(sample_unnamed_svg) == 0


def test_editability_score_in_range(tmp_path):
    svg_path = tmp_path / "built.svg"
    contour = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.int32)
    SVGBuilder().build(
        [{"label": "Square", "color": "#111111", "contours": [contour], "prefer_contours": True}],
        width=100,
        height=100,
        output_path=svg_path,
    )

    score, components = compute_editability_score(
        svg_path,
        node_count=count_svg_nodes(svg_path),
        layer_count=count_svg_layers(svg_path),
        path_count=count_svg_paths(svg_path),
        file_size=get_file_size(svg_path),
    )

    assert 0.0 <= score <= 100.0
    assert all(0.0 <= value <= 100.0 for value in components.values())


def test_editability_score_named_beats_unnamed(sample_named_svg, sample_unnamed_svg):
    named_score, _ = compute_editability_score(
        sample_named_svg,
        count_svg_nodes(sample_named_svg),
        count_svg_layers(sample_named_svg),
        count_svg_paths(sample_named_svg),
        get_file_size(sample_named_svg),
    )
    unnamed_score, _ = compute_editability_score(
        sample_unnamed_svg,
        count_svg_nodes(sample_unnamed_svg),
        count_svg_layers(sample_unnamed_svg),
        count_svg_paths(sample_unnamed_svg),
        get_file_size(sample_unnamed_svg),
    )

    assert named_score > unnamed_score


def test_compute_all_metrics_psnr_ssim_optional(tmp_path, sample_named_svg, monkeypatch):
    original = tmp_path / "original.png"
    Image.new("RGB", (100, 100), "white").save(original)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cairosvg":
            raise ImportError("disabled in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    metrics = compute_all_metrics(original, sample_named_svg, width=100, height=100)

    assert metrics["psnr"] is None
    assert metrics["ssim"] is None
    assert metrics["node_count"] == 12
    assert metrics["layer_count"] == 3
