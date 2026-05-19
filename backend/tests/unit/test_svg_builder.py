import numpy as np
from lxml import etree

from app.services.svg_builder import INKSCAPE_NS, SVGBuilder


SVG_NS = "http://www.w3.org/2000/svg"


def _parse(path):
    return etree.parse(str(path))


def test_build_empty_layers_produces_valid_svg(tmp_path):
    output = tmp_path / "empty.svg"

    SVGBuilder().build([], width=100, height=100, output_path=output)

    root = _parse(output).getroot()
    assert root.tag == f"{{{SVG_NS}}}svg"


def test_build_with_one_layer(tmp_path):
    output = tmp_path / "one.svg"
    contour = np.array([[5, 5], [45, 5], [45, 45], [5, 45]], dtype=np.int32)

    SVGBuilder().build(
        [{"label": "Square", "color": "#123456", "contours": [contour], "prefer_contours": True}],
        width=50,
        height=50,
        output_path=output,
    )

    tree = _parse(output)
    groups = tree.findall(f".//{{{SVG_NS}}}g")
    paths = groups[0].findall(f".//{{{SVG_NS}}}path")
    assert len(groups) == 1
    assert len(paths) >= 1
    assert paths[0].get("d").startswith("M")


def test_build_inkscape_labels_attached(tmp_path):
    output = tmp_path / "labels.svg"
    contour = np.array([[5, 5], [45, 5], [45, 45], [5, 45]], dtype=np.int32)

    SVGBuilder().build(
        [{"label": "Main Object", "color": "#123456", "contours": [contour], "prefer_contours": True}],
        width=50,
        height=50,
        output_path=output,
    )

    group = _parse(output).find(f".//{{{SVG_NS}}}g")
    assert group.get(f"{{{INKSCAPE_NS}}}label") == "Main Object"


def test_build_safe_id_no_spaces(tmp_path):
    output = tmp_path / "safe-id.svg"
    contour = np.array([[5, 5], [45, 5], [45, 45], [5, 45]], dtype=np.int32)

    SVGBuilder().build(
        [{"label": "Main Object", "color": "#123456", "contours": [contour], "prefer_contours": True}],
        width=50,
        height=50,
        output_path=output,
    )

    group_id = _parse(output).find(f".//{{{SVG_NS}}}g").get("id")
    assert " " not in group_id
    assert "main-object" in group_id


def test_build_with_background_color(tmp_path):
    output = tmp_path / "background.svg"

    SVGBuilder().build([], width=50, height=50, output_path=output, background_color="#abcdef")

    rect = _parse(output).find(f".//{{{SVG_NS}}}rect")
    assert rect.get("id") == "gapless-underlay"
    assert rect.get("fill") == "#abcdef"


def test_build_with_bezier_segments(tmp_path):
    output = tmp_path / "bezier.svg"
    segment = (
        np.array([0.0, 0.0]),
        np.array([10.0, 0.0]),
        np.array([10.0, 20.0]),
        np.array([20.0, 20.0]),
    )

    SVGBuilder().build(
        [{"label": "Curve", "color": "#000000", "bezier_segs": [[segment]]}],
        width=50,
        height=50,
        output_path=output,
    )

    path = _parse(output).find(f".//{{{SVG_NS}}}path")
    assert "C" in path.get("d")
