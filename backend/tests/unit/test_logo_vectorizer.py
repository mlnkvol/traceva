from lxml import etree
from PIL import Image

from app.services.logo_vectorizer import LogoVectorizer


SVG_NS = "http://www.w3.org/2000/svg"


def test_logo_on_synthetic_circle(synthetic_logo_image, tmp_path):
    output = tmp_path / "logo.svg"

    result = LogoVectorizer().vectorize(synthetic_logo_image, output, tolerance=1.0)

    assert set(result) >= {"svg_path", "layers", "metrics"}
    assert len(result["layers"]) == 1
    assert result["metrics"]["layer_count"] == 1
    assert result["metrics"]["node_count"] > 0


def test_logo_output_svg_valid(synthetic_logo_image, tmp_path):
    output = tmp_path / "logo.svg"

    LogoVectorizer().vectorize(synthetic_logo_image, output, tolerance=1.0)

    tree = etree.parse(str(output))
    path = tree.find(f".//{{{SVG_NS}}}path")
    assert path is not None
    assert path.get("fill-rule") == "evenodd"


def test_logo_handles_blank_image(tmp_path):
    image_path = tmp_path / "blank.png"
    output = tmp_path / "blank.svg"
    Image.new("RGB", (100, 100), "white").save(image_path)

    result = LogoVectorizer().vectorize(image_path, output, tolerance=1.0)

    assert output.exists()
    assert result["layers"] == [] or result["metrics"]["node_count"] == 0
