from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def deterministic_model_settings(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "VLM_LAYER_NAMING_ENABLED", False, raising=False)


@pytest.fixture
def simple_binary_mask():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    return mask


@pytest.fixture
def mask_with_hole():
    mask = np.zeros((120, 120), dtype=bool)
    mask[20:100, 20:100] = True
    mask[45:75, 45:75] = False
    return mask


@pytest.fixture
def synthetic_logo_image(tmp_path):
    path = tmp_path / "synthetic-logo.png"
    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 40, 160, 160), fill="black")
    image.save(path)
    return path


@pytest.fixture
def synthetic_photo_image(tmp_path):
    path = tmp_path / "synthetic-photo.png"
    width = height = 256
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = (60 + 120 * x).astype(np.uint8)
    image[:, :, 1] = (90 + 90 * y).astype(np.uint8)
    image[:, :, 2] = (180 - 80 * x).astype(np.uint8)

    pil = Image.fromarray(image, "RGB")
    draw = ImageDraw.Draw(pil, "RGBA")
    draw.ellipse((28, 42, 142, 162), fill=(235, 64, 82, 210))
    draw.ellipse((112, 72, 228, 196), fill=(49, 178, 112, 210))
    draw.ellipse((64, 132, 174, 242), fill=(83, 116, 226, 210))
    pil.save(path)
    return path


@pytest.fixture
def sample_named_svg(tmp_path):
    target = tmp_path / "named_layers.svg"
    shutil.copyfile(FIXTURES_DIR / "svg" / "named_layers.svg", target)
    return target


@pytest.fixture
def sample_unnamed_svg(tmp_path):
    target = tmp_path / "unnamed_layers.svg"
    shutil.copyfile(FIXTURES_DIR / "svg" / "unnamed_layers.svg", target)
    return target


class FakeSegmentationService:
    def segment(self, image: np.ndarray, max_layers: int = 10):
        h, w = image.shape[:2]
        masks = []
        regions = [
            ((slice(None), slice(None)), "#dddddd", "background"),
            ((slice(0, h // 2), slice(0, w // 2)), "#cc3344", "red shape"),
            ((slice(h // 3, None), slice(w // 3, None)), "#338855", "green shape"),
        ]
        for region, color, label in regions[:max_layers]:
            mask = np.zeros((h, w), dtype=bool)
            mask[region] = True
            masks.append(
                {
                    "mask": mask,
                    "label": label,
                    "color": color,
                    "area": int(mask.sum()),
                }
            )
        return masks

    def generate_preview_masks(self, image: np.ndarray, max_layers: int = 24):
        return self.segment(image, max_layers=max(1, min(max_layers, 3)))[1:]

    def analyze_image(self, image: np.ndarray):
        return {
            "mode": "photo",
            "tolerance": 0.5,
            "max_layers": 48,
            "reason": "deterministic test recommendation",
            "tips": [],
        }


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.core.config import settings
    import app.api.routes as routes
    from app.main import app

    upload_dir = tmp_path / "uploads"
    results_dir = tmp_path / "results"
    checkpoints_dir = tmp_path / "checkpoints"
    upload_dir.mkdir()
    results_dir.mkdir()
    checkpoints_dir.mkdir()

    monkeypatch.setattr(settings, "UPLOAD_DIR", upload_dir, raising=False)
    monkeypatch.setattr(settings, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(settings, "CHECKPOINTS_DIR", checkpoints_dir, raising=False)
    monkeypatch.setattr(routes, "get_seg_service", lambda: FakeSegmentationService())
    monkeypatch.setattr(routes.settings, "VLM_LAYER_NAMING_ENABLED", False, raising=False)
    routes.tasks.clear()

    with TestClient(app) as test_client:
        yield test_client

    routes.tasks.clear()


def wait_for_task(client, task_id: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    last_payload = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/status/{task_id}")
        response.raise_for_status()
        last_payload = response.json()
        if last_payload["status"] in {"done", "error"}:
            return last_payload
        time.sleep(0.05)
    return last_payload
