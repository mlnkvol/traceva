import numpy as np

from app.core.config import settings
from app.services.layer_namer import LayerNamer


def test_fallback_names_basic(monkeypatch):
    monkeypatch.setattr(settings, "VLM_LAYER_NAMING_ENABLED", False, raising=False)

    names = LayerNamer().name_layers(np.zeros((10, 10, 3), dtype=np.uint8), [{"area": 1000}, {"area": 500}])

    assert names == ["Background", "Object 1"]


def test_dedupe_names_unique():
    assert LayerNamer()._dedupe_names(["cat", "dog", "fish"]) == ["cat", "dog", "fish"]


def test_dedupe_names_collisions():
    assert LayerNamer()._dedupe_names(["cat", "cat", "cat"]) == ["cat", "cat 2", "cat 3"]


def test_clean_name_strips_prefixes():
    assert LayerNamer()._clean_name("the image shows a cat") == "cat"


def test_clean_name_limits_words():
    name = LayerNamer()._clean_name("very long descriptive caption with many words")

    assert name is not None
    assert len(name.split()) <= 2
    assert len(name) <= 48


def test_clean_name_filters_useless():
    assert LayerNamer()._clean_name("object") is None


def test_name_layers_with_vlm_disabled(monkeypatch):
    monkeypatch.setattr(settings, "VLM_LAYER_NAMING_ENABLED", False, raising=False)

    masks = [{"area": 1000}, {"area": 500}, {"area": 0}]
    names = LayerNamer().name_layers(np.zeros((10, 10, 3), dtype=np.uint8), masks)

    assert names == ["Background", "Object 1", "Layer 3"]
