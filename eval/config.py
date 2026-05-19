from dataclasses import dataclass, field
from pathlib import Path
from typing import List


EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
BACKEND_DIR = REPO_ROOT / "backend"
DATA_DIR = EVAL_ROOT / "data"
RESULTS_DIR = EVAL_ROOT / "results"


@dataclass
class BenchmarkConfig:
    image_dir: Path = DATA_DIR
    output_dir: Path = RESULTS_DIR
    methods: List[str] = field(
        default_factory=lambda: [
            "traceva_auto",
            "traceva_logo",
            "traceva_semantic",
            "potrace",
            "opencv_kmeans",
        ]
    )
    image_extensions: tuple = (".png", ".jpg", ".jpeg", ".webp")
    timeout_per_image_sec: int = 180
    traceva_tolerance: float = 1.0
    traceva_max_layers: int = 10
    potrace_turdsize: int = 2
    potrace_alphamax: float = 1.0
    opencv_kmeans_k: int = 6
    vlm_enabled: bool = False
