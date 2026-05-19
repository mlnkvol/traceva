import sys
from pathlib import Path

from eval.baselines import result_record
from eval.config import BACKEND_DIR, BenchmarkConfig, DATA_DIR


if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def discover_methods(image_dir: Path = DATA_DIR) -> list[str]:
    manual_dir = image_dir / "manual"
    if not manual_dir.exists():
        return []
    return [path.name for path in manual_dir.iterdir() if path.is_dir()]


def run(image_path: Path, output_svg: Path, method: str, config: BenchmarkConfig) -> dict:
    manual_svg = config.image_dir / "manual" / method / f"{image_path.stem}.svg"
    if not manual_svg.exists():
        return result_record(
            method=method,
            image_path=image_path,
            svg_path=None,
            actual_mode="manual",
            elapsed_sec=None,
            status="skipped",
            error_message=f"Manual SVG not found: {manual_svg}",
        )

    return result_record(
        method=method,
        image_path=image_path,
        svg_path=manual_svg,
        actual_mode="manual",
        elapsed_sec=None,
        status="ok",
    )
