import shutil
import subprocess
import time
from pathlib import Path

from PIL import Image

from eval.baselines import result_record
from eval.config import BenchmarkConfig


def run(image_path: Path, output_svg: Path, config: BenchmarkConfig) -> dict:
    method = "potrace"
    started = time.perf_counter()
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("potrace") is None:
        return result_record(
            method=method,
            image_path=image_path,
            svg_path=None,
            actual_mode="potrace",
            elapsed_sec=0.0,
            status="skipped",
            error_message="potrace CLI not found",
        )

    pbm_path = output_svg.with_suffix(".pbm")
    try:
        image = Image.open(image_path).convert("L")
        image = image.point(lambda value: 255 if value > 128 else 0, mode="1")
        image.save(pbm_path)

        subprocess.run(
            [
                "potrace",
                "-s",
                "-t",
                str(config.potrace_turdsize),
                "-a",
                str(config.potrace_alphamax),
                "-o",
                str(output_svg),
                str(pbm_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=config.timeout_per_image_sec,
        )
        return result_record(
            method=method,
            image_path=image_path,
            svg_path=output_svg,
            actual_mode="potrace",
            elapsed_sec=round(time.perf_counter() - started, 3),
            status="ok",
        )
    except subprocess.TimeoutExpired:
        return result_record(
            method=method,
            image_path=image_path,
            svg_path=None,
            actual_mode="potrace",
            elapsed_sec=round(time.perf_counter() - started, 3),
            status="timeout",
            error_message=f"Timed out after {config.timeout_per_image_sec}s",
        )
    except Exception as exc:
        return result_record(
            method=method,
            image_path=image_path,
            svg_path=output_svg if output_svg.exists() else None,
            actual_mode="potrace",
            elapsed_sec=round(time.perf_counter() - started, 3),
            status="error",
            error_message=str(exc),
        )
    finally:
        try:
            pbm_path.unlink(missing_ok=True)
        except Exception:
            pass
