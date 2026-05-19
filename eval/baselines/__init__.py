from pathlib import Path
from typing import Optional


RESULT_KEYS = [
    "method",
    "image",
    "svg_path",
    "actual_mode",
    "elapsed_sec",
    "status",
    "error_message",
]


def result_record(
    *,
    method: str,
    image_path: Path,
    svg_path: Optional[Path],
    actual_mode: Optional[str],
    elapsed_sec: Optional[float],
    status: str,
    error_message: Optional[str] = None,
) -> dict:
    return {
        "method": method,
        "image": image_path.name,
        "svg_path": str(svg_path) if svg_path else None,
        "actual_mode": actual_mode,
        "elapsed_sec": elapsed_sec,
        "status": status,
        "error_message": error_message,
    }
