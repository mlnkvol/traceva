import argparse
import concurrent.futures
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tabulate import tabulate

from eval.baselines import RESULT_KEYS, result_record
from eval.baselines import manual_baseline, opencv_kmeans_baseline, potrace_baseline, traceva_baseline
from eval.config import BACKEND_DIR, BenchmarkConfig, DATA_DIR, RESULTS_DIR


if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


CSV_COLUMNS = [
    "image",
    "image_category",
    "method",
    "actual_mode",
    "elapsed_sec",
    "status",
    "error_message",
    "svg_path",
    "node_count",
    "layer_count",
    "path_count",
    "closed_path_count",
    "named_layer_count",
    "file_size",
    "editability_score",
    "named_layers_score",
    "closed_paths_score",
    "node_efficiency_score",
    "layer_structure_score",
    "color_consistency_score",
    "path_complexity_score",
    "file_size_score",
    "gapless_coverage",
    "gap_pixels",
    "cad_readiness_score",
    "psnr",
    "ssim",
]


def _scan_images(image_dir: Path, extensions: tuple[str, ...], max_images: int | None) -> list[Path]:
    images = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return images[:max_images] if max_images else images


def _category_for(image_path: Path, image_dir: Path) -> str:
    try:
        rel = image_path.relative_to(image_dir)
    except ValueError:
        return "uncategorized"
    if len(rel.parts) > 1 and rel.parts[0] in {"logos", "illustrations", "photos"}:
        return rel.parts[0]
    return "uncategorized"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _output_svg_for(config: BenchmarkConfig, image_path: Path, method: str) -> Path:
    category = _category_for(image_path, config.image_dir)
    return config.output_dir / "svgs" / category / method / f"{image_path.stem}.svg"


def _run_method_job(args: tuple[str, str, str, dict[str, Any]]) -> dict:
    method, image_str, output_str, config_values = args
    image_path = Path(image_str)
    output_svg = Path(output_str)
    config = BenchmarkConfig(**config_values)

    if method.startswith("traceva_"):
        mode = method.replace("traceva_", "", 1)
        return traceva_baseline.run(image_path, output_svg, mode, config)
    if method == "potrace":
        return potrace_baseline.run(image_path, output_svg, config)
    if method == "opencv_kmeans":
        return opencv_kmeans_baseline.run(image_path, output_svg, config)
    if method.startswith("manual:"):
        manual_method = method.split(":", 1)[1]
        return manual_baseline.run(image_path, output_svg, manual_method, config)

    return result_record(
        method=method,
        image_path=image_path,
        svg_path=None,
        actual_mode=None,
        elapsed_sec=None,
        status="error",
        error_message=f"Unknown method: {method}",
    )


def _run_with_timeout(method: str, image_path: Path, output_svg: Path, config: BenchmarkConfig) -> dict:
    config_values = {
        "image_dir": config.image_dir,
        "output_dir": config.output_dir,
        "methods": config.methods,
        "image_extensions": config.image_extensions,
        "timeout_per_image_sec": config.timeout_per_image_sec,
        "traceva_tolerance": config.traceva_tolerance,
        "traceva_max_layers": config.traceva_max_layers,
        "potrace_turdsize": config.potrace_turdsize,
        "potrace_alphamax": config.potrace_alphamax,
        "opencv_kmeans_k": config.opencv_kmeans_k,
        "vlm_enabled": config.vlm_enabled,
    }
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
    future = executor.submit(_run_method_job, (method, str(image_path), str(output_svg), config_values))
    try:
        return future.result(timeout=config.timeout_per_image_sec)
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False, cancel_futures=True)
        return result_record(
            method=method,
            image_path=image_path,
            svg_path=None,
            actual_mode=None,
            elapsed_sec=float(config.timeout_per_image_sec),
            status="timeout",
            error_message=f"Timed out after {config.timeout_per_image_sec}s",
        )
    finally:
        if not future.cancelled() and future.done():
            executor.shutdown(wait=True, cancel_futures=True)


def _metric_values(image_path: Path, svg_path: str | None) -> dict:
    if not svg_path:
        return {}

    path = Path(svg_path)
    if not path.exists():
        return {}

    try:
        from app.utils.metrics import compute_all_metrics

        metrics = compute_all_metrics(image_path, path)
        components = metrics.get("editability_components") or {}
        return {
            "node_count": metrics.get("node_count"),
            "layer_count": metrics.get("layer_count"),
            "path_count": metrics.get("path_count"),
            "closed_path_count": metrics.get("closed_path_count"),
            "named_layer_count": metrics.get("named_layer_count"),
            "file_size": metrics.get("file_size"),
            "editability_score": metrics.get("editability_score"),
            "named_layers_score": components.get("named_layers"),
            "closed_paths_score": components.get("closed_paths"),
            "node_efficiency_score": components.get("node_efficiency"),
            "layer_structure_score": components.get("layer_structure"),
            "color_consistency_score": components.get("color_consistency"),
            "path_complexity_score": components.get("path_complexity"),
            "file_size_score": components.get("file_size"),
            "gapless_coverage": metrics.get("gapless_coverage"),
            "gap_pixels": metrics.get("gap_pixels"),
            "cad_readiness_score": metrics.get("cad_readiness_score"),
            "psnr": metrics.get("psnr"),
            "ssim": metrics.get("ssim"),
        }
    except Exception as exc:
        return {"error_message": f"metric computation failed: {exc}"}


def _complete_row(image_path: Path, method_result: dict, config: BenchmarkConfig) -> dict:
    row = {key: None for key in CSV_COLUMNS}
    row.update(method_result)
    row["image"] = row.get("image") or image_path.name
    row["image_category"] = _category_for(image_path, config.image_dir)
    row["method"] = row.get("method") or "unknown"
    row["status"] = row.get("status") or "error"

    if row["status"] == "ok":
        metric_values = _metric_values(image_path, row.get("svg_path"))
        if metric_values.get("error_message") and not row.get("error_message"):
            row["error_message"] = metric_values.pop("error_message")
        row.update(metric_values)

    return {key: row.get(key) for key in CSV_COLUMNS}


def _print_aggregate(df: pd.DataFrame) -> None:
    if df.empty:
        return

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("No successful SVG results to aggregate.")
        return

    summary = (
        ok.groupby("method", dropna=False)
        .agg(
            editability_mean=("editability_score", "mean"),
            editability_median=("editability_score", "median"),
            psnr_mean=("psnr", "mean"),
            ssim_mean=("ssim", "mean"),
            elapsed_mean=("elapsed_sec", "mean"),
            images=("image", "count"),
        )
        .reset_index()
    )
    print(tabulate(summary, headers="keys", tablefmt="github", showindex=False, floatfmt=".3f"))


def run_benchmark(config: BenchmarkConfig, max_images: int | None = None) -> Path | None:
    os.environ["VLM_LAYER_NAMING_ENABLED"] = "true" if config.vlm_enabled else "false"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    images = _scan_images(config.image_dir, config.image_extensions, max_images)

    if not images:
        print(f"No images found in {config.image_dir}")
        return None

    methods = list(config.methods)
    for manual_method in manual_baseline.discover_methods(config.image_dir):
        methods.append(f"manual:{manual_method}")

    rows = []
    for image_path in images:
        print(f"Image: {image_path.name}")
        for method in methods:
            display_method = method.replace("manual:", "")
            output_svg = _output_svg_for(config, image_path, _safe_name(display_method))
            print(f"  - {display_method}")
            result = _run_with_timeout(method, image_path, output_svg, config)
            if method.startswith("manual:"):
                result["method"] = method.split(":", 1)[1]
            rows.append(_complete_row(image_path, result, config))

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df["image"] = df["image"].fillna("")
    df["method"] = df["method"].fillna("")
    df["status"] = df["status"].fillna("error")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = config.output_dir / f"benchmark_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    _print_aggregate(df)
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Traceva benchmark suite.")
    parser.add_argument("--image-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods: traceva_auto traceva_logo traceva_semantic potrace opencv_kmeans",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--timeout-per-image-sec", type=int, default=180)
    parser.add_argument("--traceva-tolerance", type=float, default=1.0)
    parser.add_argument("--traceva-max-layers", type=int, default=10)
    parser.add_argument("--potrace-turdsize", type=int, default=2)
    parser.add_argument("--potrace-alphamax", type=float, default=1.0)
    parser.add_argument("--opencv-kmeans-k", type=int, default=6)
    vlm_group = parser.add_mutually_exclusive_group()
    vlm_group.add_argument("--vlm-enabled", dest="vlm_enabled", action="store_true")
    vlm_group.add_argument("--no-vlm-enabled", dest="vlm_enabled", action="store_false")
    parser.set_defaults(vlm_enabled=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        methods=args.methods or BenchmarkConfig().methods,
        timeout_per_image_sec=args.timeout_per_image_sec,
        traceva_tolerance=args.traceva_tolerance,
        traceva_max_layers=args.traceva_max_layers,
        potrace_turdsize=args.potrace_turdsize,
        potrace_alphamax=args.potrace_alphamax,
        opencv_kmeans_k=args.opencv_kmeans_k,
        vlm_enabled=args.vlm_enabled,
    )
    run_benchmark(config, max_images=args.max_images)


if __name__ == "__main__":
    main()
