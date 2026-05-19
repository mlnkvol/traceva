import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from eval.config import RESULTS_DIR


def _latest_csv(results_dir: Path) -> Path | None:
    files = sorted(results_dir.glob("benchmark_*.csv"))
    return files[-1] if files else None


def generate_summary(csv_path: Path, results_dir: Path) -> Path:
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"].copy()

    if ok.empty:
        table = pd.DataFrame(
            columns=[
                "image_category",
                "method",
                "editability_score_mean",
                "psnr_mean",
                "ssim_mean",
                "elapsed_sec_mean",
                "n",
            ]
        )
    else:
        table = (
            ok.groupby(["image_category", "method"], dropna=False)
            .agg(
                editability_score_mean=("editability_score", "mean"),
                psnr_mean=("psnr", "mean"),
                ssim_mean=("ssim", "mean"),
                elapsed_sec_mean=("elapsed_sec", "mean"),
                n=("image", "count"),
            )
            .reset_index()
        )

    markdown = tabulate(table, headers="keys", tablefmt="github", showindex=False, floatfmt=".3f")
    print(markdown)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"summary_{timestamp}.md"
    output_path.write_text(
        f"# Benchmark Summary\n\nSource CSV: `{csv_path.name}`\n\n{markdown}\n",
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate markdown benchmark summary.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv or _latest_csv(args.results_dir)
    if csv_path is None:
        print(f"No benchmark CSV found in {args.results_dir}")
        return

    output = generate_summary(csv_path, args.results_dir)
    print(f"Saved summary: {output}")


if __name__ == "__main__":
    main()
