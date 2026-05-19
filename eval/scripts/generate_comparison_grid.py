import argparse
import io
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from eval.config import DATA_DIR, RESULTS_DIR


def _latest_csv(results_dir: Path) -> Path | None:
    files = sorted(results_dir.glob("benchmark_*.csv"))
    return files[-1] if files else None


def _find_original(image_name: str, image_dir: Path) -> Path | None:
    matches = list(image_dir.rglob(image_name))
    return matches[0] if matches else None


def _render_svg(svg_path: Path, size: tuple[int, int]) -> Image.Image:
    try:
        import cairosvg

        png = cairosvg.svg2png(
            url=str(svg_path),
            output_width=size[0],
            output_height=size[1],
        )
        return Image.open(io.BytesIO(png)).convert("RGB")
    except Exception:
        placeholder = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(placeholder)
        draw.text((12, 12), "preview unavailable", fill=(80, 80, 80))
        return placeholder


def _fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _tile(label: str, image: Image.Image, tile_size: tuple[int, int]) -> Image.Image:
    header_h = 28
    tile = Image.new("RGB", (tile_size[0], tile_size[1] + header_h), "white")
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, tile.width - 1, tile.height - 1), outline=(220, 220, 220))
    draw.text((8, 7), label[:42], fill=(20, 20, 20), font=ImageFont.load_default())
    tile.paste(image, (0, header_h))
    return tile


def generate_grids(
    csv_path: Path,
    image_dir: Path,
    results_dir: Path,
    per_category: int,
    tile_width: int,
) -> list[Path]:
    df = pd.read_csv(csv_path)
    outputs: list[Path] = []
    tile_size = (tile_width, tile_width)

    for category, category_df in df.groupby("image_category", dropna=False):
        images = list(dict.fromkeys(category_df["image"].dropna().tolist()))[:per_category]
        for image_name in images:
            original = _find_original(image_name, image_dir)
            if original is None:
                continue

            image_rows = category_df[(category_df["image"] == image_name) & (category_df["status"] == "ok")]
            tiles = [_tile("original", _fit_image(original, tile_size), tile_size)]

            for _, row in image_rows.iterrows():
                svg_path = Path(str(row.get("svg_path", "")))
                if not svg_path.exists():
                    continue
                preview = _render_svg(svg_path, tile_size)
                tiles.append(_tile(str(row["method"]), preview, tile_size))

            if len(tiles) <= 1:
                continue

            grid = Image.new("RGB", (tile_size[0] * len(tiles), tile_size[1] + 28), "white")
            for index, tile in enumerate(tiles):
                grid.paste(tile, (index * tile_size[0], 0))

            output = results_dir / f"grid_{Path(image_name).stem}.png"
            grid.save(output)
            outputs.append(output)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PNG comparison grids.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--image-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=260)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv or _latest_csv(args.results_dir)
    if csv_path is None:
        print(f"No benchmark CSV found in {args.results_dir}")
        return

    outputs = generate_grids(
        csv_path=csv_path,
        image_dir=args.image_dir,
        results_dir=args.results_dir,
        per_category=args.per_category,
        tile_width=args.tile_width,
    )
    if outputs:
        for output in outputs:
            print(f"Saved grid: {output}")
    else:
        print("No grids generated.")


if __name__ == "__main__":
    main()
