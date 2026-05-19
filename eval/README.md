# Traceva Evaluation Benchmark

This folder contains an offline benchmark runner for chapter 4 experiments. It compares Traceva modes against simple baselines and writes reproducible CSV/preview artifacts.

## Data Layout

Put private test images manually into:

```text
eval/data/
  logos/
  illustrations/
  photos/
```

Images directly under `eval/data/` are assigned category `uncategorized`. Supported extensions are `.png`, `.jpg`, `.jpeg`, and `.webp`.

Manual results from tools such as Vectorizer.AI or Adobe Illustrator Image Trace can be included without API keys:

```text
eval/data/manual/
  vectorizer_ai/
    image1.svg
  adobe_image_trace/
    image1.svg
```

The benchmark matches manual SVGs by the original image stem. For `photo01.png`, place `photo01.svg`.

## Install

Python dependencies live in `backend/requirements.txt`.

```bash
cd backend
python -m pip install -r requirements.txt
```

Potrace is optional and must be installed as a host CLI:

```bash
# Linux
sudo apt install potrace

# macOS
brew install potrace
```

If `potrace` is not found, the CSV row is written with `status=skipped`.

## Run Benchmark

From the repository root:

```bash
python -m eval.run_benchmark \
  --image-dir eval/data \
  --output-dir eval/results \
  --methods traceva_auto traceva_logo traceva_semantic potrace opencv_kmeans \
  --max-images 30
```

On empty data, the script exits successfully and prints `No images found in ...`.

Outputs:

```text
eval/results/
  benchmark_<timestamp>.csv
  svgs/<category>/<method>/<image>.svg
```

## CSV Fields

Core fields:

- `image`, `image_category`, `method`, `actual_mode`
- `elapsed_sec`, `status`, `error_message`, `svg_path`

SVG/editability fields:

- `node_count`, `layer_count`, `path_count`, `closed_path_count`, `named_layer_count`, `file_size`
- `editability_score`
- `named_layers_score`, `closed_paths_score`, `node_efficiency_score`, `layer_structure_score`
- `color_consistency_score`, `path_complexity_score`, `file_size_score`

Raster comparison fields:

- `gapless_coverage`, `gap_pixels`, `cad_readiness_score`
- `psnr`, `ssim`

Rows with `status=error`, `timeout`, or `skipped` may have empty metric fields.

## Generate Summary

```bash
python -m eval.scripts.generate_summary --results-dir eval/results
```

This reads the latest benchmark CSV, prints a markdown table, and writes:

```text
eval/results/summary_<timestamp>.md
```

The table aggregates average editability, PSNR, SSIM, and elapsed time by method and category.

## Generate Comparison Grids

```bash
python -m eval.scripts.generate_comparison_grid \
  --image-dir eval/data \
  --results-dir eval/results \
  --per-category 3
```

This creates PNG grids with the original image plus rendered SVG previews for each successful method:

```text
eval/results/grid_<image_name>.png
```

SVG rendering uses CairoSVG. If rendering fails for one SVG, that cell becomes a placeholder instead of failing the whole script.

## Notes

- The benchmark is offline and does not call FastAPI.
- Processing is intentionally sequential to keep timings comparable.
- SAM2 and VLM are optional. Traceva semantic mode falls back through the existing backend segmentation service when SAM2 weights are unavailable.
- Do not commit files in `eval/data/` or generated files in `eval/results/`.
