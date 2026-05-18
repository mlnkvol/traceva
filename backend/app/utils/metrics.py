import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from lxml import etree


logger = logging.getLogger(__name__)

SVG_NS = "{http://www.w3.org/2000/svg}"


# ══════════════════════════════════════════════════════════════
# SAFE HELPERS
# ══════════════════════════════════════════════════════════════

def _safe_float(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None

    if math.isnan(value) or math.isinf(value):
        return None

    return round(float(value), digits)


def _parse_svg(svg_path: Path) -> etree._ElementTree:
    parser = etree.XMLParser(recover=True)
    return etree.parse(str(svg_path), parser)


def _get_svg_elements(svg_path: Path) -> Tuple[List[Any], List[Any]]:
    """
    Повертає SVG-групи <g> та path-елементи <path>.
    Підтримує namespace і fallback без namespace.
    """
    tree = _parse_svg(svg_path)

    groups = tree.findall(f".//{SVG_NS}g")
    paths = tree.findall(f".//{SVG_NS}path")

    if not groups:
        groups = tree.findall(".//g")

    if not paths:
        paths = tree.findall(".//path")

    return groups, paths


def _get_group_paths(group: Any) -> List[Any]:
    """
    Повертає path-елементи всередині конкретного <g>.
    """
    paths = group.findall(f".//{SVG_NS}path")

    if not paths:
        paths = group.findall(".//path")

    return paths


# ══════════════════════════════════════════════════════════════
# BASIC SVG COUNTERS
# ══════════════════════════════════════════════════════════════

def count_svg_nodes(svg_path: Path) -> int:
    """
    Рахує приблизну кількість вузлів у SVG.

    Для дипломної:
    менше вузлів при збереженні структури = краща редагованість.
    """
    _, paths = _get_svg_elements(svg_path)

    total = 0
    command_pattern = re.compile(r"[MLCQSTAmlcqsta]")

    for path_el in paths:
        d = path_el.get("d", "") or ""
        total += len(command_pattern.findall(d))

    return int(total)


def count_svg_paths(svg_path: Path) -> int:
    """
    Рахує кількість path-елементів у SVG.
    """
    _, paths = _get_svg_elements(svg_path)
    return int(len(paths))


def count_svg_layers(svg_path: Path) -> int:
    """
    Рахує кількість логічних SVG-шарів.

    Основний варіант — кількість <g>.
    Якщо <g> немає, але є <path>, вважаємо це 1 шаром.
    """
    groups, paths = _get_svg_elements(svg_path)

    if len(groups) > 0:
        return int(len(groups))

    if len(paths) > 0:
        return 1

    return 0


def get_file_size(svg_path: Path) -> int:
    """
    Розмір SVG-файлу в байтах.
    """
    if not svg_path.exists():
        return 0

    return int(svg_path.stat().st_size)


def count_closed_paths(svg_path: Path) -> int:
    """
    Рахує кількість path, які завершуються командою Z/z.
    """
    _, paths = _get_svg_elements(svg_path)

    closed = 0

    for path_el in paths:
        d = (path_el.get("d", "") or "").strip()

        if d.endswith("Z") or d.endswith("z"):
            closed += 1

    return closed


def count_named_layers(svg_path: Path) -> int:
    """
    Рахує кількість іменованих шарів.

    Для Liniq очікуваний формат:
    <g id="layer-..." label="...">
    """
    groups, _ = _get_svg_elements(svg_path)

    named = 0

    for group in groups:
        group_id = group.get("id", "") or ""
        label = group.get("label", "") or ""

        if group_id.startswith("layer-") or label.strip():
            named += 1

    return named


def extract_layer_color_counts(svg_path: Path) -> List[int]:
    """
    Для кожного шару рахує кількість унікальних fill-кольорів.

    Хороший редагований шар зазвичай має 1 основний fill.
    """
    groups, paths = _get_svg_elements(svg_path)

    color_counts: List[int] = []

    if groups:
        for group in groups:
            layer_paths = _get_group_paths(group)
            colors = set()

            for path_el in layer_paths:
                color = path_el.get("fill", "") or ""
                if color:
                    colors.add(color)

            if layer_paths:
                color_counts.append(len(colors))
    else:
        colors = set()

        for path_el in paths:
            color = path_el.get("fill", "") or ""
            if color:
                colors.add(color)

        if paths:
            color_counts.append(len(colors))

    return color_counts


def get_nodes_per_layer(svg_path: Path) -> List[int]:
    """
    Рахує приблизну кількість вузлів у кожному шарі.
    """
    groups, paths = _get_svg_elements(svg_path)
    command_pattern = re.compile(r"[MLCQSTAmlcqsta]")

    nodes_per_layer: List[int] = []

    if groups:
        for group in groups:
            layer_paths = _get_group_paths(group)

            layer_nodes = 0

            for path_el in layer_paths:
                d = path_el.get("d", "") or ""
                layer_nodes += len(command_pattern.findall(d))

            if layer_paths:
                nodes_per_layer.append(layer_nodes)
    else:
        total = 0

        for path_el in paths:
            d = path_el.get("d", "") or ""
            total += len(command_pattern.findall(d))

        if paths:
            nodes_per_layer.append(total)

    return nodes_per_layer


# ══════════════════════════════════════════════════════════════
# SVG RENDERING + IMAGE METRICS
# ══════════════════════════════════════════════════════════════

def _render_svg_to_image(
    svg_path: Path,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    """
    Пробує рендерити SVG у raster image.

    Якщо cairosvg не встановлений або рендер впав,
    повертаємо None. Не підміняємо rendered = original,
    бо це дасть фейкові PSNR=100 і SSIM=1.000.
    """
    try:
        import cairosvg
    except ImportError:
        logger.warning("cairosvg не встановлений: PSNR/SSIM будуть None.")
        return None
    except Exception as e:
        logger.warning("cairosvg is unavailable: %s", e)
        return None

    try:
        png_bytes = cairosvg.svg2png(
            url=str(svg_path),
            output_width=width,
            output_height=height,
        )

        nparr = np.frombuffer(png_bytes, np.uint8)
        rendered = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if rendered is None:
            return None

        if len(rendered.shape) == 3 and rendered.shape[2] == 4:
            bgr = rendered[:, :, :3].astype(np.float32)
            alpha = rendered[:, :, 3:4].astype(np.float32) / 255.0
            white = np.full_like(bgr, 255.0)
            return (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)

        if len(rendered.shape) == 3 and rendered.shape[2] == 3:
            return rendered

        return cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR)

    except Exception as e:
        logger.warning("SVG render failed: %s", e)
        return None


def _render_svg_to_rgba(
    svg_path: Path,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    try:
        import cairosvg
    except Exception as e:
        logger.warning("cairosvg is unavailable for alpha metrics: %s", e)
        return None

    try:
        png_bytes = cairosvg.svg2png(
            url=str(svg_path),
            output_width=width,
            output_height=height,
        )
        nparr = np.frombuffer(png_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        logger.warning("SVG alpha render failed: %s", e)
        return None


def compute_gapless_coverage(rendered_rgba: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[int]]:
    if rendered_rgba is None or rendered_rgba.size == 0:
        return None, None

    if len(rendered_rgba.shape) < 3 or rendered_rgba.shape[2] < 4:
        return 100.0, 0

    alpha = rendered_rgba[:, :, 3]
    gap_pixels = int(np.count_nonzero(alpha < 250))
    total_pixels = int(alpha.size)

    if total_pixels <= 0:
        return None, None

    coverage = 100.0 * (1.0 - gap_pixels / total_pixels)
    return float(max(0.0, min(100.0, coverage))), gap_pixels


def compute_cad_readiness_score(
    gapless_coverage: Optional[float],
    editability_score: Optional[float],
    node_count: int,
    path_count: int,
) -> Optional[float]:
    if gapless_coverage is None or editability_score is None:
        return None

    node_score = 100.0
    if node_count > 3500:
        node_score = max(35.0, 100.0 - (node_count - 3500) / 120.0)

    path_score = 100.0
    if path_count > 220:
        path_score = max(35.0, 100.0 - (path_count - 220) / 8.0)

    score = (
        0.45 * gapless_coverage
        + 0.35 * float(editability_score)
        + 0.12 * node_score
        + 0.08 * path_score
    )

    return float(max(0.0, min(100.0, score)))


def compute_psnr(original: np.ndarray, rendered: np.ndarray) -> Optional[float]:
    """
    Peak Signal-to-Noise Ratio між оригіналом і растеризованим SVG.

    Для стилізованої векторизації PSNR не є головною метрикою,
    але корисний для порівняння точності відтворення.
    """
    if original is None or rendered is None:
        return None

    if original.shape != rendered.shape:
        rendered = cv2.resize(rendered, (original.shape[1], original.shape[0]))

    mse = np.mean((original.astype(np.float32) - rendered.astype(np.float32)) ** 2)

    if mse <= 0:
        return 100.0

    psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    return float(psnr)


def compute_ssim(original: np.ndarray, rendered: np.ndarray) -> Optional[float]:
    """
    Structural Similarity Index між оригіналом і растеризованим SVG.
    """
    if original is None or rendered is None:
        return None

    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        logger.warning("scikit-image не встановлений: SSIM буде None.")
        return None

    if original.shape != rendered.shape:
        rendered = cv2.resize(rendered, (original.shape[1], original.shape[0]))

    try:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        rend_gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)

        score = structural_similarity(
            orig_gray,
            rend_gray,
            data_range=255,
        )

        return float(score)

    except Exception as e:
        logger.warning("SSIM computation failed: %s", e)
        return None


# ══════════════════════════════════════════════════════════════
# EDITABILITY SCORE COMPONENTS
# ══════════════════════════════════════════════════════════════

def compute_named_layer_score(named_layer_count: int, layer_count: int) -> float:
    """
    Оцінка іменованих шарів: 0..1.

    Іменовані <g> — ключова перевага Liniq над звичайним трасуванням.
    """
    if layer_count <= 0:
        return 0.0

    ratio = named_layer_count / max(layer_count, 1)

    return float(np.clip(ratio, 0.0, 1.0))


def compute_closed_path_score(closed_path_count: int, path_count: int) -> float:
    """
    Оцінка замкненості контурів: 0..1.

    Для редагованого SVG бажано, щоб path були замкнені.
    """
    if path_count <= 0:
        return 0.0

    ratio = closed_path_count / max(path_count, 1)

    return float(np.clip(ratio, 0.0, 1.0))


def compute_node_efficiency_score(node_count: int) -> float:
    """
    Оцінка ефективності вузлів: 0..1.

    Менше вузлів = простіше редагувати.
    Але для складних SVG допускається більше вузлів.
    """
    if node_count <= 0:
        return 0.0

    if node_count <= 300:
        return 1.0

    if node_count <= 1000:
        return 0.90

    if node_count <= 2500:
        return 0.72

    if node_count <= 5000:
        return 0.50

    if node_count <= 10000:
        return 0.35

    return 0.20


def compute_layer_structure_score(layer_count: int) -> float:
    """
    Оцінка структури шарів: 0..1.

    Для семантичної векторизації корисно мати кілька шарів,
    але надто багато шарів ускладнюють редагування.
    """
    if layer_count <= 0:
        return 0.0

    if layer_count == 1:
        return 0.75

    if 2 <= layer_count <= 8:
        return 1.0

    if 9 <= layer_count <= 12:
        return 0.80

    if 13 <= layer_count <= 20:
        return 0.55

    return 0.35


def compute_file_size_score(file_size: int) -> float:
    """
    Оцінка розміру SVG: 0..1.

    Менший файл зазвичай легше відкривати, передавати й редагувати.
    """
    if file_size <= 0:
        return 0.0

    size_kb = file_size / 1024

    if size_kb <= 50:
        return 1.0

    if size_kb <= 150:
        return 0.88

    if size_kb <= 500:
        return 0.68

    if size_kb <= 1000:
        return 0.45

    if size_kb <= 2500:
        return 0.30

    return 0.18


def compute_color_consistency_score(color_counts: List[int]) -> float:
    """
    Оцінка кольорової консистентності шарів: 0..1.

    Хороший SVG-шар зазвичай має 1 fill-колір.
    Якщо шар має багато різних кольорів, редагованість нижча.
    """
    if not color_counts:
        return 0.0

    avg_colors = float(np.mean(color_counts))

    if avg_colors <= 1.2:
        return 1.0

    if avg_colors <= 2.0:
        return 0.85

    if avg_colors <= 3.0:
        return 0.65

    if avg_colors <= 5.0:
        return 0.40

    return 0.20


def compute_path_complexity_score(
    path_count: int,
    node_count: int,
    nodes_per_layer: List[int],
) -> float:
    """
    Оцінка структурної складності SVG: 0..1.

    Штрафує:
    - надто багато path;
    - надто багато вузлів на шар;
    - надто високу щільність вузлів на path.
    """
    if path_count <= 0 or node_count <= 0:
        return 0.0

    nodes_per_path = node_count / max(path_count, 1)

    path_score = 1.0
    if path_count > 200:
        path_score = 0.30
    elif path_count > 100:
        path_score = 0.45
    elif path_count > 50:
        path_score = 0.65
    elif path_count > 20:
        path_score = 0.82

    density_score = 1.0
    if nodes_per_path > 1000:
        density_score = 0.35
    elif nodes_per_path > 500:
        density_score = 0.55
    elif nodes_per_path > 250:
        density_score = 0.75

    if nodes_per_layer:
        avg_nodes_per_layer = float(np.mean(nodes_per_layer))

        if avg_nodes_per_layer <= 250:
            layer_density_score = 1.0
        elif avg_nodes_per_layer <= 600:
            layer_density_score = 0.75
        elif avg_nodes_per_layer <= 1200:
            layer_density_score = 0.55
        else:
            layer_density_score = 0.35
    else:
        layer_density_score = 0.0

    return float((path_score + density_score + layer_density_score) / 3)


def compute_editability_score(
    svg_path: Path,
    node_count: int,
    layer_count: int,
    path_count: int,
    file_size: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Editability Score (0–100) — оригінальна метрика Liniq.

    Складові:
    - named_layers: іменовані <g id="layer-...">
    - closed_paths: замкнені path, які закінчуються Z/z
    - node_efficiency: ефективність кількості вузлів
    - layer_structure: корисна кількість шарів
    - color_consistency: один fill-колір на шар
    - path_complexity: кількість path і вузлова щільність
    - file_size: розмір SVG

    Це метрика саме редагованості, а не фотореалістичної схожості.
    """
    named_layer_count = count_named_layers(svg_path)
    closed_path_count = count_closed_paths(svg_path)
    color_counts = extract_layer_color_counts(svg_path)
    nodes_per_layer = get_nodes_per_layer(svg_path)

    named_layer_score = compute_named_layer_score(
        named_layer_count=named_layer_count,
        layer_count=layer_count,
    )

    closed_path_score = compute_closed_path_score(
        closed_path_count=closed_path_count,
        path_count=path_count,
    )

    node_efficiency_score = compute_node_efficiency_score(node_count)
    layer_structure_score = compute_layer_structure_score(layer_count)
    file_size_score = compute_file_size_score(file_size)
    color_consistency_score = compute_color_consistency_score(color_counts)

    path_complexity_score = compute_path_complexity_score(
        path_count=path_count,
        node_count=node_count,
        nodes_per_layer=nodes_per_layer,
    )

    # Ваги підібрані під наукову гіпотезу:
    # редагованість залежить не тільки від малої кількості вузлів,
    # а й від шарів, замкненості та структурованості SVG.
    components = {
        "named_layers": named_layer_score,
        "closed_paths": closed_path_score,
        "node_efficiency": node_efficiency_score,
        "layer_structure": layer_structure_score,
        "color_consistency": color_consistency_score,
        "path_complexity": path_complexity_score,
        "file_size": file_size_score,
    }

    score = (
        0.22 * named_layer_score
        + 0.18 * closed_path_score
        + 0.20 * node_efficiency_score
        + 0.14 * layer_structure_score
        + 0.10 * color_consistency_score
        + 0.10 * path_complexity_score
        + 0.06 * file_size_score
    )

    editability_score = round(float(score * 100), 2)

    component_scores = {
        key: round(float(value * 100), 2)
        for key, value in components.items()
    }

    return editability_score, component_scores


# ══════════════════════════════════════════════════════════════
# MAIN METRICS ENTRYPOINT
# ══════════════════════════════════════════════════════════════

def compute_all_metrics(
    original_path: Path,
    svg_path: Path,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Рахує всі метрики одним викликом.

    Основні метрики для дипломної:
    - node_count
    - layer_count
    - path_count
    - closed_path_count
    - named_layer_count
    - file_size
    - editability_score
    - editability_components

    PSNR/SSIM:
    - рахуються тільки якщо доступний cairosvg;
    - якщо cairosvg немає, повертаємо None, а не фейкові 100/1.000.
    """
    original_path = Path(original_path)
    svg_path = Path(svg_path)

    node_count = count_svg_nodes(svg_path)
    layer_count = count_svg_layers(svg_path)
    path_count = count_svg_paths(svg_path)
    closed_path_count = count_closed_paths(svg_path)
    named_layer_count = count_named_layers(svg_path)
    file_size = get_file_size(svg_path)

    editability_score, editability_components = compute_editability_score(
        svg_path=svg_path,
        node_count=node_count,
        layer_count=layer_count,
        path_count=path_count,
        file_size=file_size,
    )

    psnr = None
    ssim = None
    gapless_coverage = None
    gap_pixels = None

    original = cv2.imread(str(original_path))

    if original is not None:
        original_h, original_w = original.shape[:2]

        target_width = width or original_w
        target_height = height or original_h

        original_resized = cv2.resize(
            original,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

        rendered = _render_svg_to_image(
            svg_path=svg_path,
            width=target_width,
            height=target_height,
        )

        if rendered is not None:
            psnr = compute_psnr(original_resized, rendered)
            ssim = compute_ssim(original_resized, rendered)

        rendered_rgba = _render_svg_to_rgba(
            svg_path=svg_path,
            width=target_width,
            height=target_height,
        )
        gapless_coverage, gap_pixels = compute_gapless_coverage(rendered_rgba)

    cad_readiness_score = compute_cad_readiness_score(
        gapless_coverage=gapless_coverage,
        editability_score=editability_score,
        node_count=node_count,
        path_count=path_count,
    )

    return {
        "psnr": _safe_float(psnr, 2),
        "ssim": _safe_float(ssim, 4),

        "node_count": node_count,
        "layer_count": layer_count,
        "path_count": path_count,
        "closed_path_count": closed_path_count,
        "named_layer_count": named_layer_count,
        "file_size": file_size,

        "editability_score": editability_score,
        "editability_components": editability_components,
        "gapless_coverage": _safe_float(gapless_coverage, 3),
        "gap_pixels": gap_pixels,
        "cad_readiness_score": _safe_float(cad_readiness_score, 1),
    }
