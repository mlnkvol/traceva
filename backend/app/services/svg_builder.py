import re
import svgwrite
from typing import List, Dict, Tuple
from pathlib import Path


class SVGBuilder:
    """
    Модуль 4: збирає фінальний SVG із семантичними шарами.
    Кожен шар — окремий <g id="layer-...">.
    """

    def build(
        self,
        layers: List[Dict],
        width: int,
        height: int,
        output_path: Path,
        background_color: str | None = None,
    ) -> str:
        # Переконуємось, що папка для результату існує
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dwg = svgwrite.Drawing(
            filename=str(output_path),
            size=(f"{width}px", f"{height}px"),
            viewBox=f"0 0 {width} {height}",
            profile="full",
        )

        # Метадані SVG
        dwg.set_desc(
            title="Traceva Output",
            desc="Gapless semantic SVG vectorization",
        )

        if background_color:
            dwg.add(
                dwg.rect(
                    insert=(0, 0),
                    size=(width, height),
                    fill=background_color,
                    stroke="none",
                    id="gapless-underlay",
                )
            )

        for index, layer in enumerate(layers):
            label = str(layer.get("label", f"Layer {index + 1}"))
            color = str(layer.get("color", "#000000"))

            # SVG id не має містити пробіли/дивні символи
            safe_label = self._safe_id(label)

            # У svgwrite не можна додавати label як атрибут до <g>,
            # тому назву шару залишаємо тільки в id.
            group = dwg.g(
                id=f"layer-{index + 1}-{safe_label}",
            )

            if layer.get("prefer_contours"):
                path_sources = layer.get("contours", [])
                path_builder = self._contour_to_path
            else:
                path_sources = layer.get("bezier_segs", [])
                path_builder = self._segments_to_path

            for path_source in path_sources:
                if path_source is None or len(path_source) == 0:
                    continue

                path_d = path_builder(path_source)

                if not path_d:
                    continue

                path = dwg.path(
                    d=path_d,
                    fill=color,
                    fill_opacity=1,
                    stroke=color,
                    stroke_width=1.4 if layer.get("prefer_contours") else 2.25,
                    stroke_linejoin="round",
                    stroke_linecap="round",
                )

                group.add(path)

            dwg.add(group)

        dwg.save(pretty=True)
        return str(output_path)

    def _contour_to_path(self, contour) -> str:
        try:
            pts = contour.reshape(-1, 2)

            if len(pts) < 3:
                return ""

            parts = [f"M {float(pts[0][0]):.2f},{float(pts[0][1]):.2f}"]

            for point in pts[1:]:
                parts.append(f"L {float(point[0]):.2f},{float(point[1]):.2f}")

            parts.append("Z")
            return " ".join(parts)

        except Exception:
            return ""

    def _segments_to_path(self, segments: List[Tuple]) -> str:
        """
        Перетворює список сегментів (P0, P1, P2, P3)
        у рядок SVG path.
        """
        if not segments:
            return ""

        try:
            first_segment = segments[0]
            p0 = first_segment[0]

            parts = [f"M {float(p0[0]):.2f},{float(p0[1]):.2f}"]

            for segment in segments:
                if len(segment) != 4:
                    continue

                _, p1, p2, p3 = segment

                parts.append(
                    f"C {float(p1[0]):.2f},{float(p1[1]):.2f} "
                    f"{float(p2[0]):.2f},{float(p2[1]):.2f} "
                    f"{float(p3[0]):.2f},{float(p3[1]):.2f}"
                )

            if len(parts) <= 1:
                return ""

            parts.append("Z")
            return " ".join(parts)

        except Exception:
            return ""

    def _safe_id(self, value: str) -> str:
        """
        Робить безпечний id для SVG-елемента.
        Наприклад: "Main object" -> "main-object".
        """
        value = value.strip().lower()
        value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value)
        value = value.strip("-")

        return value or "layer"
