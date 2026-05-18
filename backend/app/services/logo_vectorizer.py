from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


class LogoVectorizer:
    """
    Окремий режим векторизації для логотипів / line-art / чорно-білої графіки.

    Основна ідея:
    - не використовує SAM2;
    - працює з бінарною маскою;
    - зберігає внутрішні контури через cv2.RETR_TREE;
    - створює compound SVG path з fill-rule="evenodd".
    """

    def vectorize(
        self,
        image_path: Path,
        output_path: Path,
        tolerance: float = 0.5,
        min_area: int = 20,
    ) -> Dict:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Не вдалося прочитати зображення: {image_path}")

        height, width = image.shape[:2]

        binary = self._make_binary_mask(image)
        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE,
        )

        if hierarchy is None or len(contours) == 0:
            self._write_empty_svg(output_path, width, height)
            return {
                "svg_path": str(output_path),
                "layers": [],
                "metrics": {
                    "node_count": 0,
                    "layer_count": 0,
                },
            }

        path_d, node_count = self._contours_to_compound_path(
            contours=contours,
            min_area=min_area,
            tolerance=tolerance,
        )

        fill_color = self._dominant_foreground_color(image, binary)

        self._write_svg(
            output_path=output_path,
            width=width,
            height=height,
            path_d=path_d,
            fill_color=fill_color,
        )

        return {
            "svg_path": str(output_path),
            "layers": [
                {
                    "label": "object_0",
                    "name": "Логотип",
                    "color": fill_color,
                    "area": int(np.sum(binary > 0)),
                    "node_count": node_count,
                }
            ],
            "metrics": {
                "node_count": node_count,
                "layer_count": 1,
            },
        }

    def _make_binary_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Створює бінарну маску логотипу.

        Повертає:
        - 255 = об'єкт;
        - 0 = фон.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Легке згладжування, щоб прибрати дрібний шум,
        # але не знищити деталі логотипу.
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Спочатку припускаємо: темний об'єкт на світлому фоні.
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        foreground_ratio = np.sum(binary > 0) / binary.size

        # Якщо вийшло, що "об'єкт" займає майже все зображення,
        # значить, швидше за все, інверсія неправильна.
        if foreground_ratio > 0.75:
            binary = cv2.bitwise_not(binary)

        # Мінімальне очищення.
        # Для складних логотипів не можна агресивно закривати деталі,
        # інакше очі/зуби/шерсть перетворяться на суцільну пляму.
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary

    def _contours_to_compound_path(
        self,
        contours: List[np.ndarray],
        min_area: int,
        tolerance: float,
    ) -> Tuple[str, int]:
        """
        Перетворює всі контури в один compound path.

        fill-rule="evenodd" у SVG дозволяє внутрішнім контурам
        працювати як вирізи.
        """
        parts: List[str] = []
        node_count = 0

        # Чим менший tolerance, тим точніше контур і більше вузлів.
        # Для логотипів робимо epsilon дуже помірним.
        epsilon_base = max(0.01, float(tolerance))

        for contour in contours:
            area = abs(cv2.contourArea(contour))

            if area < min_area:
                continue

            epsilon = epsilon_base
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if approx is None or len(approx) < 3:
                continue

            points = approx.reshape(-1, 2)
            node_count += len(points)

            first = points[0]
            d = [f"M {first[0]:.2f} {first[1]:.2f}"]

            for point in points[1:]:
                d.append(f"L {point[0]:.2f} {point[1]:.2f}")

            d.append("Z")
            parts.append(" ".join(d))

        return " ".join(parts), node_count

    def _dominant_foreground_color(self, image: np.ndarray, binary: np.ndarray) -> str:
        """
        Визначає колір логотипу за foreground pixels.
        Для чорних логотипів поверне майже чорний.
        """
        mask = binary > 0
        pixels = image[mask]

        if len(pixels) == 0:
            return "#000000"

        b, g, r = np.median(pixels, axis=0).astype(int)

        return f"#{r:02X}{g:02X}{b:02X}"

    def _write_svg(
        self,
        output_path: Path,
        width: int,
        height: int,
        path_d: str,
        fill_color: str,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg
  xmlns="http://www.w3.org/2000/svg"
  width="{width}"
  height="{height}"
  viewBox="0 0 {width} {height}"
>
  <title>Traceva Logo Mode Output</title>
  <desc>Binary logo vectorization with contour hierarchy</desc>

  <g id="layer-1-object_0" data-label="object_0" data-name="Логотип">
    <path
      d="{path_d}"
      fill="{fill_color}"
      fill-rule="evenodd"
      clip-rule="evenodd"
      stroke="none"
    />
  </g>
</svg>
'''

        output_path.write_text(svg, encoding="utf-8")

    def _write_empty_svg(self, output_path: Path, width: int, height: int) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg
  xmlns="http://www.w3.org/2000/svg"
  width="{width}"
  height="{height}"
  viewBox="0 0 {width} {height}"
>
  <title>Traceva Empty Logo Mode Output</title>
</svg>
'''

        output_path.write_text(svg, encoding="utf-8")
