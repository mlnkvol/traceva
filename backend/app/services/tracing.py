import cv2
import numpy as np
from typing import List, Optional, Tuple


class ContourTracer:
    """
    Модуль 2: трасування контурів масок для SVG.

    Покращена версія для Semantic Mode:
    - працює зі структурованими масками після segmentation.py;
    - підтримує зовнішні контури та внутрішні отвори;
    - використовує RETR_CCOMP для ієрархії контурів;
    - застосовує CHAIN_APPROX_TC89_KCOS для чистіших контурів;
    - фільтрує шум, тонкі артефакти та надто дрібні фрагменти;
    - гарантує замкнені контури;
    - обмежує кількість контурів на шар, щоб SVG був редагованішим.
    """

    def trace(
        self,
        mask: np.ndarray,
        simplify: bool = True,
        epsilon_factor: float = 0.003,
        min_area_ratio: float = 0.0006,
        keep_holes: bool = True,
        max_contours: int = 8,
    ) -> List[np.ndarray]:
        """
        Трасує binary mask у список контурів.

        Parameters:
            mask:
                bool ndarray або uint8 ndarray (H, W)

            simplify:
                Якщо True — контури спрощуються через approxPolyDP.

            epsilon_factor:
                Сила спрощення.
                Менше значення = більше деталей і вузлів.
                Більше значення = менше вузлів, але грубіші форми.

            min_area_ratio:
                Мінімальна площа контуру відносно площі зображення.

            keep_holes:
                Якщо True — враховуються внутрішні отвори.
                Це важливо для повного покриття шарів і коректного SVG.

            max_contours:
                Максимальна кількість контурів, яку повертаємо для одного шару.
                Це покращує editability, бо шар не розпадається на сотні шматків.

        Returns:
            List[np.ndarray], де кожен контур — масив точок Nx2.
        """
        if mask is None or mask.size == 0:
            return []

        prepared_mask = self._to_uint_mask(mask)

        h, w = prepared_mask.shape[:2]
        image_area = h * w

        if image_area == 0:
            return []

        prepared_mask = self._prepare_mask(prepared_mask)

        retrieval_mode = cv2.RETR_CCOMP if keep_holes else cv2.RETR_EXTERNAL

        contours, hierarchy = cv2.findContours(
            prepared_mask,
            retrieval_mode,
            cv2.CHAIN_APPROX_TC89_KCOS,
        )

        if not contours:
            return []

        min_area = max(16, int(image_area * min_area_ratio))

        traced_contours = self._process_contours(
            contours=contours,
            hierarchy=hierarchy,
            image_area=image_area,
            min_area=min_area,
            simplify=simplify,
            epsilon_factor=epsilon_factor,
            keep_holes=keep_holes,
        )

        traced_contours.sort(
            key=lambda pts: abs(cv2.contourArea(pts.reshape(-1, 1, 2))),
            reverse=True,
        )

        return traced_contours[:max_contours]

    # ------------------------------------------------------------------
    # Main contour processing
    # ------------------------------------------------------------------

    def _process_contours(
        self,
        contours: Tuple[np.ndarray, ...],
        hierarchy: Optional[np.ndarray],
        image_area: int,
        min_area: int,
        simplify: bool,
        epsilon_factor: float,
        keep_holes: bool,
    ) -> List[np.ndarray]:
        """
        Обробляє контури з урахуванням ієрархії.

        Якщо keep_holes=True:
        - зовнішні контури залишаються;
        - внутрішні отвори теж можуть бути повернуті як окремі контури.
        Але дуже маленькі дірки відкидаються як шум.
        """
        result: List[np.ndarray] = []

        hierarchy_data = None
        if hierarchy is not None and len(hierarchy) > 0:
            hierarchy_data = hierarchy[0]

        for index, contour in enumerate(contours):
            if contour is None or len(contour) < 3:
                continue

            area = abs(cv2.contourArea(contour))

            if area < min_area:
                continue

            is_hole = False

            if hierarchy_data is not None:
                parent_index = hierarchy_data[index][3]
                is_hole = parent_index != -1

            if is_hole and not keep_holes:
                continue

            # Для дірок фільтр площі трохи м'якший, але шум усе одно прибираємо.
            effective_min_area = max(10, int(min_area * 0.45)) if is_hole else min_area

            if area < effective_min_area:
                continue

            if not self._is_reasonable_contour(
                contour=contour,
                image_area=image_area,
                is_hole=is_hole,
            ):
                continue

            processed = contour

            if simplify:
                processed = self._simplify_contour(
                    contour=processed,
                    epsilon_factor=epsilon_factor,
                    is_hole=is_hole,
                )

            if processed is None or len(processed) < 3:
                continue

            processed = self._ensure_closed_contour(processed)

            pts = processed.reshape(-1, 2)

            if len(pts) < 3:
                continue

            result.append(pts)

        return result

    # ------------------------------------------------------------------
    # Mask preparation
    # ------------------------------------------------------------------

    def _to_uint_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Перетворює будь-яку binary-like mask у uint8 0/255.
        """
        if mask.dtype == np.bool_:
            return mask.astype(np.uint8) * 255

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        if mask.max() <= 1:
            return mask * 255

        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return binary

    def _prepare_mask(self, uint_mask: np.ndarray) -> np.ndarray:
        """
        Очищає і стабілізує маску перед трасуванням.

        Тут важливо не бути занадто агресивними:
        segmentation.py уже побудував шари, а tracing.py має зробити
        їх чистішими, не знищуючи форму.
        """
        h, w = uint_mask.shape[:2]
        min_side = min(h, w)

        if min_side < 350:
            close_size = 3
            open_size = 3
            blur_size = 3
        elif min_side < 1000:
            close_size = 5
            open_size = 3
            blur_size = 3
        else:
            close_size = 7
            open_size = 3
            blur_size = 5

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size),
        )
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_size, open_size),
        )

        cleaned = uint_mask.copy()

        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

        if blur_size >= 3:
            cleaned = cv2.GaussianBlur(cleaned, (blur_size, blur_size), 0)

        _, cleaned = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)

        cleaned = self._remove_tiny_components(cleaned)
        cleaned = self._fill_tiny_holes(cleaned)

        return cleaned

    def _remove_tiny_components(self, uint_mask: np.ndarray) -> np.ndarray:
        """
        Прибирає дуже маленькі connected components.
        """
        h, w = uint_mask.shape[:2]
        image_area = h * w

        min_component_area = max(8, int(image_area * 0.00008))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (uint_mask > 0).astype(np.uint8),
            connectivity=8,
        )

        result = np.zeros_like(uint_mask)

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]

            if area >= min_component_area:
                result[labels == label_id] = 255

        if result.sum() == 0:
            return uint_mask

        return result

    def _fill_tiny_holes(self, uint_mask: np.ndarray) -> np.ndarray:
        """
        Заповнює тільки дрібні отвори, а не всі отвори.

        Це важливо: у SVG внутрішні отвори можуть бути значущими.
        Наприклад, просвіти між формами або деталі логотипу.
        """
        h, w = uint_mask.shape[:2]
        image_area = h * w

        inverted = cv2.bitwise_not(uint_mask)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (inverted > 0).astype(np.uint8),
            connectivity=8,
        )

        result = uint_mask.copy()

        max_hole_area = max(10, int(image_area * 0.00025))

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]

            if area > max_hole_area:
                continue

            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            bw = stats[label_id, cv2.CC_STAT_WIDTH]
            bh = stats[label_id, cv2.CC_STAT_HEIGHT]

            touches_border = (
                x <= 0
                or y <= 0
                or x + bw >= w
                or y + bh >= h
            )

            if not touches_border:
                result[labels == label_id] = 255

        return result

    # ------------------------------------------------------------------
    # Contour simplification
    # ------------------------------------------------------------------

    def _simplify_contour(
        self,
        contour: np.ndarray,
        epsilon_factor: float,
        is_hole: bool = False,
    ) -> np.ndarray:
        """
        Адаптивне спрощення контуру.

        Для великих областей можна більше згладжувати.
        Для малих деталей і отворів — обережніше.
        """
        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            return contour

        area = abs(cv2.contourArea(contour))

        epsilon = epsilon_factor * perimeter

        if area > 80_000:
            epsilon *= 1.25

        if area < 3_000:
            epsilon *= 0.65

        if is_hole:
            epsilon *= 0.55

        approx = cv2.approxPolyDP(contour, epsilon, True)

        if approx is None or len(approx) < 3:
            approx = cv2.approxPolyDP(contour, epsilon * 0.5, True)

        if approx is None or len(approx) < 3:
            return contour

        return approx

    def _ensure_closed_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        Гарантує, що контур замкнений.

        OpenCV зазвичай повертає замкнені контури логічно,
        але для SVG зручніше явно мати останню точку рівною першій.
        """
        pts = contour.reshape(-1, 2)

        if len(pts) < 3:
            return contour

        first = pts[0]
        last = pts[-1]

        if np.array_equal(first, last):
            return contour

        closed = np.vstack([pts, first])
        return closed.reshape(-1, 1, 2).astype(np.int32)

    # ------------------------------------------------------------------
    # Contour filtering
    # ------------------------------------------------------------------

    def _is_reasonable_contour(
        self,
        contour: np.ndarray,
        image_area: int,
        is_hole: bool = False,
    ) -> bool:
        """
        Фільтрує підозрілі контури:
        - надто маленькі;
        - надто тонкі лінії;
        - майже порожні bounding boxes;
        - випадковий шум після segmentation.
        """
        area = abs(cv2.contourArea(contour))

        if area <= 0:
            return False

        x, y, w, h = cv2.boundingRect(contour)

        if w <= 1 or h <= 1:
            return False

        bbox_area = w * h

        if bbox_area <= 0:
            return False

        fill_ratio = area / bbox_area
        aspect_ratio = max(w / max(h, 1), h / max(w, 1))

        # Дірки можуть бути тоншими, тому фільтр м'якший.
        if is_hole:
            if aspect_ratio > 25 and area < image_area * 0.01:
                return False

            if fill_ratio < 0.006 and area < image_area * 0.01:
                return False

            return True

        if aspect_ratio > 22 and area < image_area * 0.025:
            return False

        if fill_ratio < 0.01 and area < image_area * 0.018:
            return False

        return True