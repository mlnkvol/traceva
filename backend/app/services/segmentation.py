import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SegmentationService:
    """
    Semantic segmentation service.

    Версія для дипломного проєкту, пункт 1:
    - Semantic Mode покриває ВСЕ зображення шарами;
    - spatial K-means: LAB + XY координати;
    - No Gaps Policy гарантує, що немає прозорих дірок між шарами;
    - SAM2 поки лишається в класі як допоміжний інструмент / майбутній модуль,
      але основний Semantic pipeline більше не вирізає лише subject.
    """

    def __init__(self, checkpoint: str, model_cfg: str, device: str = "cpu"):
        self.device = device
        self.mask_generator = None
        self._load_model(checkpoint, model_cfg)

    def _load_model(self, checkpoint: str, model_cfg: str) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)

            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                points_per_side=32,
                pred_iou_thresh=0.82,
                stability_score_thresh=0.88,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=180,
            )

            logger.info("SAM2 завантажено успішно")

        except Exception as e:
            logger.warning("SAM2 недоступний, використовується fallback/full-image pipeline: %s", e)
            self.mask_generator = None

    # ══════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════

    def analyze_image(self, image: np.ndarray) -> Dict:
        """
        Аналізує зображення і повертає рекомендовані параметри векторизації.

        Повертає:
        {
            "mode": "logo" | "photo" | "illustration",
            "tolerance": float,
            "max_layers": int,
            "reason": str,
            "tips": List[str]
        }
        """
        if image is None or image.size == 0:
            return self._default_recommendation()

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        small = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        pixels = small.reshape(-1, 3)
        unique_colors = len(np.unique(pixels, axis=0))
        color_richness = unique_colors / (128 * 128)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (255 * h * w)

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        binary_ratio = np.sum(binary == 0) / (h * w)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation_mean = float(hsv[:, :, 1].mean())

        face_detected = self._detect_face(gray)

        is_logo = (
            color_richness < 0.08
            and edge_density > 0.02
            and (binary_ratio < 0.15 or binary_ratio > 0.85)
        )

        is_photo = face_detected or (color_richness > 0.25 and saturation_mean > 30)

        if is_logo:
            return {
                "mode": "logo",
                "tolerance": 1.2,
                "max_layers": 4,
                "reason": (
                    "Виявлено логотип або лінійну графіку: "
                    "мало унікальних кольорів, чіткі контури."
                ),
                "tips": [
                    "Tolerance 0.8–1.5 дає чисті криві без зайвих вузлів.",
                    "Для логотипів зазвичай достатньо 1–4 шарів.",
                    "Якщо є дрібні деталі — зменш Tolerance до 0.5–0.8.",
                ],
                "debug": {
                    "color_richness": round(color_richness, 3),
                    "edge_density": round(edge_density, 4),
                    "saturation_mean": round(saturation_mean, 1),
                    "face_detected": face_detected,
                },
            }

        if is_photo:
            return {
                "mode": "photo",
                "tolerance": 0.5,
                "max_layers": 72,
                "reason": (
                    "Виявлено фото або портрет: рекомендовано Semantic Mode "
                    "з повним покриттям зображення шарами."
                ),
                "tips": [
                    "Для фото почни з Tolerance 0.8–1.0.",
                    "Max Layers 72–84 дає кращу деталізацію фото; для дуже складних сцен можна підняти до 96.",
                    "Якщо результат занадто плаский — спробуй Max Layers 8.",
                    "Якщо SVG занадто важкий — зменш Max Layers до 5–6.",
                ],
                "debug": {
                    "color_richness": round(color_richness, 3),
                    "edge_density": round(edge_density, 4),
                    "saturation_mean": round(saturation_mean, 1),
                    "face_detected": face_detected,
                },
            }

        return {
            "mode": "illustration",
                "tolerance": 0.7,
                "max_layers": 10,
            "reason": "Виявлено ілюстрацію або зображення середньої складності.",
            "tips": [
                "Tolerance 0.8–1.2 підходить для більшості ілюстрацій.",
                "10–14 шарів зазвичай достатньо для ілюстрацій; для фото краще 18–22.",
                "Якщо є градієнти — можна спробувати 8 шарів.",
            ],
            "debug": {
                "color_richness": round(color_richness, 3),
                "edge_density": round(edge_density, 4),
                "saturation_mean": round(saturation_mean, 1),
                "face_detected": face_detected,
            },
        }

    def segment(self, image: np.ndarray, max_layers: int = 10) -> List[Dict]:
        """
        Основний метод сегментації.

        Нова архітектура для Semantic Mode:
        замість вирізання одного об'єкта будуємо шари,
        які покривають усе зображення.

        Повертає список шарів:
        [{"mask": bool ndarray, "label": str, "color": str, "area": int}, ...]
        """
        max_layers = int(np.clip(max_layers, 1, 96))

        if image is None or image.size == 0:
            return []

        layers = self._segment_full_image(image, max_layers=max_layers)

        if layers:
            return layers

        return self._segment_fallback(image, max_layers=max_layers)

    def generate_preview_masks(self, image: np.ndarray, max_layers: int = 24) -> List[Dict]:
        """
        Returns object-like SAM masks for the interactive pre-vectorization editor.

        Unlike segment(), this intentionally does not create full-frame color
        quantization layers. The preview stage is for design intent: object parts,
        holes, leaves, hair, shadows, etc. can be merged/deleted/reordered before
        tracing starts.
        """
        if image is None or image.size == 0:
            return []

        if self.mask_generator is None:
            logger.warning("SAM preview unavailable; using classical object-mask fallback.")
            return self._generate_classical_preview_masks(image, max_layers=max_layers)

        h, w = image.shape[:2]
        image_area = max(1, h * w)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            sam_masks = self.mask_generator.generate(rgb)
        except Exception as exc:
            logger.warning("SAM preview mask generation failed: %s", exc)
            return self._generate_classical_preview_masks(image, max_layers=max_layers)

        candidates = []

        for item in sam_masks or []:
            mask = item.get("segmentation")
            area = int(item.get("area", 0) or 0)

            if mask is None:
                continue

            mask_bool = mask.astype(bool)
            area = area or int(mask_bool.sum())
            ratio = area / image_area

            if ratio < 0.0015 or ratio > 0.88:
                continue

            border_overlap = self._border_overlap_ratio(mask_bool)
            if ratio > 0.45 and border_overlap > 0.22:
                continue

            mask_bool = self._cleanup_mask(
                mask_bool,
                min_area=max(24, int(image_area * 0.00012)),
                close_kernel_size=3,
                open_kernel_size=1,
            )

            area = int(mask_bool.sum())
            if area <= 0:
                continue

            candidates.append(
                {
                    "mask": mask_bool,
                    "label": "sam_mask",
                    "color": self._mean_bgr_color(image, mask_bool),
                    "area": area,
                    "predicted_iou": float(item.get("predicted_iou", 0.0) or 0.0),
                    "stability_score": float(item.get("stability_score", 0.0) or 0.0),
                }
            )

        candidates.sort(
            key=lambda item: (
                item.get("stability_score", 0.0),
                item.get("predicted_iou", 0.0),
                min(item["area"], image_area * 0.28),
            ),
            reverse=True,
        )

        accepted: List[Dict] = []

        for candidate in candidates:
            candidate_mask = candidate["mask"].astype(bool)
            candidate_area = max(1, int(candidate_mask.sum()))

            too_similar = False
            for accepted_item in accepted:
                accepted_mask = accepted_item["mask"].astype(bool)
                intersection = int(np.logical_and(candidate_mask, accepted_mask).sum())
                union = int(np.logical_or(candidate_mask, accepted_mask).sum())
                overlap_candidate = intersection / candidate_area
                iou = intersection / max(1, union)

                if iou > 0.78 or overlap_candidate > 0.88:
                    too_similar = True
                    break

            if too_similar:
                continue

            candidate["label"] = f"sam_mask_{len(accepted) + 1}"
            accepted.append(candidate)

            if len(accepted) >= max_layers:
                break

        accepted.sort(key=lambda item: item["area"], reverse=True)
        if accepted:
            return accepted

        logger.warning("SAM preview returned no usable masks; using classical fallback.")
        return self._generate_classical_preview_masks(image, max_layers=max_layers)

    def _generate_classical_preview_masks(self, image: np.ndarray, max_layers: int = 24) -> List[Dict]:
        h, w = image.shape[:2]
        image_area = max(1, h * w)

        foreground = self._grabcut_foreground(image)

        if foreground is None or int(foreground.sum()) < image_area * 0.015:
            foreground = self._saliency_foreground(image)

        if foreground is None or int(foreground.sum()) == 0:
            return []

        foreground = self._cleanup_mask(
            foreground,
            min_area=max(32, int(image_area * 0.00025)),
            close_kernel_size=5,
            open_kernel_size=1,
        )

        pieces = self._split_preview_foreground_by_color(
            image=image,
            foreground=foreground,
            max_layers=max_layers,
        )

        if not pieces:
            pieces = [foreground]

        layers: List[Dict] = []

        for index, mask in enumerate(pieces[:max_layers], start=1):
            mask = mask.astype(bool)
            area = int(mask.sum())

            if area <= 0:
                continue

            layers.append(
                {
                    "mask": mask,
                    "label": f"preview_mask_{index}",
                    "color": self._mean_bgr_color(image, mask),
                    "area": area,
                }
            )

        layers.sort(key=lambda item: item["area"], reverse=True)
        return layers

    def _grabcut_foreground(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            h, w = image.shape[:2]
            margin_x = max(2, int(w * 0.04))
            margin_y = max(2, int(h * 0.04))
            rect = (
                margin_x,
                margin_y,
                max(1, w - 2 * margin_x),
                max(1, h - 2 * margin_y),
            )

            gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            cv2.grabCut(
                image,
                gc_mask,
                rect,
                bgd_model,
                fgd_model,
                4,
                cv2.GC_INIT_WITH_RECT,
            )

            return (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        except Exception as exc:
            logger.warning("GrabCut preview fallback failed: %s", exc)
            return None

    def _saliency_foreground(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            edges = cv2.Canny(gray, 60, 150)
            edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

            sat_threshold = max(45, int(np.percentile(saturation, 72)))
            dark_threshold = int(np.percentile(value, 18))
            bright_threshold = int(np.percentile(value, 88))

            mask = (
                (saturation >= sat_threshold)
                | (value <= dark_threshold)
                | ((value >= bright_threshold) & (saturation >= 25))
                | (edges > 0)
            )

            border = max(4, int(min(image.shape[:2]) * 0.025))
            mask[:border, :] = False
            mask[-border:, :] = False
            mask[:, :border] = False
            mask[:, -border:] = False
            return mask
        except Exception as exc:
            logger.warning("Saliency preview fallback failed: %s", exc)
            return None

    def _split_preview_foreground_by_color(
        self,
        image: np.ndarray,
        foreground: np.ndarray,
        max_layers: int,
    ) -> List[np.ndarray]:
        foreground = foreground.astype(bool)
        h, w = foreground.shape[:2]
        image_area = max(1, h * w)
        flat_indices = np.flatnonzero(foreground)

        if len(flat_indices) < 20:
            return []

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        ys, xs = np.where(foreground)
        pixels = lab[foreground].astype(np.float32)
        spatial_scale = max(h, w)
        spatial = np.column_stack(
            [
                xs.astype(np.float32) / spatial_scale * 28.0,
                ys.astype(np.float32) / spatial_scale * 28.0,
            ]
        )
        features = np.column_stack([pixels, spatial]).astype(np.float32)

        k = int(np.clip(min(max_layers, max(4, max_layers // 2)), 2, 18))

        try:
            _compactness, labels, _centers = cv2.kmeans(
                features,
                k,
                None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 0.6),
                2,
                cv2.KMEANS_PP_CENTERS,
            )
        except Exception:
            return []

        pieces: List[np.ndarray] = []
        min_piece_area = max(24, int(image_area * 0.0012))

        for cluster_id in range(k):
            cluster_mask = np.zeros((h, w), dtype=bool)
            selector = labels.reshape(-1) == cluster_id
            cluster_mask.flat[flat_indices[selector]] = True

            num_labels, component_labels, stats, _ = cv2.connectedComponentsWithStats(
                cluster_mask.astype(np.uint8),
                connectivity=8,
            )

            for label_id in range(1, num_labels):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                if area < min_piece_area:
                    continue

                component = component_labels == label_id
                component = self._cleanup_mask(
                    component,
                    min_area=min_piece_area,
                    close_kernel_size=3,
                    open_kernel_size=1,
                )

                if int(component.sum()) >= min_piece_area:
                    pieces.append(component)

        pieces.sort(key=lambda mask: int(mask.sum()), reverse=True)
        return pieces

    # ══════════════════════════════════════════════════════════════
    # FULL IMAGE SEMANTIC PIPELINE
    # ══════════════════════════════════════════════════════════════

    def _segment_full_image(self, image: np.ndarray, max_layers: int = 10) -> List[Dict]:
        """
        Full-image segmentation pipeline.

        Ідея:
        - не вирізаємо людину;
        - не обрізаємо портретне вікно;
        - розбиваємо все зображення на просторово-кольорові шари;
        - гарантуємо повне покриття кадру через No Gaps Policy.
        """
        h, w = image.shape[:2]
        total = h * w

        if total == 0:
            return []

        processed = self._preprocess_image(image)
        full_mask = np.ones((h, w), dtype=bool)

        layer_count = int(np.clip(max_layers, 3, 96))

        layers = self._quantize_region_spatial(
            image=processed,
            region_mask=full_mask,
            n_clusters=layer_count,
            label_prefix="semantic_layer",
            spatial_weight=0.12,
            fill_layer_holes=False,
        )

        if not layers:
            logger.warning("Full-image spatial K-means не дав шарів, пробуємо LAB K-means.")
            layers = self._quantize_region(
                image=processed,
                region_mask=full_mask,
                n_clusters=layer_count,
                label_prefix="semantic_layer",
                fill_layer_holes=False,
            )

        if not layers:
            logger.warning("Full-image segmentation не дала шарів.")
            return []

        layers = self._ensure_full_coverage(
            layers=layers,
            image=processed,
        )

        layers = self._remove_empty_layers(layers)

        layers = sorted(
            layers,
            key=lambda layer: layer["area"],
            reverse=True,
        )

        normalized = []

        for idx, layer in enumerate(layers, start=1):
            normalized.append(
                {
                    "mask": layer["mask"],
                    "label": f"semantic_layer_{idx}",
                    "color": layer["color"],
                    "area": int(layer["area"]),
                }
            )

        logger.info(
            "Full-image segmentation: layers=%s, coverage=100%%",
            len(normalized),
        )

        return normalized

    # ══════════════════════════════════════════════════════════════
    # FACE DETECTION
    # ══════════════════════════════════════════════════════════════

    def _detect_face(self, gray: np.ndarray) -> bool:
        """Спрощена перевірка на обличчя через Haar cascades OpenCV."""
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30),
            )

            return len(faces) > 0

        except Exception:
            return False

    # ══════════════════════════════════════════════════════════════
    # LEGACY SUBJECT PIPELINE
    # Залишаємо в файлі як резерв / для майбутнього Portrait Mode.
    # Основний segment() зараз його НЕ викликає.
    # ══════════════════════════════════════════════════════════════

    def _segment_with_sam(self, image: np.ndarray, max_layers: int = 6) -> List[Dict]:
        """
        Legacy SAM2 subject pipeline.

        Зараз не використовується основним Semantic Mode.
        Може знадобитися пізніше для Portrait Mode.
        """
        processed = self._preprocess_image(image)
        subject_mask = self._extract_subject_mask(processed)

        if subject_mask is None or subject_mask.sum() == 0:
            logger.warning("SAM2 subject mask порожня, fallback.")
            return self._segment_fallback(image, max_layers=max_layers)

        h, w = image.shape[:2]
        total = max(h * w, 1)
        ratio = subject_mask.sum() / total

        if ratio < 0.02 or ratio > 0.92:
            logger.warning("Subject mask ratio=%.3f підозрілий, fallback.", ratio)
            return self._segment_fallback(image, max_layers=max_layers)

        subject_mask = self._cleanup_mask(
            subject_mask,
            min_area=max(40, int(total * 0.0015)),
        )

        subject_mask = self._crop_mask_to_subject_window(subject_mask)

        base_layer_count = min(max_layers, 7)

        tone_layers = self._quantize_region_spatial(
            image=processed,
            region_mask=subject_mask,
            n_clusters=base_layer_count,
            label_prefix="subject_tone",
            spatial_weight=0.18,
            fill_layer_holes=False,
        )

        if not tone_layers:
            logger.warning("Spatial K-means не дав шарів, fallback до LAB K-means.")
            tone_layers = self._quantize_region(
                image=processed,
                region_mask=subject_mask,
                n_clusters=base_layer_count,
                label_prefix="subject_tone",
                fill_layer_holes=False,
            )

        if not tone_layers:
            return self._segment_fallback(image, max_layers=max_layers)

        tone_layers = self._ensure_soft_coverage(
            layers=tone_layers,
            image=processed,
            region_mask=subject_mask,
        )

        detail_layer = self._build_detail_layer(
            image=processed,
            subject_mask=subject_mask,
        )

        if detail_layer is not None:
            tone_layers.append(detail_layer)

        tone_layers = self._remove_empty_layers(tone_layers)

        tone_layers = sorted(
            tone_layers,
            key=lambda layer: (
                1 if layer["label"] == "subject_details" else 0,
                self._hex_luminance(layer["color"]),
            ),
            reverse=True,
        )

        normalized = []

        tone_idx = 1
        for layer in tone_layers:
            if layer["label"] == "subject_details":
                label = "subject_details"
            else:
                label = f"subject_tone_{tone_idx}"
                tone_idx += 1

            normalized.append(
                {
                    "mask": layer["mask"],
                    "label": label,
                    "color": layer["color"],
                    "area": int(layer["area"]),
                }
            )

        return normalized

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        М'яке згладжування шуму зі збереженням країв.

        Для full-image segmentation не можна згладжувати занадто агресивно,
        бо тоді зникають дрібні деталі.
        """
        return cv2.bilateralFilter(
            image,
            d=7,
            sigmaColor=35,
            sigmaSpace=35,
        )

    def _extract_subject_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Legacy: отримує маску головного суб'єкта через SAM2.
        Зараз лишено для майбутнього Portrait Mode.
        """
        if self.mask_generator is None:
            return None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam_masks = self.mask_generator.generate(rgb)

        if not sam_masks:
            return None

        h, w = image.shape[:2]
        total = h * w
        border_margin = max(4, int(min(h, w) * 0.02))

        prepared = []

        for item in sam_masks:
            mask = item.get("segmentation")
            area = int(item.get("area", 0))

            if mask is None or area <= 0:
                continue

            mask_bool = mask.astype(bool)
            ratio = area / max(total, 1)

            if ratio < 0.003 or ratio > 0.90:
                continue

            cx, cy = self._mask_centroid(mask_bool)
            x1, y1, x2, y2 = self._mask_bbox(mask_bool)

            bbox_w = max(1, x2 - x1 + 1)
            bbox_h = max(1, y2 - y1 + 1)
            aspect = bbox_w / max(bbox_h, 1)

            dx = abs(cx - w / 2) / max(w / 2, 1)
            dy = abs(cy - h * 0.45) / max(h / 2, 1)
            centrality = 1.0 - min(1.0, dx * 0.75 + dy * 0.45)

            border_overlap = self._border_overlap_ratio(mask_bool)
            touches = self._touches_border(mask_bool, margin=border_margin)

            border_penalty = 0.0
            if touches and border_overlap > 0.08:
                border_penalty += 0.45
            if ratio > 0.25 and touches:
                border_penalty += 0.5

            bg_penalty = 0.0

            if x1 <= border_margin and x2 >= w - border_margin and ratio > 0.12:
                bg_penalty += 0.9

            if y1 <= border_margin and y2 >= h - border_margin and ratio > 0.12:
                bg_penalty += 0.9

            if x1 > 0.48 * w and bbox_h > 0.35 * h and ratio > 0.04:
                bg_penalty += 0.75

            if x2 >= w - border_margin and bbox_h > 0.35 * h and ratio > 0.04:
                bg_penalty += 0.75

            area_score = min(ratio / 0.22, 1.0)

            portrait_bonus = 0.0
            if 0.25 <= aspect <= 1.35 and ratio > 0.02:
                portrait_bonus += 0.2

            score = (
                centrality * 1.8
                + area_score * 0.65
                + portrait_bonus
                - border_penalty
                - bg_penalty
            )

            prepared.append(
                {
                    "mask": mask_bool,
                    "area": area,
                    "ratio": ratio,
                    "cx": cx,
                    "cy": cy,
                    "score": score,
                    "border_overlap": border_overlap,
                    "touches": touches,
                    "bbox": (x1, y1, x2, y2),
                }
            )

        if not prepared:
            return None

        prepared.sort(key=lambda x: x["score"], reverse=True)
        best = prepared[0]

        if best["score"] < -0.25:
            logger.warning("Немає впевненого seed, fallback.")
            return None

        expanded = best["mask"].copy()

        dilate_k = max(9, int(min(h, w) * 0.035))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))

        seed_dilated = cv2.dilate(
            best["mask"].astype(np.uint8),
            kernel,
            iterations=1,
        ).astype(bool)

        for item in prepared[1:]:
            current_mask = item["mask"]
            ratio = item["ratio"]
            x1, y1, x2, y2 = item["bbox"]
            bbox_h = max(1, y2 - y1 + 1)

            if x1 <= border_margin and x2 >= w - border_margin and ratio > 0.12:
                continue

            if y1 <= border_margin and y2 >= h - border_margin and ratio > 0.12:
                continue

            if x1 > 0.48 * w and bbox_h > 0.35 * h and ratio > 0.04:
                continue

            if x2 >= w - border_margin and bbox_h > 0.35 * h and ratio > 0.04:
                continue

            if item["touches"] and item["border_overlap"] > 0.12 and ratio > 0.08:
                continue

            overlap = self._mask_overlap_ratio(current_mask, seed_dilated)
            in_central_band = 0.08 * w < item["cx"] < 0.92 * w

            if overlap > 0.04 or (in_central_band and item["score"] > 0.35):
                expanded |= current_mask

        expanded = self._cleanup_mask(
            expanded,
            min_area=max(40, int(total * 0.0015)),
        )

        refined = self._refine_with_grabcut(image, expanded)

        if refined is not None and refined.sum() > total * 0.015:
            expanded = refined

        final = self._cleanup_mask(
            expanded,
            min_area=max(40, int(total * 0.0015)),
        )

        final = self._crop_mask_to_subject_window(final)

        if final.sum() == 0:
            return best["mask"]

        return final

    # ══════════════════════════════════════════════════════════════
    # FALLBACK
    # ══════════════════════════════════════════════════════════════

    def _segment_fallback(self, image: np.ndarray, max_layers: int = 10) -> List[Dict]:
        """
        Fallback без SAM2.

        Для нової архітектури fallback теж покриває все зображення,
        а не вирізає subject.
        """
        h, w = image.shape[:2]

        if h == 0 or w == 0:
            return []

        processed = self._preprocess_image(image)
        full_mask = np.ones((h, w), dtype=bool)

        layers = self._quantize_region_spatial(
            image=processed,
            region_mask=full_mask,
            n_clusters=int(np.clip(max_layers, 3, 96)),
            label_prefix="semantic_layer",
            spatial_weight=0.12,
            fill_layer_holes=False,
        )

        if not layers:
            layers = self._quantize_region(
                image=processed,
                region_mask=full_mask,
                n_clusters=int(np.clip(max_layers, 3, 96)),
                label_prefix="semantic_layer",
                fill_layer_holes=False,
            )

        layers = self._ensure_full_coverage(
            layers=layers,
            image=processed,
        )

        layers = self._remove_empty_layers(layers)

        layers = sorted(
            layers,
            key=lambda layer: layer["area"],
            reverse=True,
        )

        normalized = []

        for idx, layer in enumerate(layers, start=1):
            normalized.append(
                {
                    "mask": layer["mask"],
                    "label": f"semantic_layer_{idx}",
                    "color": layer["color"],
                    "area": int(layer["area"]),
                }
            )

        return normalized

    # ══════════════════════════════════════════════════════════════
    # QUANTIZATION
    # ══════════════════════════════════════════════════════════════

    def _quantize_region_spatial(
        self,
        image: np.ndarray,
        region_mask: np.ndarray,
        n_clusters: int,
        label_prefix: str,
        spatial_weight: float = 0.22,
        fill_layer_holes: bool = False,
    ) -> List[Dict]:
        """
        Просторово-кольорова кластеризація: LAB + XY координати.

        Враховує не лише колір, а й позицію пікселя.
        Це робить шари більш локальними й редагованими.
        """
        h, w = image.shape[:2]
        region_mask_bool = region_mask.astype(bool)

        if int(region_mask_bool.sum()) == 0:
            return []

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        ys, xs = np.mgrid[0:h, 0:w]

        x_norm = (xs / max(w, 1) * 255 * spatial_weight).astype(np.float32)
        y_norm = (ys / max(h, 1) * 255 * spatial_weight).astype(np.float32)

        feature_map = np.dstack(
            [
                lab[:, :, 0],
                lab[:, :, 1],
                lab[:, :, 2],
                x_norm,
                y_norm,
            ]
        )

        region_pixels = feature_map[region_mask_bool]

        if len(region_pixels) == 0:
            return []

        sample_size = min(len(region_pixels), 100000)

        if len(region_pixels) > sample_size:
            idx = np.random.choice(len(region_pixels), sample_size, replace=False)
            sample = np.float32(region_pixels[idx])
        else:
            sample = np.float32(region_pixels)

        k = max(1, min(int(n_clusters), len(sample)))

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            45,
            0.35,
        )

        try:
            _, _, centers = cv2.kmeans(
                sample,
                k,
                None,
                criteria,
                10,
                cv2.KMEANS_PP_CENTERS,
            )

        except Exception as e:
            logger.warning("Spatial K-means failed: %s", e)
            return self._quantize_region(
                image=image,
                region_mask=region_mask,
                n_clusters=n_clusters,
                label_prefix=label_prefix,
                fill_layer_holes=fill_layer_holes,
            )

        all_pixels = feature_map.reshape(-1, 5).astype(np.float32)

        all_labels = self._assign_nearest_center(
            pixels=all_pixels,
            centers=centers,
        ).reshape(h, w)

        min_area = max(5, int(h * w * 0.00008))
        layers = []

        for cluster_id in range(k):
            full_mask = (all_labels == cluster_id) & region_mask_bool

            if int(full_mask.sum()) < min_area:
                continue

            if fill_layer_holes:
                full_mask = self._fill_holes(full_mask, close_kernel_size=5)

            full_mask = self._cleanup_mask(
                full_mask,
                min_area=min_area,
                close_kernel_size=3,
                open_kernel_size=1,
            )

            if int(full_mask.sum()) < min_area:
                continue

            color = self._mean_bgr_color(image, full_mask)

            layers.append(
                {
                    "mask": full_mask,
                    "label": f"{label_prefix}_{cluster_id + 1}",
                    "color": color,
                    "area": int(full_mask.sum()),
                }
            )

        return layers

    def _quantize_region(
        self,
        image: np.ndarray,
        region_mask: np.ndarray,
        n_clusters: int,
        label_prefix: str,
        fill_layer_holes: bool = False,
    ) -> List[Dict]:
        """K-means кластеризація тільки в LAB-просторі."""
        region_mask = region_mask.astype(bool)
        h, w = image.shape[:2]
        total = h * w

        if region_mask.sum() == 0:
            return []

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        region_pixels = lab[region_mask]

        if len(region_pixels) == 0:
            return []

        sample_size = min(len(region_pixels), 90000)

        if len(region_pixels) > sample_size:
            idx = np.random.choice(len(region_pixels), sample_size, replace=False)
            sample = np.float32(region_pixels[idx])
        else:
            sample = np.float32(region_pixels)

        k = max(1, min(int(n_clusters), len(sample)))

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            40,
            0.35,
        )

        try:
            _, _, centers = cv2.kmeans(
                sample,
                k,
                None,
                criteria,
                8,
                cv2.KMEANS_PP_CENTERS,
            )

        except Exception as e:
            logger.warning("K-means failed: %s", e)
            return []

        centers = np.float32(centers)
        region_pixels_f = np.float32(region_pixels)

        labels = self._assign_nearest_center(
            pixels=region_pixels_f,
            centers=centers,
        )

        min_area = max(5, int(total * 0.00008))
        flat_indices = np.flatnonzero(region_mask)

        layers = []

        for cluster_id in range(k):
            selector = labels == cluster_id

            if int(selector.sum()) < min_area:
                continue

            cluster_mask = np.zeros((h, w), dtype=bool)
            cluster_mask.flat[flat_indices[selector]] = True

            if fill_layer_holes:
                cluster_mask = self._fill_holes(cluster_mask, close_kernel_size=5)

            cluster_mask = self._cleanup_mask(
                cluster_mask,
                min_area=min_area,
                close_kernel_size=3,
                open_kernel_size=1,
            )

            if int(cluster_mask.sum()) < min_area:
                continue

            layers.append(
                {
                    "mask": cluster_mask,
                    "label": f"{label_prefix}_{cluster_id + 1}",
                    "color": self._mean_bgr_color(image, cluster_mask),
                    "area": int(cluster_mask.sum()),
                }
            )

        if not layers and k > 1:
            return self._quantize_region(
                image=image,
                region_mask=region_mask,
                n_clusters=k - 1,
                label_prefix=label_prefix,
                fill_layer_holes=fill_layer_holes,
            )

        return layers

    def _ensure_full_coverage(
        self,
        layers: List[Dict],
        image: np.ndarray,
    ) -> List[Dict]:
        """
        No Gaps Policy.

        Гарантує, що всі пікселі зображення покриті хоча б одним шаром.
        Прозорих ділянок у SVG після цього бути не має.
        """
        if not layers:
            return layers

        h, w = image.shape[:2]

        coverage = np.zeros((h, w), dtype=bool)

        for layer in layers:
            coverage |= layer["mask"]

        uncovered = ~coverage
        uncovered_count = int(uncovered.sum())

        if uncovered_count == 0:
            return layers

        logger.info(
            "No Gaps Policy: uncovered pixels=%s, %.2f%%",
            uncovered_count,
            100 * uncovered_count / max(h * w, 1),
        )

        if uncovered_count < int(h * w * 0.001):
            biggest = max(layers, key=lambda layer: layer["area"])
            biggest["mask"] = biggest["mask"] | uncovered
            biggest["area"] = int(biggest["mask"].sum())
            return layers

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        uncovered_pixels = lab[uncovered]

        layer_colors_lab = []

        for layer in layers:
            if int(layer["mask"].sum()) > 0:
                layer_colors_lab.append(lab[layer["mask"]].mean(axis=0))
            else:
                layer_colors_lab.append(np.array([128.0, 128.0, 128.0]))

        centers = np.array(layer_colors_lab, dtype=np.float32)

        distances = np.sum(
            (uncovered_pixels[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )

        closest = np.argmin(distances, axis=1)
        uncovered_indices = np.flatnonzero(uncovered)

        for layer_idx in range(len(layers)):
            belongs_to_this_layer = uncovered_indices[closest == layer_idx]

            if len(belongs_to_this_layer) > 0:
                layers[layer_idx]["mask"].flat[belongs_to_this_layer] = True
                layers[layer_idx]["area"] = int(layers[layer_idx]["mask"].sum())

        return layers

    def _ensure_soft_coverage(
        self,
        layers: List[Dict],
        image: np.ndarray,
        region_mask: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Legacy м'яка No Gaps Policy для subject/portrait pipeline.
        Зараз основний Semantic Mode використовує _ensure_full_coverage().
        """
        if not layers:
            return layers

        h, w = image.shape[:2]

        if region_mask is None:
            target_mask = np.ones((h, w), dtype=bool)
        else:
            target_mask = region_mask.astype(bool)

        coverage = np.zeros((h, w), dtype=bool)

        for layer in layers:
            coverage |= layer["mask"]

        uncovered = target_mask & (~coverage)
        uncovered_count = int(uncovered.sum())

        if uncovered_count == 0:
            return layers

        target_count = int(target_mask.sum())
        uncovered_ratio = uncovered_count / max(target_count, 1)

        if uncovered_ratio > 0.08:
            return layers

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        uncovered_pixels = lab[uncovered]

        layer_colors_lab = []

        for layer in layers:
            if int(layer["mask"].sum()) > 0:
                layer_colors_lab.append(lab[layer["mask"]].mean(axis=0))
            else:
                layer_colors_lab.append(np.array([128.0, 128.0, 128.0]))

        centers = np.array(layer_colors_lab, dtype=np.float32)

        distances = np.sum(
            (uncovered_pixels[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )

        closest = np.argmin(distances, axis=1)
        uncovered_indices = np.flatnonzero(uncovered)

        for layer_idx in range(len(layers)):
            belongs_to_this_layer = uncovered_indices[closest == layer_idx]

            if len(belongs_to_this_layer) > 0:
                layers[layer_idx]["mask"].flat[belongs_to_this_layer] = True
                layers[layer_idx]["area"] = int(layers[layer_idx]["mask"].sum())

        return layers

    # ══════════════════════════════════════════════════════════════
    # DETAIL LAYER
    # Legacy для майбутнього Portrait Mode.
    # ══════════════════════════════════════════════════════════════

    def _build_detail_layer(
        self,
        image: np.ndarray,
        subject_mask: np.ndarray,
    ) -> Optional[Dict]:
        """
        Legacy detail layer для портретного режиму.
        Основний full-image Semantic Mode зараз його не використовує.
        """
        h, w = image.shape[:2]
        total = h * w

        if subject_mask is None or subject_mask.sum() == 0:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        subject_pixels = gray[subject_mask]
        if len(subject_pixels) == 0:
            return None

        dark_threshold = float(np.percentile(subject_pixels, 28))
        dark_mask = (gray <= dark_threshold) & subject_mask

        edges = cv2.Canny(gray, 45, 120)
        edge_mask = (edges > 0) & subject_mask

        detail_mask = dark_mask | edge_mask

        detail_mask = self._filter_detail_components(
            detail_mask,
            min_area=max(6, int(total * 0.00005)),
            max_area=max(80, int(total * 0.018)),
        )

        detail_mask = self._cleanup_mask(
            detail_mask,
            min_area=max(6, int(total * 0.00005)),
            close_kernel_size=3,
            open_kernel_size=2,
        )

        detail_area = int(detail_mask.sum())

        if detail_area < max(10, int(total * 0.00008)):
            return None

        color = self._dominant_bgr_color(image, detail_mask)

        if self._hex_luminance(color) > 95:
            color = "#2B2523"

        return {
            "mask": detail_mask,
            "label": "subject_details",
            "color": color,
            "area": detail_area,
        }

    def _filter_detail_components(
        self,
        mask: np.ndarray,
        min_area: int,
        max_area: int,
    ) -> np.ndarray:
        """
        Залишає невеликі детальні компоненти, відкидає великі плями.
        """
        mask_u8 = mask.astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8,
        )

        result = np.zeros_like(mask_u8)

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]

            if min_area <= area <= max_area:
                result[labels == label_id] = 1

        return result.astype(bool)

    # ══════════════════════════════════════════════════════════════
    # GRABCUT
    # Legacy для майбутнього Portrait Mode.
    # ══════════════════════════════════════════════════════════════

    def _refine_with_grabcut(
        self,
        image: np.ndarray,
        subject_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """GrabCut уточнення SAM2 маски. Legacy для Portrait Mode."""
        try:
            h, w = image.shape[:2]

            if subject_mask is None or subject_mask.sum() == 0:
                return None

            gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
            gc_mask[subject_mask] = cv2.GC_PR_FGD

            kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

            sure_fg = cv2.erode(
                subject_mask.astype(np.uint8),
                kernel_fg,
                iterations=1,
            ).astype(bool)

            gc_mask[sure_fg] = cv2.GC_FGD

            border = max(4, int(min(h, w) * 0.025))

            gc_mask[:border, :] = cv2.GC_BGD
            gc_mask[-border:, :] = cv2.GC_BGD
            gc_mask[:, :border] = cv2.GC_BGD
            gc_mask[:, -border:] = cv2.GC_BGD

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            cv2.grabCut(
                image,
                gc_mask,
                None,
                bgd_model,
                fgd_model,
                3,
                cv2.GC_INIT_WITH_MASK,
            )

            refined = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                1,
                0,
            ).astype(bool)

            refined = self._cleanup_mask(
                refined,
                min_area=max(40, int(h * w * 0.0015)),
            )

            refined = self._crop_mask_to_subject_window(refined)

            return refined

        except Exception as e:
            logger.warning("GrabCut refinement failed: %s", e)
            return None

    # ══════════════════════════════════════════════════════════════
    # MASK UTILITIES
    # ══════════════════════════════════════════════════════════════

    def _fill_holes(self, mask: np.ndarray, close_kernel_size: int = 5) -> np.ndarray:
        """
        Заповнює внутрішні порожнини в масці.
        """
        mask_u8 = mask.astype(np.uint8) * 255

        flood = mask_u8.copy()
        padded = cv2.copyMakeBorder(
            flood,
            1,
            1,
            1,
            1,
            cv2.BORDER_CONSTANT,
            value=0,
        )

        cv2.floodFill(padded, None, (0, 0), 255)

        flood_filled = padded[1:-1, 1:-1]

        holes = (mask_u8 == 0) & (flood_filled == 0)

        filled = mask_u8.copy()
        filled[holes] = 255

        if close_kernel_size >= 3:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (close_kernel_size, close_kernel_size),
            )
            filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)

        return filled > 0

    def _cleanup_mask(
        self,
        mask: np.ndarray,
        min_area: int,
        close_kernel_size: int = 7,
        open_kernel_size: int = 3,
    ) -> np.ndarray:
        """Морфологічний cleanup + видалення малих компонентів."""
        mask_u8 = mask.astype(np.uint8) * 255

        close_kernel_size = max(1, int(close_kernel_size))
        open_kernel_size = max(1, int(open_kernel_size))

        if close_kernel_size % 2 == 0:
            close_kernel_size += 1

        if open_kernel_size % 2 == 0:
            open_kernel_size += 1

        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel_size, close_kernel_size),
        )
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_kernel_size, open_kernel_size),
        )

        cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (cleaned > 0).astype(np.uint8),
            connectivity=8,
        )

        result = np.zeros_like(cleaned, dtype=np.uint8)

        largest_area = 0
        largest_label = None

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]

            if area > largest_area:
                largest_area = area
                largest_label = label_id

            if area >= min_area:
                result[labels == label_id] = 255

        if result.sum() == 0 and largest_label is not None:
            result[labels == largest_label] = 255

        return result > 0

    def _crop_mask_to_subject_window(self, mask: np.ndarray) -> np.ndarray:
        """
        Legacy: обмежує маску портретним вікном.
        Зараз не використовується full-image Semantic Mode.
        """
        if mask is None or mask.sum() == 0:
            return mask

        h, w = mask.shape[:2]
        ys, xs = np.where(mask)

        if len(xs) == 0 or len(ys) == 0:
            return mask

        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())

        subject_w = x_max - x_min + 1
        subject_h = y_max - y_min + 1

        cx = int(np.mean(xs))

        left_pad = int(subject_w * 0.18)
        right_pad = int(subject_w * 0.08)
        top_pad = int(subject_h * 0.06)
        bottom_pad = int(subject_h * 0.14)

        crop_x1 = max(0, x_min - left_pad)
        crop_x2 = min(w, x_max + right_pad)
        crop_y1 = max(0, y_min - top_pad)
        crop_y2 = min(h, y_max + bottom_pad)

        crop_x2 = min(crop_x2, cx + int(w * 0.30))
        crop_x1 = max(crop_x1, cx - int(w * 0.42))

        window = np.zeros_like(mask, dtype=bool)
        window[crop_y1:crop_y2, crop_x1:crop_x2] = True

        cropped = mask & window

        if cropped.sum() < mask.sum() * 0.42:
            return mask

        return cropped

    def _remove_empty_layers(self, layers: List[Dict]) -> List[Dict]:
        result = []

        for layer in layers:
            area = int(layer["mask"].sum())

            if area <= 0:
                continue

            layer["area"] = area
            result.append(layer)

        return result

    def _assign_nearest_center(
        self,
        pixels: np.ndarray,
        centers: np.ndarray,
        chunk_size: int = 120000,
    ) -> np.ndarray:
        labels = np.empty((len(pixels),), dtype=np.int32)
        centers_f = centers.astype(np.float32)

        for start in range(0, len(pixels), chunk_size):
            end = min(start + chunk_size, len(pixels))
            chunk = pixels[start:end].astype(np.float32)

            distances = np.sum(
                (chunk[:, None, :] - centers_f[None, :, :]) ** 2,
                axis=2,
            )

            labels[start:end] = np.argmin(distances, axis=1)

        return labels

    def _touches_border(self, mask: np.ndarray, margin: int = 2) -> bool:
        h, w = mask.shape

        return bool(
            mask[:margin, :].any()
            or mask[h - margin:, :].any()
            or mask[:, :margin].any()
            or mask[:, w - margin:].any()
        )

    def _border_overlap_ratio(self, mask: np.ndarray) -> float:
        h, w = mask.shape
        area = int(mask.sum())

        if area == 0:
            return 0.0

        margin = max(5, int(min(h, w) * 0.035))

        border = np.zeros((h, w), dtype=bool)
        border[:margin, :] = True
        border[-margin:, :] = True
        border[:, :margin] = True
        border[:, -margin:] = True

        return int(np.logical_and(mask, border).sum()) / area

    def _mask_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        ys, xs = np.where(mask)

        if len(xs) == 0:
            return 0.0, 0.0

        return float(xs.mean()), float(ys.mean())

    def _mask_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask)

        if len(xs) == 0:
            return 0, 0, 0, 0

        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _mask_overlap_ratio(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        return float(np.logical_and(mask_a, mask_b).sum()) / max(1, int(mask_a.sum()))

    # ══════════════════════════════════════════════════════════════
    # COLOR HELPERS
    # ══════════════════════════════════════════════════════════════

    def _dominant_bgr_color(self, image: np.ndarray, mask: np.ndarray) -> str:
        """
        Медіанний колір пікселів під маскою.
        Медіана стійкіша за середнє значення.
        """
        pixels = image[mask]

        if len(pixels) == 0:
            return "#888888"

        median = np.median(pixels, axis=0).astype(int)
        b, g, r = int(median[0]), int(median[1]), int(median[2])

        return f"#{r:02X}{g:02X}{b:02X}"

    def _mean_bgr_color(self, image: np.ndarray, mask: np.ndarray) -> str:
        pixels = image[mask]

        if len(pixels) == 0:
            return "#888888"

        mean = np.mean(pixels, axis=0)
        b, g, r = [int(np.clip(round(value), 0, 255)) for value in mean]

        return f"#{r:02X}{g:02X}{b:02X}"

    def _lab_to_hex(self, lab_center: np.ndarray) -> str:
        lab_pixel = np.uint8([[lab_center]])
        bgr = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)[0, 0]

        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])

        return f"#{r:02X}{g:02X}{b:02X}"

    def _hex_luminance(self, hex_color: str) -> float:
        hex_value = hex_color.lstrip("#")

        if len(hex_value) != 6:
            return 0.0

        r = int(hex_value[0:2], 16)
        g = int(hex_value[2:4], 16)
        b = int(hex_value[4:6], 16)

        return 0.299 * r + 0.587 * g + 0.114 * b

    def _default_recommendation(self) -> Dict:
        return {
            "mode": "illustration",
            "tolerance": 1.0,
            "max_layers": 10,
            "reason": "Не вдалося проаналізувати зображення.",
            "tips": [
                "Спробуй Tolerance 0.7 і 10–14 шарів як відправну точку."
            ],
            "debug": {},
        }
