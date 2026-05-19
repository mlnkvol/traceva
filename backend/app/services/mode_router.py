import logging

import cv2
import numpy as np


logger = logging.getLogger(__name__)


def is_logo_candidate(image: np.ndarray) -> bool:
    """
    Conservative auto-mode heuristic shared by API routes and offline eval.
    """
    try:
        if image is None or image.size == 0:
            return False

        h, w = image.shape[:2]
        max_side = max(h, w)

        if max_side > 256:
            scale = 256 / max_side
            small = cv2.resize(
                image,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = image.copy()

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        small_h, small_w = small.shape[:2]
        total_pixels = small_h * small_w

        quantized_rgb = (small // 32).astype(np.uint8)
        unique_rgb_colors = len(np.unique(quantized_rgb.reshape(-1, 3), axis=0))
        unique_gray_levels = len(np.unique((gray // 16).astype(np.uint8)))

        sat_mean = float(np.mean(hsv[:, :, 1]))
        sat_std = float(np.std(hsv[:, :, 1]))
        gray_std = float(np.std(gray))

        threshold_value, binary_inv = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        fg_ratio = float(np.mean(binary_inv > 0))

        if fg_ratio <= 0.01 or fg_ratio >= 0.85:
            logger.info("Auto mode: semantic because fg_ratio=%.3f", fg_ratio)
            return False

        fg_pixels = gray[binary_inv > 0]
        bg_pixels = gray[binary_inv == 0]
        if len(fg_pixels) == 0 or len(bg_pixels) == 0:
            return False

        fg_mean = float(np.mean(fg_pixels))
        bg_mean = float(np.mean(bg_pixels))
        reconstructed = np.where(binary_inv > 0, fg_mean, bg_mean).astype(np.float32)
        binary_reconstruction_error = float(
            np.mean(np.abs(gray.astype(np.float32) - reconstructed))
        )

        fg_std = float(np.std(fg_pixels))
        bg_std = float(np.std(bg_pixels))
        within_region_std = (fg_std + bg_std) / 2

        num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_inv,
            connectivity=8,
        )
        meaningful_components = sum(
            1
            for label_id in range(1, num_labels)
            if stats[label_id, cv2.CC_STAT_AREA] > total_pixels * 0.001
        )

        edges = cv2.Canny(gray, 80, 160)
        edge_ratio = float(np.mean(edges > 0))
        dark_ratio = float(np.mean(gray < 70))
        light_ratio = float(np.mean(gray > 185))
        midtone_ratio = 1.0 - dark_ratio - light_ratio

        if midtone_ratio > 0.35:
            logger.info("Auto mode: semantic because midtone_ratio=%.3f", midtone_ratio)
            return False

        looks_like_photo = (
            unique_rgb_colors > 55
            or unique_gray_levels > 22
            or binary_reconstruction_error > 18
            or within_region_std > 24
            or meaningful_components > 18
        )

        if looks_like_photo:
            logger.info(
                "Auto mode: semantic because photo-like. unique_rgb=%s, "
                "unique_gray=%s, bin_error=%.2f, within_std=%.2f, "
                "components=%s, midtone=%.3f",
                unique_rgb_colors,
                unique_gray_levels,
                binary_reconstruction_error,
                within_region_std,
                meaningful_components,
                midtone_ratio,
            )
            return False

        is_logo = (
            unique_rgb_colors <= 55
            and unique_gray_levels <= 22
            and gray_std >= 25
            and binary_reconstruction_error <= 18
            and within_region_std <= 24
            and meaningful_components <= 18
            and edge_ratio <= 0.22
            and midtone_ratio <= 0.35
            and 0.02 < fg_ratio < 0.80
        )

        logger.info(
            "Auto mode analysis: unique_rgb=%s, unique_gray=%s, sat_mean=%.2f, "
            "sat_std=%.2f, gray_std=%.2f, fg_ratio=%.3f, components=%s, "
            "edge_ratio=%.3f, midtone_ratio=%.3f, bin_error=%.2f, "
            "within_std=%.2f, threshold=%.2f, is_logo=%s",
            unique_rgb_colors,
            unique_gray_levels,
            sat_mean,
            sat_std,
            gray_std,
            fg_ratio,
            meaningful_components,
            edge_ratio,
            midtone_ratio,
            binary_reconstruction_error,
            within_region_std,
            threshold_value,
            is_logo,
        )
        return is_logo
    except Exception as exc:
        logger.warning("Could not determine image type, fallback to semantic mode: %s", exc)
        return False
