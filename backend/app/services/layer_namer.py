import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings


logger = logging.getLogger(__name__)


@dataclass
class MaskCrop:
    image: Image.Image
    area_ratio: float
    bbox: tuple[int, int, int, int]


class LayerNamer:
    """
    Names segmentation layers with an optional open vision-language model.

    The model is loaded lazily so the vectorization pipeline still works on
    machines without VLM dependencies or local model weights.
    """

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._device = None
        self._load_failed = False

    def name_layers(self, image: np.ndarray, masks: List[Dict]) -> List[str]:
        if not masks:
            return []

        if not getattr(settings, "VLM_LAYER_NAMING_ENABLED", True):
            return self._fallback_names(masks)

        names: List[str] = []

        for index, mask_info in enumerate(masks):
            fallback = self._fallback_name(mask_info, index)

            try:
                crop = self._make_mask_crop(image, mask_info.get("mask"))
                if crop is None:
                    names.append(fallback)
                    continue

                raw_name = self._caption_crop(crop.image)
                name = self._clean_name(raw_name) or fallback
                names.append(name)
            except Exception as exc:
                logger.warning("VLM layer naming failed for layer %s: %s", index + 1, exc)
                names.append(fallback)

        return self._dedupe_names(names)

    def _caption_crop(self, crop: Image.Image) -> Optional[str]:
        if not self._ensure_model_loaded():
            return None

        prompt = getattr(settings, "VLM_LAYER_NAMING_PROMPT", "<CAPTION>")

        inputs = self._processor(text=prompt, images=crop, return_tensors="pt")
        inputs = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=int(getattr(settings, "VLM_LAYER_NAMING_MAX_TOKENS", 12)),
            do_sample=False,
        )

        if hasattr(self._processor, "batch_decode"):
            text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            text = self._processor.decode(generated_ids[0], skip_special_tokens=True)

        if hasattr(self._processor, "post_process_generation"):
            try:
                parsed = self._processor.post_process_generation(
                    text,
                    task=prompt,
                    image_size=crop.size,
                )
                if isinstance(parsed, dict):
                    text = str(parsed.get(prompt) or next(iter(parsed.values()), text))
            except Exception:
                pass

        return text

    def _ensure_model_loaded(self) -> bool:
        if self._processor is not None and self._model is not None:
            return True

        if self._load_failed:
            return False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            model_id = getattr(settings, "VLM_LAYER_NAMING_MODEL", "microsoft/Florence-2-base")
            device_setting = str(getattr(settings, "VLM_LAYER_NAMING_DEVICE", settings.DEVICE))
            self._device = "cuda" if device_setting == "cuda" and torch.cuda.is_available() else "cpu"

            self._processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
            ).to(self._device)
            self._model.eval()

            logger.info("Loaded VLM layer namer: %s on %s", model_id, self._device)
            return True
        except Exception as exc:
            self._load_failed = True
            logger.warning("VLM layer naming unavailable, using fallback names: %s", exc)
            return False

    def _make_mask_crop(self, image: np.ndarray, mask: Optional[np.ndarray]) -> Optional[MaskCrop]:
        if mask is None:
            return None

        mask_bool = mask.astype(bool)
        if int(mask_bool.sum()) == 0:
            return None

        h, w = mask_bool.shape[:2]
        ys, xs = np.where(mask_bool)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1

        pad = max(6, int(max(x2 - x1, y2 - y1) * 0.08))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop_bgr = image[y1:y2, x1:x2].copy()
        crop_mask = mask_bool[y1:y2, x1:x2]

        if crop_bgr.size == 0:
            return None

        white = np.full_like(crop_bgr, 255)
        masked = np.where(crop_mask[:, :, None], crop_bgr, white)
        rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

        max_side = int(getattr(settings, "VLM_LAYER_NAMING_CROP_SIZE", 384))
        ch, cw = rgb.shape[:2]
        if max(ch, cw) > max_side:
            scale = max_side / max(ch, cw)
            rgb = cv2.resize(
                rgb,
                (max(1, int(cw * scale)), max(1, int(ch * scale))),
                interpolation=cv2.INTER_AREA,
            )

        area_ratio = float(mask_bool.sum()) / max(1, h * w)
        return MaskCrop(Image.fromarray(rgb), area_ratio, (x1, y1, x2, y2))

    def _clean_name(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None

        value = str(text).strip()
        value = re.sub(r"<[^>]+>", " ", value)
        value = re.sub(r"^(the image shows|this is|a photo of|an image of)\s+", "", value, flags=re.I)
        value = re.sub(r"[^\w' -]+", " ", value, flags=re.UNICODE)
        words = [word for word in value.split() if word.lower() not in {"a", "an", "the"}]
        value = " ".join(words[:2]).strip(" -'")

        if not value or value.lower() in {"object", "thing", "image"}:
            return None

        return value[:48]

    def _fallback_names(self, masks: List[Dict]) -> List[str]:
        return self._dedupe_names(
            [self._fallback_name(mask_info, index) for index, mask_info in enumerate(masks)]
        )

    def _fallback_name(self, mask_info: Dict, index: int) -> str:
        label = str(mask_info.get("label") or "").strip()
        if label and not label.lower().startswith("layer"):
            return label.replace("_", " ")

        area = int(mask_info.get("area") or 0)
        if index == 0:
            return "Background"
        if area > 0:
            return f"Object {index}"
        return f"Layer {index + 1}"

    def _dedupe_names(self, names: List[str]) -> List[str]:
        counts: Dict[str, int] = {}
        result: List[str] = []

        for name in names:
            base = name or "Layer"
            key = base.lower()
            counts[key] = counts.get(key, 0) + 1
            result.append(base if counts[key] == 1 else f"{base} {counts[key]}")

        return result
