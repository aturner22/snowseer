"""Road segmentation via Segformer-B0 pretrained on Cityscapes.

Cityscapes has 19 semantic classes; class 0 is 'road'. The model is trained
exclusively on clear-weather European street imagery — no snow. Used frozen
as the road-prior generator on clear reference frames.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

_MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
_ROAD_CLASS_ID = 0
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoadSegmenter:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or _DEVICE
        self._processor: SegformerImageProcessor | None = None
        self._model: SegformerForSemanticSegmentation | None = None

    def _load(self) -> None:
        if self._processor is None:
            self._processor = SegformerImageProcessor.from_pretrained(_MODEL_ID)
        if self._model is None:
            self._model = (
                SegformerForSemanticSegmentation.from_pretrained(_MODEL_ID).eval().to(self.device)
            )

    @torch.inference_mode()
    def segment_road(self, img_rgb: np.ndarray) -> np.ndarray:
        """Return a binary (H, W) uint8 mask: 1 where road, 0 elsewhere.

        Output is in the original image resolution.
        """
        self._load()
        h, w = img_rgb.shape[:2]
        pil = Image.fromarray(img_rgb)
        inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
        logits = self._model(**inputs).logits  # (1, C, h', w')
        # Upsample to original resolution.
        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
        return (pred == _ROAD_CLASS_ID).astype(np.uint8)
