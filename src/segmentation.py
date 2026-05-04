"""Road segmentation via Mask2Former-tiny pretrained on Cityscapes.

Cityscapes has 19 semantic classes; class 0 is 'road'. The model is trained
exclusively on clear-weather European street imagery — no snow. Used frozen
as the road-prior generator on clear reference frames.

Mask2Former-tiny replaced Segformer-B0 in v0.3 because Segformer-B0 over-
predicted road class on tunnel walls and under-predicted on shadowed asphalt.
Mask2Former is the standard Cityscapes upgrade — sharper, less hallucinated.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

_MODEL_ID = "facebook/mask2former-swin-tiny-cityscapes-semantic"
_ROAD_CLASS_ID = 0
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class RoadSegmenter:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or _DEVICE
        self._processor: Mask2FormerImageProcessor | None = None
        self._model: Mask2FormerForUniversalSegmentation | None = None

    def _load(self) -> None:
        if self._processor is None:
            self._processor = Mask2FormerImageProcessor.from_pretrained(_MODEL_ID)
        if self._model is None:
            self._model = (
                Mask2FormerForUniversalSegmentation.from_pretrained(_MODEL_ID).eval().to(self.device)
            )

    @torch.inference_mode()
    def segment_road(
        self,
        img_rgb: np.ndarray,
        *,
        dashboard_y_frac: float = 1.0,
    ) -> np.ndarray:
        """Return a binary (H, W) uint8 mask: 1 where road, 0 elsewhere.

        Output is in the original image resolution. Optionally zeros pixels
        below `dashboard_y_frac * H` (default 1.0 = no cutoff). The cutoff
        was useful with Segformer-B0 which over-predicted road class on flat
        blue dashboards; Mask2Former does not have this failure mode and
        many Mapillary uploads use roof-mounted cameras where the road
        legitimately extends to the bottom of the frame, so we do not cut by
        default. Override per-call if a specific upload needs it.
        """
        self._load()
        h, w = img_rgb.shape[:2]
        pil = Image.fromarray(img_rgb)
        inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
        outputs = self._model(**inputs)

        # Mask2Former's processor handles the per-query mask combination and
        # returns a per-pixel class index at the requested target size.
        seg = self._processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(h, w)]
        )[0]
        if hasattr(seg, "cpu"):
            seg = seg.cpu().numpy()
        else:
            seg = np.asarray(seg)
        mask = (seg == _ROAD_CLASS_ID).astype(np.uint8)

        cutoff = int(round(dashboard_y_frac * h))
        if cutoff < h:
            mask[cutoff:, :] = 0
        return mask
