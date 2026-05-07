"""Road segmentation via Mask2Former-tiny pretrained on Cityscapes.

Cityscapes has 19 semantic classes; class 0 is 'road'. The model is trained
exclusively on clear-weather European street imagery — no snow. Used frozen
as the road-prior generator on clear reference frames.

Mask2Former-tiny replaced Segformer-B0 in v0.3 because Segformer-B0 over-
predicted road class on tunnel walls and under-predicted on shadowed asphalt.
Mask2Former is the standard Cityscapes upgrade — sharper, less hallucinated.
"""

from __future__ import annotations

import cv2
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
        prob_threshold: float | None = None,
        morph_radius: int = 0,
    ) -> np.ndarray:
        """Return a binary (H, W) uint8 mask: 1 where road, 0 elsewhere.

        Output is in the original image resolution. Optionally zeros pixels
        below `dashboard_y_frac * H` (default 1.0 = no cutoff).

        `prob_threshold` (default None = argmax behaviour): if set, keep
        only pixels where the road class's per-pixel score (from Mask2Former's
        query/mask aggregation) exceeds this threshold. Tightens the mask
        on summer scenes where the segmenter is confident on the lane but
        leaks onto adjacent surfaces.

        `morph_radius` (default 0 = off): if > 0, run an opening then
        closing pass with a circular kernel of this radius after thresholding.
        Suppresses thin extrusions and one- or two-pixel jaggies that warp
        into amplified jitter on the snow side.
        """
        self._load()
        h, w = img_rgb.shape[:2]
        pil = Image.fromarray(img_rgb)
        inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
        outputs = self._model(**inputs)

        if prob_threshold is None:
            seg = self._processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(h, w)]
            )[0]
            if hasattr(seg, "cpu"):
                seg = seg.cpu().numpy()
            else:
                seg = np.asarray(seg)
            mask = (seg == _ROAD_CLASS_ID).astype(np.uint8)
        else:
            class_logits = outputs.class_queries_logits          # (1, Q, C+1)
            mask_logits = outputs.masks_queries_logits           # (1, Q, h', w')
            class_probs = class_logits.softmax(dim=-1)[..., :-1]  # drop no-object
            mask_probs = mask_logits.sigmoid()                    # (1, Q, h', w')
            semantic = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
            semantic = torch.nn.functional.interpolate(
                semantic, size=(h, w), mode="bilinear", align_corners=False,
            )
            road_prob = semantic[0, _ROAD_CLASS_ID].cpu().numpy()
            mask = (road_prob > prob_threshold).astype(np.uint8)

        if morph_radius > 0:
            k = 2 * morph_radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cutoff = int(round(dashboard_y_frac * h))
        if cutoff < h:
            mask[cutoff:, :] = 0
        return mask
