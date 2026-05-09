"""DISK + LightGlue feature matching via kornia.

DISK is pretrained on MegaDepth (outdoor scenes). LightGlue is trained
to match DISK features. Both are used frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import kornia.feature as KF
import numpy as np
import torch


@dataclass
class MatchResult:
    kpts0: np.ndarray  # (N, 2) — (x, y) pixel coordinates in image 0
    kpts1: np.ndarray  # (N, 2) — (x, y) pixel coordinates in image 1
    confidence: np.ndarray  # (N,) — match confidence in [0, 1]


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MAX_KPTS = 2048


class Matcher:
    """Lazily-loaded DISK + LightGlue matcher."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or _DEVICE
        self._extractor: KF.DISK | None = None
        self._matcher: KF.LightGlueMatcher | None = None

    def _load(self) -> None:
        if self._extractor is None:
            self._extractor = KF.DISK.from_pretrained("depth").eval().to(self.device)
        if self._matcher is None:
            self._matcher = KF.LightGlueMatcher("disk").eval().to(self.device)

    @torch.inference_mode()
    def match(self, img_a: np.ndarray, img_b: np.ndarray) -> MatchResult:
        """Match two RGB images (HxWx3, uint8). Returns paired keypoints."""
        self._load()
        ta = _to_rgb_tensor(img_a, self.device)
        tb = _to_rgb_tensor(img_b, self.device)

        feats_a = self._extractor(ta, n=_MAX_KPTS, pad_if_not_divisible=True)[0]
        feats_b = self._extractor(tb, n=_MAX_KPTS, pad_if_not_divisible=True)[0]

        kp_a = feats_a.keypoints  # (N, 2) (x, y)
        kp_b = feats_b.keypoints
        desc_a = feats_a.descriptors  # (N, 128)
        desc_b = feats_b.descriptors

        ha, wa = ta.shape[-2:]
        hb, wb = tb.shape[-2:]
        lafs_a = KF.laf_from_center_scale_ori(kp_a[None])
        lafs_b = KF.laf_from_center_scale_ori(kp_b[None])
        dists, idxs = self._matcher(
            desc_a, desc_b, lafs_a, lafs_b, hw1=(ha, wa), hw2=(hb, wb)
        )
        idxs_np = idxs.cpu().numpy()
        kp_a_np = kp_a.cpu().numpy()
        kp_b_np = kp_b.cpu().numpy()
        kpts0 = kp_a_np[idxs_np[:, 0]]
        kpts1 = kp_b_np[idxs_np[:, 1]]
        # LightGlue returns descriptor distances in [0, ~1]; smaller is better.
        # Convert to a confidence proxy in [0, 1].
        d = dists.cpu().numpy().reshape(-1)
        if d.size:
            d = np.clip(d, 0.0, 1.0)
            conf = 1.0 - d
        else:
            conf = np.zeros(0, dtype=np.float32)
        return MatchResult(kpts0=kpts0, kpts1=kpts1, confidence=conf)


def _to_rgb_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    t = torch.from_numpy(img).to(device).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)


def draw_matches(
    img_a: np.ndarray,
    img_b: np.ndarray,
    result: MatchResult,
    inlier_mask: np.ndarray | None = None,
    out_path: str | Path | None = None,
    max_inliers: int = 10,
) -> np.ndarray:
    """Side-by-side visualisation. Draws up to `max_inliers` correspondences
    in green between the two images. Rejected (non-inlier) matches are not
    drawn.
    """
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    h = max(h_a, h_b)
    canvas = np.zeros((h, w_a + w_b, 3), dtype=np.uint8)
    canvas[:h_a, :w_a] = img_a
    canvas[:h_b, w_a : w_a + w_b] = img_b

    n = len(result.kpts0)
    if inlier_mask is None:
        inlier_mask = np.ones(n, dtype=bool)
    inlier_mask = inlier_mask.astype(bool)

    rng = np.random.RandomState(0)
    in_idx = np.where(inlier_mask)[0]
    if len(in_idx) > max_inliers:
        in_idx = rng.choice(in_idx, size=max_inliers, replace=False)

    in_colour = (40, 200, 80)
    for i in in_idx:
        xa, ya = result.kpts0[i]
        xb, yb = result.kpts1[i]
        pa = (int(round(xa)), int(round(ya)))
        pb = (int(round(xb)) + w_a, int(round(yb)))
        cv2.line(canvas, pa, pb, in_colour, 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pa, 3, in_colour, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pb, 3, in_colour, -1, lineType=cv2.LINE_AA)

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return canvas
