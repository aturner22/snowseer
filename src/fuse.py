"""Mask-fusion utilities for the K-prior video pipeline.

Given K warped road masks in snow image space (one per prior), each with
an associated `valid_region` (the snow-space projection of the prior's
frame, i.e. where the prior's data exists at all), fuse them into a
single binary mask.

All masks are uint8 with values in {0, 1}.
"""

from __future__ import annotations

import cv2
import numpy as np


def edge_eroded(mask: np.ndarray, valid_region: np.ndarray, *, erosion_px: int = 20) -> np.ndarray:
    """Drop a `mask` pixel if it is within `erosion_px` of the boundary of
    `valid_region`. Removes the harsh frame-edge artefact that appears when
    the prior's frame is clipped at the snow image boundary.
    """
    if erosion_px <= 0:
        return mask
    k = max(3, erosion_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded_valid = cv2.erode(valid_region.astype(np.uint8), kernel)
    return (mask.astype(bool) & eroded_valid.astype(bool)).astype(np.uint8)


def weighted_soft_average(
    masks: list[np.ndarray],
    weights: list[float],
    valid_regions: list[np.ndarray] | None = None,
    *,
    threshold: float = 0.4,
    erosion_px: int = 8,
) -> np.ndarray:
    """Per pixel: Σ (mask_k × weight_k) / Σ weight_k where mask_k is contributed
    only where its valid_region (lightly eroded) holds. Threshold at `threshold`.
    A prior with high `weight_k` (e.g. inlier count) dominates over a weak one.
    """
    if not masks:
        return np.zeros((1, 1), dtype=np.uint8)
    h, w = masks[0].shape[:2]
    num = np.zeros((h, w), dtype=np.float32)
    den = np.zeros((h, w), dtype=np.float32)
    for i, m in enumerate(masks):
        wt = float(max(0.0, weights[i] if i < len(weights) else 1.0))
        if wt == 0.0:
            continue
        if valid_regions is not None:
            vr = edge_eroded(np.ones_like(m), valid_regions[i], erosion_px=erosion_px)
        else:
            vr = np.ones_like(m)
        contrib = (m.astype(np.float32) > 0).astype(np.float32) * vr.astype(np.float32) * wt
        den += vr.astype(np.float32) * wt
        num += contrib
    score = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
    return (score >= threshold).astype(np.uint8)


def crop_foreground(mask: np.ndarray, *, foreground_y_frac: float = 0.45) -> np.ndarray:
    """Drop mask pixels where y < `foreground_y_frac * H`. Image y
    correlates with depth (lower in frame = closer to camera), so this
    keeps only the immediate foreground and discards distant projections
    a roof-mounted plough camera should not claim as drivable.
    """
    out = mask.copy()
    h = mask.shape[0]
    cutoff = int(round(foreground_y_frac * h))
    if cutoff > 0:
        out[:cutoff] = 0
    return out
