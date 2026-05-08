"""Mask-fusion strategies for the multi-prior pipeline.

Given K warped road masks in snow image space (one per prior), each with an
associated `valid_region` (the snow-space projection of the prior's frame —
i.e., where the prior's data exists at all), fuse them into a single binary
mask via three independent strategies. We expose all three so we can A/B them
in the curator.

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


def union_with_edge_erosion(
    masks: list[np.ndarray],
    valid_regions: list[np.ndarray],
    *,
    erosion_px: int = 20,
) -> np.ndarray:
    """Logical OR of K masks after each is edge-eroded against its own
    `valid_region`. Recovers the most road area; rejects per-prior frame edges.
    """
    if not masks:
        return np.zeros((1, 1), dtype=np.uint8)
    out = np.zeros_like(masks[0])
    for m, v in zip(masks, valid_regions):
        out = out | edge_eroded(m, v, erosion_px=erosion_px)
    return out.astype(np.uint8)


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
        # Den only counted where this prior was valid.
        den += vr.astype(np.float32) * wt
        num += contrib
    score = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
    return (score >= threshold).astype(np.uint8)


def majority_vote(
    masks: list[np.ndarray],
    valid_regions: list[np.ndarray] | None = None,
    *,
    erosion_px: int = 8,
) -> np.ndarray:
    """Per pixel road if ≥ ⌊K_valid/2⌋+1 priors that *cover* the pixel agree.
    K_valid is the number of priors whose valid_region includes that pixel —
    so corners of the snow image with sparse coverage still get a fair vote.
    """
    if not masks:
        return np.zeros((1, 1), dtype=np.uint8)
    h, w = masks[0].shape[:2]
    votes = np.zeros((h, w), dtype=np.int16)
    coverage = np.zeros((h, w), dtype=np.int16)
    for i, m in enumerate(masks):
        if valid_regions is not None:
            vr = edge_eroded(np.ones_like(m), valid_regions[i], erosion_px=erosion_px)
        else:
            vr = np.ones_like(m)
        votes += (m.astype(bool) & vr.astype(bool)).astype(np.int16)
        coverage += vr.astype(np.int16)
    needed = (coverage // 2) + 1
    return ((votes >= needed) & (coverage > 0)).astype(np.uint8)


def crop_foreground(mask: np.ndarray, *, foreground_y_frac: float = 0.45) -> np.ndarray:
    """Drop mask pixels where y < `foreground_y_frac * H`. The user's
    'crop the back of the overlay' requirement: priors typically extend
    further into the distance than the snow camera should claim. Image y
    correlates with depth (lower in frame = closer to camera) so we keep
    only the immediate foreground.
    """
    out = mask.copy()
    h = mask.shape[0]
    cutoff = int(round(foreground_y_frac * h))
    if cutoff > 0:
        out[:cutoff] = 0
    return out
