"""Mask primitives: warp, component cleanup, fusion, foreground crop, and
alpha blending.

All masks are uint8 with values in {0, 1}. Used by both the static-stills
pipeline and the per-frame video pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np


def warp_mask(
    mask_src: np.ndarray, H_src_to_dst: np.ndarray, dst_shape: tuple[int, int]
) -> np.ndarray:
    """Warp a binary mask from img0 to img1 using a homography img0 -> img1."""
    h, w = dst_shape
    return cv2.warpPerspective(
        mask_src.astype(np.uint8), H_src_to_dst, (w, h), flags=cv2.INTER_NEAREST
    )


def keep_largest_component(mask: np.ndarray, *, min_area_px: int = 500) -> np.ndarray:
    """Reduce a binary mask to its single largest connected component.

    A snow plough cares about the one drivable surface directly in front
    of it, not the long tail of small islands that warp aliasing or
    segmenter noise can leave behind. Returns an empty mask if no
    component clears `min_area_px`.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if not len(areas):
        return np.zeros_like(mask)
    largest_idx = int(np.argmax(areas)) + 1  # +1 to skip background
    largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])
    if largest_area < min_area_px:
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    out[labels == largest_idx] = 1
    return out


def edge_eroded(mask: np.ndarray, valid_region: np.ndarray, *, erosion_px: int = 20) -> np.ndarray:
    """Drop a `mask` pixel if it is within `erosion_px` of the boundary of
    `valid_region`. Removes the harsh frame-edge artefact that appears
    when the prior's frame is clipped at the snow image boundary.
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
    """Per-pixel weighted average over K warped masks, then thresholded.

    For each pixel: Σ (mask_k × weight_k) / Σ weight_k, where mask_k
    contributes only where its valid_region (lightly eroded) holds.
    A prior with high weight (e.g. inlier count) dominates over a weak
    one.
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
    """Drop mask pixels above `foreground_y_frac * H`. Image y correlates
    with depth (lower in frame = closer to camera), so this keeps only
    the immediate foreground and discards distant projections that a
    roof-mounted plough camera should not claim as drivable.
    """
    out = mask.copy()
    h = mask.shape[0]
    cutoff = int(round(foreground_y_frac * h))
    if cutoff > 0:
        out[:cutoff] = 0
    return out


def alpha_blend(
    img: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 100),
    alpha: float = 0.45,
    edge_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Alpha-blend a binary mask onto the image with an outlined edge.

    `edge_color` defaults to a darker, saturated variant of `color` so
    the road boundary is visible on both snow and asphalt.
    """
    out = img.copy()
    color_arr = np.array(color, dtype=np.uint8)
    blended = (alpha * color_arr + (1 - alpha) * out[mask > 0]).astype(np.uint8)
    out[mask > 0] = blended
    if edge_color is None:
        edge_color = tuple(max(int(c) - 80, 0) for c in color)
    edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
    out[edges > 0] = np.array(edge_color, dtype=np.uint8)
    return out
