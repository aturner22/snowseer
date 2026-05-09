"""RANSAC homography.

A single homography is exact only for a planar scene; buildings and the
road are not coplanar. Restricting RANSAC to correspondences in the
lower portion of both images biases the fit toward the ground plane, so
the warped road mask lands on the road instead of an off-plane structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .matching import MatchResult


@dataclass
class HomographyResult:
    H: np.ndarray | None  # (3, 3) homography from img0 -> img1, or None on failure
    inlier_mask: np.ndarray  # (N,) bool over the input MatchResult, True if used
    n_inliers: int
    used_ground_plane_restriction: bool


def estimate(
    matches: MatchResult,
    img0_shape: tuple[int, int],
    img1_shape: tuple[int, int],
    *,
    ground_plane_y_frac: float = 0.5,
    dashboard_y_frac: float = 1.0,
    min_matches: int = 8,
    ransac_thresh_px: float = 3.0,
    confidence_thresh: float = 0.0,
) -> HomographyResult:
    """Estimate H from img0 -> img1 using ground-plane matches when available.

    img*_shape: (H, W) of each image.
    ground_plane_y_frac: keep matches where y > frac * height (drop sky / building tops).
    dashboard_y_frac: drop matches where y > frac * height (the bottom strip is
        the camera-vehicle's dashboard, not the road plane; Mapillary uploads
        from cars routinely have a 10-15% dashboard band that confuses RANSAC).
    """
    n = len(matches.kpts0)
    full_mask = np.ones(n, dtype=bool)

    if confidence_thresh > 0:
        full_mask &= matches.confidence >= confidence_thresh

    h0, _ = img0_shape
    h1, _ = img1_shape
    y_min_0 = ground_plane_y_frac * h0
    y_min_1 = ground_plane_y_frac * h1
    y_max_0 = dashboard_y_frac * h0
    y_max_1 = dashboard_y_frac * h1
    ground_mask = (
        (matches.kpts0[:, 1] >= y_min_0) & (matches.kpts0[:, 1] <= y_max_0)
        & (matches.kpts1[:, 1] >= y_min_1) & (matches.kpts1[:, 1] <= y_max_1)
    )

    candidate_mask = full_mask & ground_mask
    used_restriction = candidate_mask.sum() >= min_matches
    if not used_restriction:
        candidate_mask = full_mask

    if candidate_mask.sum() < min_matches:
        return HomographyResult(H=None, inlier_mask=np.zeros(n, dtype=bool), n_inliers=0, used_ground_plane_restriction=used_restriction)

    src = matches.kpts0[candidate_mask].astype(np.float32)
    dst = matches.kpts1[candidate_mask].astype(np.float32)

    H, ransac_mask = cv2.findHomography(
        src, dst,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransac_thresh_px,
        confidence=0.999,
        maxIters=10_000,
    )
    if H is None or ransac_mask is None:
        return HomographyResult(H=None, inlier_mask=np.zeros(n, dtype=bool), n_inliers=0, used_ground_plane_restriction=used_restriction)

    ransac_mask = ransac_mask.ravel().astype(bool)
    final_mask = np.zeros(n, dtype=bool)
    candidate_indices = np.where(candidate_mask)[0]
    final_mask[candidate_indices[ransac_mask]] = True

    return HomographyResult(
        H=H,
        inlier_mask=final_mask,
        n_inliers=int(final_mask.sum()),
        used_ground_plane_restriction=used_restriction,
    )


def refine_iteratively(
    matches: MatchResult,
    H_initial: np.ndarray,
    img0_shape: tuple[int, int],
    img1_shape: tuple[int, int],
    road_mask_clear: np.ndarray,
    *,
    max_iters: int = 2,
    ransac_thresh_px: float = 3.0,
    min_matches: int = 8,
) -> HomographyResult:
    """Refine the homography by re-fitting on matches whose snow keypoints
    fall inside the warped road region.

    The initial homography (from `estimate(...)`) uses a generic lower-
    image-half ground-plane restriction. When that restriction does not
    engage cleanly (tunnel scenes where most of the lower half is wall,
    not road) the fit drifts. This refinement uses the road segmentation
    itself to pick matches that should be on the road, then re-fits.

    Returns a HomographyResult with the refined H and an inlier_mask over
    the original matches so callers can compare against the initial fit.
    """
    n = len(matches.kpts0)
    if H_initial is None:
        return HomographyResult(H=None, inlier_mask=np.zeros(n, dtype=bool), n_inliers=0, used_ground_plane_restriction=False)

    H = H_initial.copy()
    h0, w0 = img0_shape
    final_mask = np.zeros(n, dtype=bool)

    for _ in range(max_iters):
        # Warp clear road mask -> snow image (img0) space.
        H_inv = np.linalg.inv(H)
        road_in_snow = cv2.warpPerspective(
            road_mask_clear.astype(np.uint8), H_inv, (w0, h0), flags=cv2.INTER_NEAREST
        )
        # Pick matches whose img0 keypoint sits inside the warped road region.
        kpts0_int = matches.kpts0.astype(int)
        in_bounds = (
            (kpts0_int[:, 0] >= 0) & (kpts0_int[:, 0] < w0)
            & (kpts0_int[:, 1] >= 0) & (kpts0_int[:, 1] < h0)
        )
        in_road = np.zeros(n, dtype=bool)
        if in_bounds.any():
            xs = kpts0_int[in_bounds, 0].clip(0, w0 - 1)
            ys = kpts0_int[in_bounds, 1].clip(0, h0 - 1)
            in_road[in_bounds] = road_in_snow[ys, xs] > 0
        if in_road.sum() < min_matches:
            break
        src = matches.kpts0[in_road].astype(np.float32)
        dst = matches.kpts1[in_road].astype(np.float32)
        H_new, ransac_mask = cv2.findHomography(
            src, dst, method=cv2.USAC_MAGSAC, ransacReprojThreshold=ransac_thresh_px,
            confidence=0.999, maxIters=10_000,
        )
        if H_new is None or ransac_mask is None:
            break
        # Translate the per-subset RANSAC mask back into a full-N mask.
        new_full_mask = np.zeros(n, dtype=bool)
        idxs = np.where(in_road)[0]
        new_full_mask[idxs[ransac_mask.ravel().astype(bool)]] = True
        # Stop early if inlier set has stabilised.
        if np.array_equal(new_full_mask, final_mask):
            break
        H = H_new
        final_mask = new_full_mask

    return HomographyResult(
        H=H, inlier_mask=final_mask, n_inliers=int(final_mask.sum()),
        used_ground_plane_restriction=True,
    )
