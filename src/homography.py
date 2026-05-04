"""RANSAC homography with ground-plane bias.

A single homography is exact only for a planar scene. Buildings and the road are
not coplanar. By restricting RANSAC to correspondences in the lower portion of
both images, we bias the fit toward the ground plane — which is what we need
to correctly transfer the road mask.
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
    dashboard_y_frac: float = 0.85,
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
