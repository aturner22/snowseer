"""Spatial-diversity gate tests for src/homography.py.

Synthesises MatchResult fixtures with controlled keypoint distributions
(clustered vs spread) and verifies:
- The diversity metric is computed correctly.
- min_spatial_diversity rejects clustered inlier sets.
- Default (min_spatial_diversity=None) preserves v2 behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_matches(kpts0: np.ndarray, kpts1: np.ndarray) -> "MatchResult":
    from src.matching import MatchResult
    n = len(kpts0)
    return MatchResult(
        kpts0=kpts0.astype(np.float32),
        kpts1=kpts1.astype(np.float32),
        confidence=np.ones(n, dtype=np.float32),
    )


def _spread_synthetic_matches(n: int = 40, w: int = 1024, h: int = 768) -> "MatchResult":
    """N matches spread roughly across the lower half of img0 + identity-warped to img1."""
    rng = np.random.default_rng(42)
    xs = rng.uniform(0.05 * w, 0.95 * w, n)
    ys = rng.uniform(0.5 * h, 0.95 * h, n)
    kpts0 = np.stack([xs, ys], axis=1)
    # Identity warp: kpts1 = kpts0 + tiny jitter (so RANSAC finds H=I).
    kpts1 = kpts0 + rng.normal(0, 0.5, kpts0.shape)
    return _make_matches(kpts0, kpts1)


def _clustered_synthetic_matches(n: int = 40, w: int = 1024, h: int = 768) -> "MatchResult":
    """N matches all packed into a 30x30 px corner of img0 + identity-warped."""
    rng = np.random.default_rng(7)
    xs = rng.uniform(10, 40, n)
    ys = rng.uniform(0.85 * h, 0.85 * h + 30, n)
    kpts0 = np.stack([xs, ys], axis=1)
    kpts1 = kpts0 + rng.normal(0, 0.5, kpts0.shape)
    return _make_matches(kpts0, kpts1)


def test_spread_matches_have_high_diversity() -> None:
    from src.homography import estimate
    matches = _spread_synthetic_matches()
    res = estimate(matches, (768, 1024), (768, 1024))
    assert res.H is not None
    # Bbox should cover most of the lower-half region — diversity well above 0.1.
    assert res.spatial_diversity_frac > 0.1, f"got {res.spatial_diversity_frac}"
    assert res.n_inliers >= 8


def test_clustered_matches_have_low_diversity() -> None:
    from src.homography import estimate
    matches = _clustered_synthetic_matches()
    res = estimate(matches, (768, 1024), (768, 1024))
    # 30x30 / (1024*768) ≈ 0.001
    assert res.spatial_diversity_frac < 0.005, f"got {res.spatial_diversity_frac}"


def test_min_spatial_diversity_rejects_clustered() -> None:
    """When min_spatial_diversity is set, a clustered fit returns H=None
    even though RANSAC technically found a valid set of inliers."""
    from src.homography import estimate
    matches = _clustered_synthetic_matches()
    res = estimate(matches, (768, 1024), (768, 1024), min_spatial_diversity=0.05)
    assert res.H is None
    # But the inlier mask + diversity score are still recorded for diagnostics.
    assert res.n_inliers > 0
    assert res.spatial_diversity_frac < 0.05


def test_min_spatial_diversity_admits_spread() -> None:
    from src.homography import estimate
    matches = _spread_synthetic_matches()
    res = estimate(matches, (768, 1024), (768, 1024), min_spatial_diversity=0.05)
    assert res.H is not None
    assert res.spatial_diversity_frac >= 0.05


def test_default_min_spatial_diversity_none_preserves_v2() -> None:
    """Without min_spatial_diversity set, behaviour is identical to v2 —
    even clustered fits return a non-None H."""
    from src.homography import estimate
    matches = _clustered_synthetic_matches()
    res = estimate(matches, (768, 1024), (768, 1024))
    assert res.H is not None  # v2 would have accepted; we still do


def test_diversity_zero_for_too_few_inliers() -> None:
    from src.homography import _compute_spatial_diversity
    kpts = np.array([[10.0, 10.0]], dtype=np.float32)
    mask = np.array([True], dtype=bool)
    assert _compute_spatial_diversity(kpts, mask, (768, 1024)) == 0.0


def test_diversity_unit_for_corner_to_corner() -> None:
    """Inliers at all four image corners → bbox covers the whole image."""
    from src.homography import _compute_spatial_diversity
    kpts = np.array([
        [0.0, 0.0], [1024.0, 0.0], [0.0, 768.0], [1024.0, 768.0],
    ], dtype=np.float32)
    mask = np.ones(4, dtype=bool)
    div = _compute_spatial_diversity(kpts, mask, (768, 1024))
    assert abs(div - 1.0) < 1e-6
