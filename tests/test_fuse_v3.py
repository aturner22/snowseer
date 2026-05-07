"""v3 fusion tests for src/fuse.py.

Covers the new outlier-drop helper (v3.P.4) and the weighted-soft-average
behaviour under the v3 weight strategies (v3.P.2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _square_mask(h: int, w: int, x0: int, y0: int, side: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y0 + side, x0:x0 + side] = 1
    return m


def _full_valid(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=np.uint8)


def test_drop_outlier_keeps_all_when_fewer_than_three() -> None:
    from src.fuse import drop_outlier_priors
    h, w = 100, 100
    m1 = _square_mask(h, w, 10, 10, 30)
    m2 = _square_mask(h, w, 60, 60, 30)
    masks, valids, weights = drop_outlier_priors(
        [m1, m2], [_full_valid(h, w)] * 2, [1.0, 1.0],
    )
    assert len(masks) == 2  # < 3 priors → no drop


def test_drop_outlier_removes_disagreer() -> None:
    from src.fuse import drop_outlier_priors
    h, w = 100, 100
    # Three priors: two agree on a region near (10..50, 10..50);
    # one disagrees and points at a corner.
    consensus_a = _square_mask(h, w, 10, 10, 40)
    consensus_b = _square_mask(h, w, 12, 12, 40)
    rogue = _square_mask(h, w, 70, 70, 20)
    masks, valids, weights = drop_outlier_priors(
        [consensus_a, consensus_b, rogue],
        [_full_valid(h, w)] * 3,
        [50.0, 50.0, 50.0],
        iou_threshold=0.15,
    )
    assert len(masks) == 2
    # The two kept should be the consensus pair (high IoU).
    iou = (masks[0].astype(bool) & masks[1].astype(bool)).sum() / (
        masks[0].astype(bool) | masks[1].astype(bool)
    ).sum()
    assert iou > 0.5


def test_drop_outlier_keeps_all_when_consensus() -> None:
    from src.fuse import drop_outlier_priors
    h, w = 100, 100
    # Three priors all overlapping at (10..50, 10..50)
    m1 = _square_mask(h, w, 10, 10, 40)
    m2 = _square_mask(h, w, 12, 12, 40)
    m3 = _square_mask(h, w, 11, 11, 40)
    masks, _, _ = drop_outlier_priors(
        [m1, m2, m3], [_full_valid(h, w)] * 3, [1.0, 1.0, 1.0],
        iou_threshold=0.15,
    )
    assert len(masks) == 3  # all kept


def test_weighted_soft_average_respects_weights() -> None:
    """A high-weight mask dominates a low-weight one in the threshold."""
    from src.fuse import weighted_soft_average
    h, w = 50, 50
    m_strong = _square_mask(h, w, 10, 10, 20)
    m_weak = _square_mask(h, w, 30, 30, 20)
    fused = weighted_soft_average(
        [m_strong, m_weak],
        [50.0, 1.0],  # strong has 50× the weight
        [_full_valid(h, w), _full_valid(h, w)],
        threshold=0.4,
    )
    # Strong region (10..30, 10..30) should be in; weak (30..50, 30..50) should not.
    assert fused[15, 15] == 1
    assert fused[45, 45] == 0


def test_pipeline_v_imports_with_v3_kwargs() -> None:
    """run_track accepts the new v3 kwargs without import errors."""
    import inspect
    from src.video_runtime.pipeline_v import run_track
    sig = inspect.signature(run_track)
    for kw in ("min_spatial_diversity", "weight_strategy", "outlier_drop"):
        assert kw in sig.parameters, f"missing kwarg {kw}"
