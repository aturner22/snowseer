"""Prior pool — selects K nearest summer priors for a snow frame, plus an
optional sliding window of past-frame synthetic priors (K.3 — past snow
frames acting as same-domain priors for the current frame).

Caches the expensive summer-side artefacts (loaded image + Mask2Former road
mask) so they're computed once per summer frame, not once per (snow, summer)
pair.

Keypoint caching is deferred — the existing Matcher.match() always re-extracts
DISK keypoints on both sides. We pay that cost; profiling can add a cached
path later.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.video_runtime.track import FrameMeta, Track


@dataclass
class PriorEntry:
    """One prior picked for a snow frame.

    Same shape for a summer prior and a synthetic (past-snow) prior. The
    `kind` field lets the renderer / fusion treat them differently if needed.
    """
    meta: "FrameMeta | None"    # None for synthetic priors (no static FrameMeta)
    distance_m: float           # UTM Euclidean distance from snow pose
    image: np.ndarray           # full RGB at processing resolution
    road_mask: np.ndarray       # full-resolution road mask (uint8, 0/255)
    kind: str = "summer"        # 'summer' or 'synthetic'


class PriorPool:
    """Cache + K-NN selector for summer priors.

    Initialised once per Track. On each call to `select(snow_meta)`, it picks
    K summer priors closest in (easting, northing) and returns them with
    cached image + segmentation already attached.
    """

    def __init__(self, track: "Track", *, K: int = 3, max_dim: int = 1024):
        from scipy.spatial import cKDTree

        self.track = track
        self.K = K
        self.max_dim = max_dim
        self._summer_xy = np.array(
            [[m.easting, m.northing] for m in track.summer_meta],
            dtype=np.float64,
        )
        self._tree = cKDTree(self._summer_xy)
        # Caches keyed by summer FrameMeta.idx (local index in the window).
        self._image_cache: dict[int, np.ndarray] = {}
        self._mask_cache: dict[int, np.ndarray] = {}
        # Lazy models — shared across snow frames.
        self._matcher = None
        self._segmenter = None

    def matcher(self):
        if self._matcher is None:
            from src.matching import Matcher
            self._matcher = Matcher()
        return self._matcher

    def segmenter(self):
        if self._segmenter is None:
            from src.segmentation import RoadSegmenter
            self._segmenter = RoadSegmenter()
        return self._segmenter

    def _summer_image(self, m: "FrameMeta") -> np.ndarray:
        if m.idx not in self._image_cache:
            self._image_cache[m.idx] = self.track.load_frame(m, max_dim=self.max_dim)
        return self._image_cache[m.idx]

    def _summer_mask(self, m: "FrameMeta", img: np.ndarray) -> np.ndarray:
        if m.idx not in self._mask_cache:
            from src.overlay import keep_largest_component
            mask = self.segmenter().segment_road(img)
            mask = keep_largest_component(mask)
            self._mask_cache[m.idx] = mask
        return self._mask_cache[m.idx]

    def select(self, snow_meta: "FrameMeta") -> list[PriorEntry]:
        """Return up to K summer priors closest in UTM distance to the snow pose."""
        d, idx = self._tree.query([snow_meta.easting, snow_meta.northing], k=self.K)
        if np.isscalar(d):
            d = np.array([d])
            idx = np.array([idx])
        out: list[PriorEntry] = []
        for di, ii in zip(d, idx):
            m = self.track.summer_meta[int(ii)]
            img = self._summer_image(m)
            mask = self._summer_mask(m, img)
            out.append(PriorEntry(
                meta=m, distance_m=float(di),
                image=img, road_mask=mask, kind="summer",
            ))
        return out


@dataclass
class SyntheticPriorQueue:
    """Sliding window of past (snow_image, fused_mask) tuples used as
    same-domain priors for the current frame (K.3).

    The matcher loves these — snow→snow gives many more inliers than
    snow→summer, because lighting / texture / lens conditions are identical.
    We trade a sliver of memory + an extra match call per frame for big
    gains in temporal coherence and overall match confidence.

    Caveat: a bad fused mask in frame t-1 propagates to frame t. EMA on the
    smoothed display mask helps; the synthetic queue here stores the *raw*
    pre-EMA fused mask so the smoothing remains a display concern, not a
    feedback loop on the matcher.

    `max_size`: how many past frames to retain (3 is a good default —
    one back, one further, one furthest, captures both immediate and
    medium-distance evidence without quadratic cost growth).
    """
    max_size: int = 3
    _q: "deque[tuple[np.ndarray, np.ndarray]]" = None  # type: ignore

    def __post_init__(self):
        self._q = deque(maxlen=self.max_size)

    def reset(self) -> None:
        self._q.clear()

    def push(self, snow_image: np.ndarray, fused_mask: np.ndarray | None) -> None:
        """After processing a frame, register its (image, mask) for future frames.
        Skip frames whose fused mask is empty or None."""
        if fused_mask is None or int(fused_mask.sum()) == 0:
            return
        self._q.append((snow_image.copy(), fused_mask.copy()))

    def entries(self) -> list[PriorEntry]:
        """Return all current entries as PriorEntry list (in chronological
        order: oldest first). Empty if no past frames yet."""
        out: list[PriorEntry] = []
        # Iterate newest → oldest so when fusion uses inlier-count weighting,
        # the most recent (typically best-matched) gets reported first in
        # diagnostics. Fusion treats them all the same regardless of order.
        for img, mask in reversed(self._q):
            out.append(PriorEntry(
                meta=None, distance_m=0.0,
                image=img, road_mask=mask, kind="synthetic",
            ))
        return out
