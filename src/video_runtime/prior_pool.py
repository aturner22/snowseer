"""Prior pool — selects K nearest summer priors for a snow frame and caches
the expensive summer-side artefacts (loaded image + Mask2Former road mask)
so they're computed once per summer frame, not once per (snow, summer) pair.

Keypoint caching is deferred — the existing Matcher.match() always re-extracts
DISK keypoints on both sides. K.2 baseline pays that cost; we'll add a cached
path in K.3+ if profiling demands it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.video_runtime.track import FrameMeta, Track


@dataclass
class PriorEntry:
    """One summer prior picked for a snow frame."""
    meta: "FrameMeta"
    distance_m: float           # UTM Euclidean distance from snow pose
    image: np.ndarray           # full RGB at processing resolution
    road_mask: np.ndarray       # full-resolution road mask (uint8, 0/255)


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
                image=img, road_mask=mask,
            ))
        return out
