"""Augmentation pass for quad-mode rendering.

Given an existing FrameResult cache (`_cache_<tag>.pkl`), compute per-frame:
  1. The naive direct-on-snow Cityscapes road segmentation (the failure
     condition that motivates the cross-season pipeline).
  2. The closest summer prior image + cached road mask, ready for the
     "clear prior" panel.

Persist these to `_aug_<tag>.pkl` so multiple renders can reuse them.

Usage:
    uv run python -m src.video_runtime.augment --track <id> --cache-tag <tag>
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from src.overlay import keep_largest_component
from src.video_runtime.pipeline_v import FrameResult
from src.video_runtime.prior_pool import PriorPool
from src.video_runtime.track import Track

ROOT = Path(__file__).resolve().parents[2]


def _augment_one(snow_image: np.ndarray, segmenter) -> np.ndarray:
    naive = segmenter.segment_road(snow_image)
    return keep_largest_component(naive)


def augment(track_id: str, cache_tag: str, *, K: int = 3, max_dim: int = 1024) -> Path:
    cache_path = ROOT / f"outputs/video/{track_id}/_cache_{cache_tag}.pkl"
    aug_path = ROOT / f"outputs/video/{track_id}/_aug_{cache_tag}.pkl"
    if not cache_path.exists():
        raise SystemExit(f"missing cache {cache_path}; run the matching pass first")

    with open(cache_path, "rb") as fh:
        cached = pickle.load(fh)
    results: list[FrameResult] = cached["results"]
    print(f"[{track_id}] augmenting {len(results)} frames "
          f"(cache: {cache_path.name})")

    track = Track(track_id)
    pool = PriorPool(track, K=K, max_dim=max_dim)

    # Lazily instantiate segmenter once (~3s/frame on CPU).
    from src.segmentation import RoadSegmenter
    segmenter = RoadSegmenter()

    summer_panels: list[dict | None] = []
    naive_masks: list[np.ndarray | None] = []

    for i, r in enumerate(results):
        t0 = time.time()
        # Naive direct-on-snow segmentation.
        naive = _augment_one(r.snow_image, segmenter)
        naive_masks.append(naive if int(naive.sum()) > 0 else None)

        # Closest summer prior — pool.select returns up to K, take priors[0].
        priors = pool.select(r.snow_meta)
        if priors:
            p0 = priors[0]
            summer_panels.append({
                "image": p0.image,
                "road_mask": p0.road_mask,
                "distance_m": p0.distance_m,
            })
        else:
            summer_panels.append(None)

        elapsed = time.time() - t0
        print(f"  [{i + 1:>3d}/{len(results)}] frame {r.snow_meta.idx}  "
              f"naive={int(naive.sum() > 0)}  t={elapsed:.1f}s")

    aug_path.parent.mkdir(parents=True, exist_ok=True)
    with open(aug_path, "wb") as fh:
        pickle.dump({"summer_panels": summer_panels, "naive_masks": naive_masks}, fh)
    print(f"[{track_id}] wrote augmentation cache → {aug_path}")
    return aug_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--cache-tag", required=True)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--max-dim", type=int, default=1024)
    args = p.parse_args()

    augment(args.track, args.cache_tag, K=args.K, max_dim=args.max_dim)


if __name__ == "__main__":
    main()
