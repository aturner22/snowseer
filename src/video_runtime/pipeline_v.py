"""Per-frame video pipeline — runs the static cross-season pipeline once
per snow frame against K nearest summer priors, fuses the K warped masks,
and emits a list of FrameResult that the renderer consumes.

K.2 baseline: no temporal smoothing.
K.4 (this update): optional Smoother applied after fusion.

Matching is the dominant cost, so the pipeline can persist the *raw fused
mask per frame* (pre-smoother) into a cache file. Subsequent renders that
only change the Smoother (EMA / flow / none) load from cache and skip
matching — that's how we run the K.4 ablation cheaply.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.fuse import weighted_soft_average, crop_foreground
from src.homography import estimate
from src.overlay import keep_largest_component, warp_mask
from src.video_runtime.prior_pool import PriorPool, PriorEntry
from src.video_runtime.temporal import Smoother
from src.video_runtime.track import Track, FrameMeta


@dataclass
class FrameResult:
    snow_meta: FrameMeta
    snow_image: np.ndarray
    fused_mask: np.ndarray | None       # in snow-image space, uint8 0/1
    per_prior: list[dict] = field(default_factory=list)   # diagnostic per-prior info
    n_priors_used: int = 0
    elapsed_s: float = 0.0


def _process_one_prior(
    snow_image: np.ndarray, prior: PriorEntry, matcher
) -> tuple[np.ndarray | None, int, np.ndarray | None]:
    """Match snow → this prior, warp the prior's road mask back to snow space.

    Returns (warped_mask_in_snow_space, n_inliers, valid_region_in_snow_space)
    or (None, 0, None) on match failure.
    """
    result = matcher.match(snow_image, prior.image)
    homo = estimate(result, snow_image.shape[:2], prior.image.shape[:2])
    if homo.H is None or homo.inlier_mask is None:
        return None, 0, None
    n_inliers = int(np.sum(homo.inlier_mask))
    if n_inliers < 8:
        return None, n_inliers, None
    H_inv = np.linalg.inv(homo.H)
    mask_in_snow = warp_mask(prior.road_mask, H_inv, snow_image.shape[:2])
    mask_in_snow = keep_largest_component(mask_in_snow)
    # Track the warped extent of the prior frame so fusion can edge-erode and
    # weight only where each prior actually covered.
    valid = warp_mask(np.ones_like(prior.road_mask, dtype=np.uint8), H_inv,
                      snow_image.shape[:2])
    return mask_in_snow, n_inliers, valid


def run_track(
    track_id: str,
    *,
    K: int = 3,
    max_dim: int = 1024,
    start: int = 0,
    end: int | None = None,
    stride: int = 1,
    foreground_y_frac: float = 0.45,
    smoother: "Smoother | None" = None,
    cache_path: Path | None = None,
    rebuild_cache: bool = False,
) -> list[FrameResult]:
    """Run the per-frame pipeline over the snow stream of `track_id`.

    Returns one FrameResult per processed snow frame.
    """
    # Cache fast-path: if a cache file exists for the same (start, end,
    # stride, K, max_dim, foreground_y_frac) it contains the per-frame raw
    # fused mask plus snow_image; we just re-apply the smoother and skip
    # matching entirely.
    if cache_path is not None and cache_path.exists() and not rebuild_cache:
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        cached_results: list[FrameResult] = cached["results"]
        # Re-apply smoother on the saved raw fusion outputs.
        if smoother is not None:
            smoother.reset()
            for r in cached_results:
                r.fused_mask = smoother.smooth(r.fused_mask, r.snow_image)
        print(f"[{track_id}] loaded {len(cached_results)} frames from cache "
              f"({cache_path.name}); smoother='{type(smoother).__name__ if smoother else 'none'}'")
        return cached_results

    track = Track(track_id)
    pool = PriorPool(track, K=K, max_dim=max_dim)
    results: list[FrameResult] = []

    end = end or track.snow_frame_count()
    indices = list(range(start, end, stride))
    print(f"[{track_id}] processing {len(indices)} snow frames "
          f"(K={K} priors each, max_dim={max_dim})")

    matcher = pool.matcher()  # warm load

    for frame_n, snow_idx in enumerate(indices):
        t0 = time.time()
        snow_meta = track.snow_meta[snow_idx]
        snow_image = track.load_frame(snow_meta, max_dim=max_dim)

        priors = pool.select(snow_meta)
        per_prior_records: list[dict] = []
        masks: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        weights: list[float] = []

        for k_idx, prior in enumerate(priors):
            mask, n_inliers, valid = _process_one_prior(snow_image, prior, matcher)
            per_prior_records.append({
                "prior_idx": int(prior.meta.idx),
                "distance_m": float(prior.distance_m),
                "n_inliers": int(n_inliers),
                "ok": mask is not None,
            })
            if mask is not None:
                masks.append(mask)
                valids.append(valid)
                weights.append(float(n_inliers))

        fused: np.ndarray | None = None
        if masks:
            fused_soft = weighted_soft_average(masks, np.array(weights, dtype=np.float32),
                                               valids, threshold=0.4)
            fused_soft = keep_largest_component(fused_soft)
            fused_soft = crop_foreground(fused_soft, foreground_y_frac=foreground_y_frac)
            fused = fused_soft

        # Note: we save the *raw* fused mask to the FrameResult (not the
        # smoothed one) so the cache holds matching+fusion outputs and
        # different smoothers can be applied later from cache.
        elapsed = time.time() - t0
        results.append(FrameResult(
            snow_meta=snow_meta,
            snow_image=snow_image,
            fused_mask=fused,
            per_prior=per_prior_records,
            n_priors_used=len(masks),
            elapsed_s=elapsed,
        ))
        ok_inliers = [r["n_inliers"] for r in per_prior_records if r["ok"]]
        ok_summary = ",".join(map(str, ok_inliers)) if ok_inliers else "—"
        print(f"  [{frame_n + 1:>3d}/{len(indices)}] snow_idx={snow_idx}  "
              f"priors_ok={len(masks)}/{K}  inliers={ok_summary}  "
              f"t={elapsed:.1f}s")

    # Persist raw results to cache before smoothing, so the K.4 ablation
    # can re-render with different smoothers without re-matching.
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump({"results": results}, fh)
        print(f"[{track_id}] cached {len(results)} raw FrameResults to {cache_path}")

    # Temporal smoothing — applied after caching so the cache holds raw
    # fusion outputs.
    if smoother is not None:
        smoother.reset()
        for r in results:
            r.fused_mask = smoother.smooth(r.fused_mask, r.snow_image)

    return results
