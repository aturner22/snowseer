"""Per-frame video pipeline.

Runs the static cross-season pipeline once per snow frame against K
nearest summer priors, fuses the K warped masks, and emits a list of
FrameResult that the renderer consumes.

Matching is the dominant cost. Two reliability features keep this
practical:

1. Cache: once matching completes, persist the raw fused masks to
   `_cache_<tag>.pkl`. Subsequent renders that change only the smoother
   (`temporal=ema|none`) load the cache and skip matching.
2. Checkpoint resume: every CHECKPOINT_EVERY frames, atomically write
   `_cache_<tag>.partial.pkl`. If the process is killed mid-run, restart
   reads the partial and resumes from the next unprocessed frame.

All progress lines flush at write time. ETA is logged every
ETA_EVERY frames. Each process holds ~9 GB resident, so on a 16 GB
machine the pipeline must run sequentially, not in parallel.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from typing import Protocol

from src.homography import estimate
from src.overlay import crop_foreground, keep_largest_component, warp_mask, weighted_soft_average
from src.video_runtime.prior_pool import PriorPool, PriorEntry
from src.video_runtime.track import Track, FrameMeta

CHECKPOINT_EVERY = 50           # write partial cache every N processed frames
ETA_EVERY = 10                   # log ETA line every N processed frames


def _log(msg: str) -> None:
    """Single-line log with explicit flush so output appears in real time
    even when stdout is redirected to a file (Python defaults to block
    buffering on non-tty stdout)."""
    print(msg, flush=True)


class Smoother(Protocol):
    """A temporal smoother takes the raw per-frame mask plus the snow
    image and returns a smoothed binary mask in snow-image space."""

    def smooth(self, raw_mask: np.ndarray | None, snow_image: np.ndarray) -> np.ndarray | None:
        ...

    def reset(self) -> None:
        ...


@dataclass
class EMASmoother:
    """Exponential moving average on the binary mask, treated as float [0, 1].

        smooth_t = alpha * raw_t + (1 - alpha) * smooth_{t-1}

    `alpha=0.5` weights past and present equally; `alpha=0.7` favours
    present (less sticky); `alpha=0.3` favours past (very sticky, can
    lag the camera). Default 0.5.

    On a missing raw_mask (matcher failure for one frame), return the
    last smoothed mask unchanged so the road doesn't disappear when one
    frame fails.

    Threshold at 0.5 to get the binary output.
    """
    alpha: float = 0.5
    _state: np.ndarray | None = field(default=None, init=False)

    def reset(self) -> None:
        self._state = None

    def smooth(self, raw_mask: np.ndarray | None, snow_image: np.ndarray) -> np.ndarray | None:
        if raw_mask is None:
            if self._state is None:
                return None
            return (self._state >= 0.5).astype(np.uint8)
        raw_f = (raw_mask > 0).astype(np.float32)
        if self._state is None or self._state.shape != raw_f.shape:
            self._state = raw_f.copy()
        else:
            self._state = self.alpha * raw_f + (1.0 - self.alpha) * self._state
        return (self._state >= 0.5).astype(np.uint8)


def make_smoother(name: str | None, **kwargs) -> Smoother | None:
    """Factory for the CLI: `--temporal {none,ema}`."""
    if name in (None, "none", "off"):
        return None
    if name == "ema":
        return EMASmoother(alpha=kwargs.get("alpha", 0.5))
    raise ValueError(f"Unknown temporal smoother: {name!r}. Try 'none' or 'ema'.")


@dataclass
class FrameResult:
    snow_meta: FrameMeta
    snow_image: np.ndarray
    fused_mask: np.ndarray | None       # in snow-image space, uint8 0/1
    per_prior: list[dict] = field(default_factory=list)   # diagnostic per-prior info
    n_priors_used: int = 0


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


def _atomic_pickle(path: Path, payload: dict) -> None:
    """Pickle to a sibling .tmp then rename so a crash mid-write doesn't
    leave a corrupt cache. The user can `kill -9` mid-checkpoint safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh)
    os.replace(tmp, path)


def _format_eta(frame_n: int, total: int, t0: float) -> str:
    """Human-readable elapsed/ETA suffix."""
    elapsed = time.time() - t0
    if frame_n == 0:
        return f"elapsed={elapsed:.0f}s ETA=?"
    rate = frame_n / max(elapsed, 1e-3)
    remaining = (total - frame_n) / max(rate, 1e-9)
    return f"elapsed={elapsed:.0f}s ETA={remaining:.0f}s ({rate:.2f} fr/s)"


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
    seg_prob_threshold: float | None = None,
    seg_morph_radius: int = 0,
) -> list[FrameResult]:
    """Run the per-frame pipeline over the snow stream of `track_id`.

    Cache fast-path: if `cache_path` exists and `rebuild_cache=False`, the
    matching pass is skipped entirely; only the smoother is re-applied.

    Checkpoint resume: if `<cache_path>.partial.pkl` exists, the matching
    pass starts from the next unprocessed frame instead of from scratch.
    Useful after a kill or memory-thrash crash.
    """
    # Cache fast-path — completed runs.
    if cache_path is not None and cache_path.exists() and not rebuild_cache:
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        cached_results: list[FrameResult] = cached["results"]
        # Honour --start / --end / --stride at the snow-stream level (not the
        # cache-position level). Filter cached results whose snow_meta.idx
        # falls in [start, end). This makes a render with the same window as
        # the cache build a no-op (returns the full cache), and a render
        # with a sub-window correctly subsets by snow stream index.
        n_cached = len(cached_results)
        if start > 0 or end is not None or stride > 1:
            end_eff = end if end is not None else float("inf")
            sub: list[FrameResult] = []
            kept = 0
            for r in cached_results:
                if start <= r.snow_meta.idx < end_eff:
                    if (kept % stride) == 0:
                        sub.append(r)
                    kept += 1
            cached_results = sub
            _log(f"[{track_id}] filtered cache to snow_idx [{start}:{end}:{stride}] "
                 f"→ {len(cached_results)} of {n_cached} frames")
        if smoother is not None:
            smoother.reset()
            for r in cached_results:
                r.fused_mask = smoother.smooth(r.fused_mask, r.snow_image)
        _log(f"[{track_id}] loaded {len(cached_results)} frames from cache "
             f"({cache_path.name}); smoother='{type(smoother).__name__ if smoother else 'none'}'")
        return cached_results

    track = Track(track_id)

    # Pre-flight pose-only sanity guard: cheap KD-tree query to confirm
    # the snow track has summer poses in the neighbourhood. Catches
    # structurally broken windows before paying matcher compute.
    try:
        from scipy.spatial import cKDTree
        summer_xy = np.array([[m.easting, m.northing] for m in track.summer_meta],
                             dtype=np.float64)
        snow_xy = np.array([[m.easting, m.northing] for m in track.snow_meta],
                           dtype=np.float64)
        if len(summer_xy) > 0 and len(snow_xy) > 0:
            tree = cKDTree(summer_xy)
            d, _ = tree.query(snow_xy, k=1)
            n_within = int(np.sum(d <= 30.0))
            n_total = len(snow_xy)
            if n_within < 0.5 * n_total:
                _log(f"[{track_id}] PRE-FLIGHT WARN: only {n_within}/{n_total} "
                     f"({100 * n_within // max(n_total, 1)}%) snow poses within "
                     f"30m of any summer pose; the matcher will probably fail "
                     f"on most frames. Continuing anyway.")
    except Exception as _exc:
        _log(f"[{track_id}] pre-flight pose check skipped ({_exc}); proceeding")

    pool = PriorPool(
        track, K=K, max_dim=max_dim,
        seg_prob_threshold=seg_prob_threshold,
        seg_morph_radius=seg_morph_radius,
    )

    n_avail = track.snow_frame_count()
    end = min(end, n_avail) if end is not None else n_avail
    start = min(max(start, 0), n_avail)
    indices = list(range(start, end, stride))

    # Resume from partial checkpoint if one exists for this cache_path.
    results: list[FrameResult] = []
    resume_from = 0
    partial_path = (
        cache_path.with_name(cache_path.stem + ".partial.pkl")
        if cache_path is not None else None
    )
    if partial_path is not None and partial_path.exists() and not rebuild_cache:
        try:
            with open(partial_path, "rb") as fh:
                partial = pickle.load(fh)
            cached_indices = partial.get("indices", [])
            if cached_indices == indices[: len(partial["results"])]:
                results = partial["results"]
                resume_from = len(results)
                _log(f"[{track_id}] resuming from partial cache ({partial_path.name}): "
                     f"{resume_from}/{len(indices)} frames already done")
            else:
                _log(f"[{track_id}] partial cache present but indices mismatch — starting over")
        except Exception as e:
            _log(f"[{track_id}] partial cache unreadable ({e}); starting over")

    _log(f"[{track_id}] processing {len(indices)} snow frames from index {resume_from} "
         f"(K={K} priors each, max_dim={max_dim})")

    matcher = pool.matcher()  # warm load

    t0 = time.time()
    for frame_n in range(resume_from, len(indices)):
        snow_idx = indices[frame_n]
        frame_t0 = time.time()
        snow_meta = track.snow_meta[snow_idx]
        snow_image = track.load_frame(snow_meta, max_dim=max_dim)

        priors = pool.select(snow_meta)
        per_prior_records: list[dict] = []
        masks: list[np.ndarray] = []
        valids: list[np.ndarray] = []
        weights: list[float] = []

        for prior in priors:
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
            fused_soft = weighted_soft_average(
                masks, np.array(weights, dtype=np.float32), valids, threshold=0.4)
            fused_soft = keep_largest_component(fused_soft)
            fused_soft = crop_foreground(fused_soft, foreground_y_frac=foreground_y_frac)
            fused = fused_soft

        elapsed = time.time() - frame_t0
        results.append(FrameResult(
            snow_meta=snow_meta, snow_image=snow_image, fused_mask=fused,
            per_prior=per_prior_records, n_priors_used=len(masks),
        ))

        ok_inliers = [r["n_inliers"] for r in per_prior_records if r["ok"]]
        ok_summary = ",".join(map(str, ok_inliers)) if ok_inliers else "—"
        _log(f"  [{frame_n + 1:>3d}/{len(indices)}] snow_idx={snow_idx}  "
             f"priors_ok={len(masks)}/{len(priors)}  inliers={ok_summary}  "
             f"t={elapsed:.1f}s")

        # Periodic ETA + checkpoint.
        progressed = frame_n - resume_from + 1
        if progressed % ETA_EVERY == 0:
            eta = _format_eta(progressed, len(indices) - resume_from, t0)
            _log(f"  ── {eta}")
        if partial_path is not None and progressed % CHECKPOINT_EVERY == 0 and progressed > 0:
            _atomic_pickle(partial_path, {"indices": indices[: len(results)],
                                          "results": results})
            _log(f"  ── checkpoint: wrote {len(results)} frames to {partial_path.name}")

    # Final cache write.
    if cache_path is not None:
        _atomic_pickle(cache_path, {"results": results})
        _log(f"[{track_id}] cached {len(results)} raw FrameResults to {cache_path}")
        # Drop the partial — main cache supersedes.
        if partial_path is not None and partial_path.exists():
            partial_path.unlink()

    # Temporal smoothing applied AFTER caching so cache holds raw fusion.
    if smoother is not None:
        smoother.reset()
        for r in results:
            r.fused_mask = smoother.smooth(r.fused_mask, r.snow_image)

    return results
