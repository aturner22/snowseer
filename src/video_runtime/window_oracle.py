"""Pre-flight 'is this demo-able?' oracle — never burn cache compute on a
structurally broken window again.

The pipeline works only when (a) the snow frame has a *summer prior* in the
neighbourhood and (b) the summer prior has *visible road* that the
segmenter can find. If either fails, the matcher has nothing to anchor on
and the EMA just holds a stale mask. That's not the same as a good demo.

This module verifies both conditions before the user pays cache-build cost:

  1. Prior availability — KD-tree query of summer UTM poses; reject snow
     frames whose K-nearest summer pose is farther than `distance_thresh`
     metres.
  2. Summer-segmentation quality — Mask2Former on each candidate summer
     prior; reject priors whose road-mask coverage in the foreground (lower
     70 % of the image) is below `coverage_thresh`.

Output:
  - per-frame report: (snow_idx, n_priors_ok, distances, coverages)
  - longest contiguous run of OK frames
  - candidate windows (top N by score)

CLI:
  uv run python -m src.video_runtime.window_oracle --track <id> \
      [--stride 1] [--K 3] [--max-dim 1024] \
      [--distance-thresh 30] [--coverage-thresh 0.05] \
      [--out-json outputs/<track>_oracle.json]

The user reads the printed table, picks a window, then commits to a cache
build (`make reproduce-track TRACK=<id>` with the chosen --start/--end).

Compute cost: dominated by the segmentation pass, which is one-shot per
unique summer frame and cached. With stride 10 over a 9 000-frame snow
track + K=3 priors the typical hit is ~500 unique summer segmentations
(~5–10 min on Mac CPU).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.video_runtime.track import Track


@dataclass
class FrameOracle:
    """Per-snow-frame oracle result."""
    snow_idx: int
    snow_seq_idx: int
    n_priors_total: int
    n_priors_ok: int            # priors that passed BOTH distance + coverage
    distances: list[float]
    coverages: list[float]      # mean road coverage per prior (0..1)

    @property
    def is_ok(self) -> bool:
        return self.n_priors_ok >= 1

    @property
    def mean_coverage_ok(self) -> float:
        good = [c for c, d in zip(self.coverages, self.distances) if d <= 9999]
        return float(np.mean(good)) if good else 0.0


@dataclass
class CandidateWindow:
    """A contiguous run of demo-able snow frames."""
    start_idx: int           # local snow index
    end_idx: int             # inclusive
    n_frames: int
    score: float             # ≥ 1 priors_ok per frame; score = mean(n_priors_ok) × mean(coverage)

    def __str__(self) -> str:
        return (f"frames {self.start_idx:>4d}–{self.end_idx:<4d} "
                f"({self.n_frames:>4d} frames, ~{self.n_frames/10:5.1f} s @ 10 fps)  "
                f"score={self.score:.3f}")


def _foreground_coverage(mask: np.ndarray, foreground_y_frac: float = 0.30) -> float:
    """Fraction of road pixels in the lower (1 - foreground_y_frac) of the image.

    The Boreas roof-mounted camera puts road in the bottom ~70 %; the upper
    portion is sky / building tops which the segmenter can hallucinate as
    road on bad summer frames. Restrict the coverage measure to the
    informative region.
    """
    h = mask.shape[0]
    cut = int(round(foreground_y_frac * h))
    fg = mask[cut:]
    fg_pixels = fg.size
    road_pixels = int(np.count_nonzero(fg))
    return road_pixels / fg_pixels if fg_pixels else 0.0


def evaluate_track(
    track_id: str,
    *,
    K: int = 3,
    max_dim: int = 1024,
    stride: int = 1,
    distance_thresh: float = 30.0,
    coverage_thresh: float = 0.05,
    foreground_y_frac: float = 0.30,
) -> tuple[list[FrameOracle], Track]:
    """Run the oracle over all snow frames at the given stride.

    Returns the per-frame results in snow-window order (filtered by stride)
    plus the loaded Track for the caller's use.
    """
    from scipy.spatial import cKDTree
    from src.segmentation import RoadSegmenter
    from src.overlay import keep_largest_component

    track = Track(track_id)
    n_snow = len(track.snow_meta)
    n_summer = len(track.summer_meta)
    print(f"[oracle] {track_id}  snow={n_snow}  summer={n_summer}  "
          f"K={K}  stride={stride}  d_thresh={distance_thresh}m  "
          f"cov_thresh={coverage_thresh:.2f}", flush=True)

    summer_xy = np.array([[m.easting, m.northing] for m in track.summer_meta],
                         dtype=np.float64)
    tree = cKDTree(summer_xy)

    segmenter = RoadSegmenter()
    coverage_cache: dict[int, float] = {}

    def _summer_coverage(summer_local_idx: int) -> float:
        if summer_local_idx in coverage_cache:
            return coverage_cache[summer_local_idx]
        m = track.summer_meta[summer_local_idx]
        img = track.load_frame(m, max_dim=max_dim)
        mask = keep_largest_component(segmenter.segment_road(img))
        cov = _foreground_coverage(mask, foreground_y_frac=foreground_y_frac)
        coverage_cache[summer_local_idx] = cov
        return cov

    results: list[FrameOracle] = []
    snow_indices = range(0, n_snow, stride)
    t0 = time.time()
    for n, snow_idx in enumerate(snow_indices):
        sm = track.snow_meta[snow_idx]
        d, idx = tree.query([sm.easting, sm.northing], k=K)
        if np.isscalar(d):
            d = np.array([d])
            idx = np.array([idx])
        coverages = []
        n_ok = 0
        for di, ii in zip(d, idx):
            if di > distance_thresh:
                coverages.append(0.0)
                continue
            cov = _summer_coverage(int(ii))
            coverages.append(cov)
            if cov >= coverage_thresh:
                n_ok += 1
        results.append(FrameOracle(
            snow_idx=snow_idx,
            snow_seq_idx=sm.seq_idx,
            n_priors_total=K,
            n_priors_ok=n_ok,
            distances=[float(x) for x in d],
            coverages=[float(c) for c in coverages],
        ))
        if (n + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (n + 1) / elapsed if elapsed else 0.0
            n_remaining = len(snow_indices) - (n + 1)
            eta = n_remaining / rate if rate else 0.0
            print(f"[oracle]   evaluated {n+1}/{len(snow_indices)}  "
                  f"({len(coverage_cache)} unique summer segments cached)  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    print(f"[oracle] done in {time.time() - t0:.0f}s  "
          f"{len(coverage_cache)} unique summer segments cached", flush=True)
    return results, track


def find_candidate_windows(
    results: list[FrameOracle],
    *,
    min_window: int = 50,
    top_n: int = 5,
) -> list[CandidateWindow]:
    """Find contiguous runs of demo-able snow frames, ranked by score."""
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, r in enumerate(results):
        if r.is_ok and not in_run:
            start = i
            in_run = True
        elif not r.is_ok and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(results) - 1))

    out: list[CandidateWindow] = []
    for s, e in runs:
        n = e - s + 1
        if n < min_window:
            continue
        # snow-window indices (account for stride)
        snow_s = results[s].snow_idx
        snow_e = results[e].snow_idx
        n_frames_in_snow_space = snow_e - snow_s + 1
        priors_ok = [results[i].n_priors_ok for i in range(s, e + 1)]
        coverages = [results[i].mean_coverage_ok for i in range(s, e + 1)]
        score = float(np.mean(priors_ok) * np.mean(coverages))
        out.append(CandidateWindow(
            start_idx=snow_s,
            end_idx=snow_e,
            n_frames=n_frames_in_snow_space,
            score=score,
        ))
    out.sort(key=lambda w: -w.score)
    return out[:top_n]


def print_report(results: list[FrameOracle], windows: list[CandidateWindow], track_id: str) -> None:
    n_total = len(results)
    n_ok = sum(1 for r in results if r.is_ok)
    print(f"\n=== oracle report — {track_id} ===")
    print(f"  evaluated:        {n_total} snow frames")
    print(f"  demo-able:        {n_ok} ({100 * n_ok // max(n_total, 1)} %)")
    print(f"  candidate windows (top {len(windows)} by score):")
    if not windows:
        print(f"    (none — track has no contiguous demo-able run ≥ 50 frames)")
        return
    for i, w in enumerate(windows):
        marker = "★" if i == 0 else " "
        print(f"    {marker} {w}")


def evaluate_full_track_poses_only(
    track_id: str,
    *,
    distance_thresh: float = 30.0,
) -> list[dict]:
    """Lite oracle on the FULL parent camera_poses.csv (no segmentation).

    Bypasses the 350-frame snow window; reads `data/video/tracks/<id>/
    snow/camera_poses.csv` directly to find the best contiguous range
    where summer coverage exists. Useful for re-windowing decisions
    BEFORE fetching frames for the new window.

    Returns a list of per-row dicts: {idx, easting, northing, dist_m}.
    The full segmentation oracle (`evaluate_track`) should run on the
    chosen window once frames are downloaded.
    """
    from scipy.spatial import cKDTree
    from src.video_runtime.track import TRACKS_DIR

    snow_csv = TRACKS_DIR / track_id / "snow" / "camera_poses.csv"
    summer_csv = TRACKS_DIR / track_id / "summer" / "camera_poses.csv"
    if not snow_csv.exists() or not summer_csv.exists():
        raise FileNotFoundError(
            f"need both {snow_csv} and {summer_csv}. "
            f"Run `make video-fetch TRACK={track_id}` first to bootstrap."
        )

    snow = np.genfromtxt(snow_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    summer = np.genfromtxt(summer_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    summer_xy = np.column_stack([summer["easting"], summer["northing"]])
    tree = cKDTree(summer_xy)

    rows: list[dict] = []
    for i in range(len(snow)):
        e = float(snow["easting"][i])
        n = float(snow["northing"][i])
        d, _ = tree.query([e, n], k=1)
        rows.append({"idx": i, "easting": e, "northing": n, "dist_m": float(d)})
    return rows


def find_pose_only_windows(rows: list[dict], distance_thresh: float = 30.0,
                           min_window: int = 100) -> list[CandidateWindow]:
    """Longest contiguous runs of rows with dist_m ≤ threshold."""
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, r in enumerate(rows):
        ok = r["dist_m"] <= distance_thresh
        if ok and not in_run:
            start = i
            in_run = True
        elif not ok and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(rows) - 1))

    out: list[CandidateWindow] = []
    for s, e in runs:
        n = e - s + 1
        if n < min_window:
            continue
        dists = [rows[i]["dist_m"] for i in range(s, e + 1)]
        # Score: longer + lower mean distance is better
        score = float(n / (1.0 + np.mean(dists)))
        out.append(CandidateWindow(start_idx=s, end_idx=e, n_frames=n, score=score))
    out.sort(key=lambda w: -w.score)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--K", type=int, default=3, help="number of summer priors per snow frame")
    p.add_argument("--max-dim", type=int, default=1024)
    p.add_argument("--stride", type=int, default=1,
                   help="evaluate every Nth snow frame (≥10 recommended for long tracks)")
    p.add_argument("--distance-thresh", type=float, default=30.0,
                   help="reject priors whose UTM distance exceeds this (metres)")
    p.add_argument("--coverage-thresh", type=float, default=0.05,
                   help="reject priors whose foreground road coverage is below this fraction")
    p.add_argument("--foreground-y-frac", type=float, default=0.30,
                   help="cut off the top fraction of the image when measuring road coverage")
    p.add_argument("--min-window", type=int, default=50,
                   help="minimum contiguous OK frames to qualify as a candidate window")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--out-json", default=None,
                   help="write the full per-frame report to this JSON path")
    p.add_argument("--poses-only", action="store_true",
                   help="lite mode: evaluate full parent camera_poses.csv WITHOUT segmentation. "
                        "Use to find re-windowing candidates before fetching new frames.")
    args = p.parse_args()

    if args.poses_only:
        rows = evaluate_full_track_poses_only(
            args.track, distance_thresh=args.distance_thresh,
        )
        windows = find_pose_only_windows(
            rows, distance_thresh=args.distance_thresh,
            min_window=max(args.min_window, 100),
        )
        n_ok = sum(1 for r in rows if r["dist_m"] <= args.distance_thresh)
        print(f"\n=== pose-only oracle — {args.track} ===")
        print(f"  total rows in full snow camera_poses.csv: {len(rows)}")
        print(f"  rows within {args.distance_thresh}m of any summer pose: "
              f"{n_ok} ({100 * n_ok // max(len(rows), 1)}%)")
        print(f"  candidate windows (top {min(args.top_n, len(windows))} by score):")
        if not windows:
            print(f"    (none ≥ {max(args.min_window, 100)} contiguous frames)")
        for i, w in enumerate(windows[:args.top_n]):
            mark = "★" if i == 0 else " "
            print(f"    {mark} {w}")
        if args.out_json:
            out = Path(args.out_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "track": args.track,
                "mode": "poses_only",
                "params": {"distance_thresh": args.distance_thresh,
                           "min_window": args.min_window},
                "n_total_rows": len(rows),
                "n_within_threshold": n_ok,
                "candidate_windows": [
                    {"start_idx": w.start_idx, "end_idx": w.end_idx,
                     "n_frames": w.n_frames, "score": w.score}
                    for w in windows[:args.top_n]
                ],
            }, indent=2))
            print(f"  → wrote {out}")
        return

    results, track = evaluate_track(
        args.track,
        K=args.K, max_dim=args.max_dim, stride=args.stride,
        distance_thresh=args.distance_thresh,
        coverage_thresh=args.coverage_thresh,
        foreground_y_frac=args.foreground_y_frac,
    )
    windows = find_candidate_windows(
        results, min_window=args.min_window, top_n=args.top_n,
    )
    print_report(results, windows, args.track)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "track": args.track,
            "params": {
                "K": args.K, "max_dim": args.max_dim, "stride": args.stride,
                "distance_thresh": args.distance_thresh,
                "coverage_thresh": args.coverage_thresh,
                "foreground_y_frac": args.foreground_y_frac,
            },
            "n_snow_frames": len(track.snow_meta),
            "n_evaluated": len(results),
            "n_ok": sum(1 for r in results if r.is_ok),
            "candidate_windows": [
                {"start_idx": w.start_idx, "end_idx": w.end_idx,
                 "n_frames": w.n_frames, "score": w.score}
                for w in windows
            ],
            "per_frame": [
                {"snow_idx": r.snow_idx, "snow_seq_idx": r.snow_seq_idx,
                 "n_priors_ok": r.n_priors_ok,
                 "distances": r.distances, "coverages": r.coverages}
                for r in results
            ],
        }, indent=2))
        print(f"  → wrote {out}")


if __name__ == "__main__":
    main()
