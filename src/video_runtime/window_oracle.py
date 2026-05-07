"""Curation tool — *not* part of the runtime path.

Reads an existing matching cache and recommends the cleanest sub-window
of demo-able frames. The shipped reproduction does not run this; clean
clips have hardcoded windows in the Make targets / `track.json`.
This module is what we run *during curation* to choose those windows.

Two cheap modes:

  - **default** (cache-backed) — sub-second; reads
    `outputs/video/<track>/_cache_<tag>.pkl` and derives per-frame
    quality from `FrameResult.n_priors_used`. Applies a hole-tolerant
    morphological closing on the OK signal so a single bad frame in
    the middle of a long run doesn't split the candidate window.
    Appends a markdown row to `docs/audit_log.md` recording the chosen
    window for posterity.

  - **--poses-only** — sub-second; reads the FULL parent
    `camera_poses.csv` (no cache needed) and reports rows whose nearest
    summer pose is within `--distance-thresh` metres. Used during
    re-windowing decisions before any cache build.

The pre-flight pose-only sanity guard at the head of a cache build
lives in `pipeline_v.run_track`, not here. This module is purely
retrospective.

CLI:
  uv run python -m src.video_runtime.window_oracle --track <id>
  uv run python -m src.video_runtime.window_oracle --track <id> --poses-only

The legacy Mask2Former-based oracle (`evaluate_track`) is at
`_archive/src/video_runtime/window_oracle_legacy_pre_rewrite.py` and is
not imported anywhere; it's kept on disk for reference.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FrameOracle:
    """Per-snow-frame quality summary derived from cache data."""
    snow_idx: int           # local index in the cache (sequential)
    snow_seq_idx: int       # absolute index in the original snow stream
    n_priors_used: int      # how many of K priors produced a usable mask

    @property
    def is_ok(self) -> bool:
        return self.n_priors_used >= 1


@dataclass
class CandidateWindow:
    """A contiguous (after hole-tolerant smoothing) run of demo-able frames."""
    start_idx: int          # local snow index (cache position)
    end_idx: int            # inclusive
    n_frames: int           # frames in [start_idx, end_idx]
    n_ok: int               # frames in window where n_priors_used >= 1
    pct_ok: float           # n_ok / n_frames
    score: float            # length × OK% × mean(n_priors_used) — bigger is better
    snow_seq_start: int     # absolute snow stream index of start
    snow_seq_end: int       # absolute snow stream index of end (inclusive)

    def __str__(self) -> str:
        return (f"frames {self.start_idx:>4d}–{self.end_idx:<4d}  "
                f"({self.n_frames:>4d} frames, ~{self.n_frames/10:5.1f} s @ 10 fps)  "
                f"OK={100*self.pct_ok:5.1f}%  score={self.score:.3f}")


def _hole_tolerant_runs(ok_signal: list[bool], hole_tolerance: int) -> list[tuple[int, int]]:
    """Find runs of True in ok_signal, filling any False stretch shorter than
    or equal to hole_tolerance. Returns inclusive (start, end) index pairs.

    Equivalent to a binary morphological closing with a flat structuring
    element of width 2*hole_tolerance + 1, then run-length encoding the
    surviving True regions.
    """
    n = len(ok_signal)
    if n == 0:
        return []
    smoothed = list(ok_signal)
    if hole_tolerance > 0:
        # Find runs of False; fill them if length <= hole_tolerance.
        i = 0
        while i < n:
            if not smoothed[i]:
                j = i
                while j < n and not smoothed[j]:
                    j += 1
                gap_len = j - i
                # Only fill internal gaps (bordered by True on both sides);
                # leading and trailing False stretches stay False — the
                # window selection should not blindly extend past the run.
                bordered = (i > 0 and smoothed[i - 1]) and (j < n and j < n and smoothed[j] if j < n else False)
                # Re-evaluate: smoothed is being mutated; check original ok_signal.
                left_ok = i > 0 and smoothed[i - 1]
                right_ok = j < n and smoothed[j]
                if left_ok and right_ok and gap_len <= hole_tolerance:
                    for k in range(i, j):
                        smoothed[k] = True
                i = j
            else:
                i += 1
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, ok in enumerate(smoothed):
        if ok and not in_run:
            start = i
            in_run = True
        elif not ok and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, n - 1))
    return runs


def find_candidate_windows(
    frame_oracles: list[FrameOracle],
    *,
    min_window: int = 50,
    top_n: int = 5,
    hole_tolerance: int = 5,
) -> list[CandidateWindow]:
    """Find the top candidate windows after hole-tolerant smoothing.

    A window of length `min_window` need not be strictly contiguous; runs
    of bad frames shorter than or equal to `hole_tolerance` are filled.
    This matches how the renderer experiences the track: EMA holds
    previous good masks across short failure stretches, so isolated
    bad frames do not visibly degrade the clip.
    """
    ok_signal = [r.is_ok for r in frame_oracles]
    runs = _hole_tolerant_runs(ok_signal, hole_tolerance)

    out: list[CandidateWindow] = []
    for s, e in runs:
        n = e - s + 1
        if n < min_window:
            continue
        window_oracles = frame_oracles[s:e + 1]
        n_ok = sum(1 for r in window_oracles if r.is_ok)
        pct = n_ok / n
        mean_priors = float(np.mean([r.n_priors_used for r in window_oracles]))
        score = n * pct * (1.0 + mean_priors)
        out.append(CandidateWindow(
            start_idx=s,
            end_idx=e,
            n_frames=n,
            n_ok=n_ok,
            pct_ok=pct,
            score=score,
            snow_seq_start=window_oracles[0].snow_seq_idx,
            snow_seq_end=window_oracles[-1].snow_seq_idx,
        ))
    out.sort(key=lambda w: -w.score)
    return out[:top_n]


def curate_from_cache(track_id: str, cache_tag: str = "canonical", *,
                      hole_tolerance: int = 5,
                      min_window: int = 50,
                      top_n: int = 5) -> tuple[list[FrameOracle], list[CandidateWindow]]:
    """Read the matching cache for a track + return per-frame quality
    + ranked candidate windows. Sub-second; no model loading."""
    cache_path = ROOT / f"outputs/video/{track_id}/_cache_{cache_tag}.pkl"
    if not cache_path.exists():
        raise SystemExit(
            f"No cache at {cache_path}. Run `make track TRACK={track_id}` "
            f"first; this module curates an existing cache, it does not "
            f"compute one."
        )
    with open(cache_path, "rb") as fh:
        payload = pickle.load(fh)
    results = payload["results"]

    frame_oracles = [
        FrameOracle(
            snow_idx=i,
            snow_seq_idx=int(r.snow_meta.idx),
            n_priors_used=int(r.n_priors_used),
        )
        for i, r in enumerate(results)
    ]
    windows = find_candidate_windows(
        frame_oracles,
        hole_tolerance=hole_tolerance,
        min_window=min_window,
        top_n=top_n,
    )
    return frame_oracles, windows


def print_report(frame_oracles: list[FrameOracle],
                 windows: list[CandidateWindow],
                 track_id: str,
                 hole_tolerance: int) -> None:
    n = len(frame_oracles)
    n_ok = sum(1 for r in frame_oracles if r.is_ok)
    print(f"\n=== curation report — {track_id} ===")
    print(f"  frames evaluated:   {n}")
    print(f"  frames with priors: {n_ok} ({100 * n_ok // max(n, 1)} %)")
    print(f"  hole tolerance:     {hole_tolerance} frames "
          f"(~{hole_tolerance/10:.1f} s @ 10 fps)")
    print(f"  candidate windows (top {len(windows)} by score, after smoothing):")
    if not windows:
        print(f"    (none meeting min-window threshold; cache may be too sparse)")
        return
    for i, w in enumerate(windows):
        marker = "★" if i == 0 else " "
        print(f"    {marker} {w}")
    top = windows[0]
    print()
    print(f"  recommended `make track` window:  "
          f"TRACK_START={top.snow_seq_start} TRACK_END={top.snow_seq_end + 1}")


# ── Pose-only mode (re-windowing aid) ───────────────────────────────────

def evaluate_full_track_poses_only(
    track_id: str,
    *,
    distance_thresh: float = 30.0,
) -> list[dict]:
    """Read the full parent camera_poses.csv (no cache, no segmentation)
    and report the nearest-summer-pose distance per snow frame."""
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
        nrt = float(snow["northing"][i])
        d, _ = tree.query([e, nrt], k=1)
        rows.append({"idx": i, "easting": e, "northing": nrt, "dist_m": float(d)})
    return rows


def find_pose_only_windows(rows: list[dict], distance_thresh: float = 30.0,
                           min_window: int = 100,
                           hole_tolerance: int = 5) -> list[CandidateWindow]:
    """Hole-tolerant runs of rows with dist_m ≤ threshold."""
    ok_signal = [r["dist_m"] <= distance_thresh for r in rows]
    runs = _hole_tolerant_runs(ok_signal, hole_tolerance)

    out: list[CandidateWindow] = []
    for s, e in runs:
        nframes = e - s + 1
        if nframes < min_window:
            continue
        n_ok = sum(1 for i in range(s, e + 1) if ok_signal[i])
        pct = n_ok / nframes
        dists = [rows[i]["dist_m"] for i in range(s, e + 1)]
        # Score: longer + lower mean distance is better.
        score = float(nframes * pct / (1.0 + np.mean(dists)))
        out.append(CandidateWindow(
            start_idx=s,
            end_idx=e,
            n_frames=nframes,
            n_ok=n_ok,
            pct_ok=pct,
            score=score,
            snow_seq_start=s,
            snow_seq_end=e,
        ))
    out.sort(key=lambda w: -w.score)
    return out


# ── Persistent curation log ─────────────────────────────────────────────

_AUDIT_LOG_PATH = ROOT / "docs" / "dev" / "audit_log.md"
_CURATION_HEADING = "## Curation decisions per track"


def append_audit_log_row(track_id: str, top_window: CandidateWindow,
                         hole_tolerance: int) -> None:
    """Append (or update) a row in docs/audit_log.md under a stable
    'Curation decisions per track' section. Idempotent: re-running the
    oracle replaces the previous row for the same track_id."""
    if not _AUDIT_LOG_PATH.exists():
        return  # don't auto-create; the doc is hand-curated
    text = _AUDIT_LOG_PATH.read_text()

    today = datetime.now().strftime("%Y-%m-%d")
    row = (
        f"| `{track_id}` | "
        f"frames {top_window.snow_seq_start}–{top_window.snow_seq_end} "
        f"({top_window.n_frames}) | "
        f"{100 * top_window.pct_ok:.1f}% | "
        f"hole≤{hole_tolerance}f | "
        f"{today} |"
    )

    if _CURATION_HEADING not in text:
        # Append a new section at the end of the file with the table header.
        section = (
            f"\n\n---\n\n{_CURATION_HEADING}\n\n"
            f"Generated by `src.video_runtime.window_oracle` "
            f"after each `make track`.\n\n"
            f"| Track | Best window (snow stream indices) | OK% | Smoothing | Curated |\n"
            f"|---|---|---|---|---|\n"
            f"{row}\n"
        )
        _AUDIT_LOG_PATH.write_text(text + section)
        return

    # Section exists. Replace any previous row for this track_id, or append.
    lines = text.splitlines()
    # Find the table block under our heading.
    in_section = False
    table_start = None
    table_end = None
    for i, line in enumerate(lines):
        if line.strip() == _CURATION_HEADING:
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and line.startswith("|---"):
            table_start = i + 1
        if in_section and table_start is not None and table_end is None:
            if i >= table_start and (not line.startswith("|") or line.strip() == ""):
                table_end = i
                break
    if table_start is None:
        return  # malformed; leave alone
    if table_end is None:
        table_end = len(lines)

    new_lines: list[str] = []
    replaced = False
    for i, line in enumerate(lines):
        if table_start <= i < table_end and line.startswith(f"| `{track_id}`"):
            new_lines.append(row)
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.insert(table_end, row)
    _AUDIT_LOG_PATH.write_text("\n".join(new_lines) + "\n")


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--cache-tag", default="canonical")
    p.add_argument("--hole-tolerance", type=int, default=5,
                   help="Frames-of-bad-priors gap to bridge during smoothing "
                        "(default 5 ≈ 0.5 s at 10 fps).")
    p.add_argument("--min-window", type=int, default=50,
                   help="Reject candidate windows shorter than this many frames.")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--out-json", default=None,
                   help="Write the per-frame report + candidate windows here.")
    p.add_argument("--poses-only", action="store_true",
                   help="Skip cache-reading; report nearest-summer-pose "
                        "distance from the full parent camera_poses.csv.")
    p.add_argument("--no-audit-log", action="store_true",
                   help="Skip appending a row to docs/audit_log.md.")
    p.add_argument("--distance-thresh", type=float, default=30.0,
                   help="(--poses-only) reject snow frames whose nearest "
                        "summer pose is farther than this many metres.")
    args = p.parse_args()

    if args.poses_only:
        rows = evaluate_full_track_poses_only(
            args.track, distance_thresh=args.distance_thresh,
        )
        windows = find_pose_only_windows(
            rows, distance_thresh=args.distance_thresh,
            min_window=max(args.min_window, 100),
            hole_tolerance=args.hole_tolerance,
        )
        n_ok = sum(1 for r in rows if r["dist_m"] <= args.distance_thresh)
        print(f"\n=== pose-only oracle — {args.track} ===")
        print(f"  total snow rows:        {len(rows)}")
        print(f"  within {args.distance_thresh}m of summer:  "
              f"{n_ok} ({100 * n_ok // max(len(rows), 1)}%)")
        print(f"  candidate windows (after hole={args.hole_tolerance} smoothing):")
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
                           "hole_tolerance": args.hole_tolerance,
                           "min_window": args.min_window},
                "n_total_rows": len(rows),
                "n_within_threshold": n_ok,
                "candidate_windows": [
                    {"start_idx": w.start_idx, "end_idx": w.end_idx,
                     "n_frames": w.n_frames, "pct_ok": w.pct_ok,
                     "score": w.score}
                    for w in windows[:args.top_n]
                ],
            }, indent=2))
            print(f"  → wrote {out}")
        return

    # Default: cache-backed curation report.
    frame_oracles, windows = curate_from_cache(
        args.track,
        cache_tag=args.cache_tag,
        hole_tolerance=args.hole_tolerance,
        min_window=args.min_window,
        top_n=args.top_n,
    )
    print_report(frame_oracles, windows, args.track, args.hole_tolerance)

    if windows and not args.no_audit_log:
        append_audit_log_row(args.track, windows[0], args.hole_tolerance)
        print(f"  → appended row to {_AUDIT_LOG_PATH.relative_to(ROOT)}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "track": args.track,
            "mode": "cache_backed",
            "cache_tag": args.cache_tag,
            "params": {
                "hole_tolerance": args.hole_tolerance,
                "min_window": args.min_window,
            },
            "n_frames": len(frame_oracles),
            "n_ok": sum(1 for r in frame_oracles if r.is_ok),
            "candidate_windows": [
                {"start_idx": w.start_idx, "end_idx": w.end_idx,
                 "snow_seq_start": w.snow_seq_start, "snow_seq_end": w.snow_seq_end,
                 "n_frames": w.n_frames, "n_ok": w.n_ok, "pct_ok": w.pct_ok,
                 "score": w.score}
                for w in windows
            ],
        }, indent=2))
        print(f"  → wrote {out}")


if __name__ == "__main__":
    main()
