"""Phase K.1 — Boreas track fetcher.

Pulls a small (~30s) snow + summer subset of a Boreas track using direct
S3 REST. No AWS CLI required (bucket is public CC BY 4.0).

Usage:
    uv run python -m src.video_runtime.fetch_track --track <track_id>

For now the only supported track is `boreas_2021_01_26` which pairs
`boreas-2021-01-26-11-22` (heavy snow) with `boreas-2021-07-27-14-43`
(clear summer). Both traverse the same UTIAS-Toronto loop so a snow
window can be GPS-aligned to a summer window via UTM (easting, northing).

Output layout:
    data/video/tracks/<track_id>/
        snow/
            camera_poses.csv         (full, already pulled)
            calib/                   (intrinsics + extrinsics)
            frames/<ts>.png          (subset, ~300 frames)
            window.json              (window metadata: ts range, gps bounds)
        summer/
            (same)
        track.json                   (top-level track config + license)

Reuses the existing S3 listing pattern (REST + ?list-type=2).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/video/tracks"

S3 = "https://boreas.s3.us-west-2.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# ----------------------------------------------------------------------------
# Track registry — small enough to keep inline.
# ----------------------------------------------------------------------------

TRACKS = {
    "boreas_2021_01_26": {
        "snow_seq": "boreas-2021-01-26-11-22",
        "summer_seq": "boreas-2021-07-27-14-43",
        # Visually inspected window in the snow sequence — residential street
        # with the road buried, lane markings invisible, sidewalk seam not
        # drawn. ~30 s @ 10 Hz = ~300 frames.
        "snow_window_seconds": (140.0, 175.0),
        "license": "CC BY 4.0 (Boreas, Burnett et al. UTIAS-ASRL, IJRR 2023)",
        "attribution": "Boreas dataset (UTIAS-ASRL). Cite Burnett et al. 2023; CC BY 4.0.",
    },
    # Note: boreas-2021-02-09-12-55 has no camera_poses.csv on S3 (Boreas
    # publishes Applanix post-processing only for some sequences). Skipped.
    # Other 2021 winter sequences with poses available, if a sunny-day
    # alternative is needed: 2020-12-18-13-44, 2021-01-15-12-17,
    # 2021-01-19-15-08, 2021-02-02-14-07, 2021-03-02-13-38.
    "boreas_2024_12_23": {
        "snow_seq": "boreas-2024-12-23-16-27",
        # Use the known-good 2021-07-27 summer for all alt tracks. Per-sequence
        # local UTM frames vary in origin between Boreas runs, so cross-sequence
        # alignment with arbitrary summer pairings fails. 2021-07-27 has been
        # verified to align cleanly with the Glen Shields loop coordinates.
        "summer_seq": "boreas-2021-07-27-14-43",
        # Dusk residential, quiet street, light snow on ground.
        "snow_window_seconds": (80.0, 115.0),
        "license": "CC BY 4.0 (Boreas, Burnett et al. UTIAS-ASRL, IJRR 2023)",
        "attribution": "Boreas dataset (UTIAS-ASRL). Cite Burnett et al. 2023; CC BY 4.0.",
    },
    "boreas_2025_02_15": {
        "snow_seq": "boreas-2025-02-15-16-58",
        "summer_seq": "boreas-2021-07-27-14-43",
        # Late afternoon active snowfall, commercial street with traffic and
        # apartment blocks. Most cinematic of the alt set.
        "snow_window_seconds": (80.0, 115.0),
        "license": "CC BY 4.0 (Boreas, Burnett et al. UTIAS-ASRL, IJRR 2023)",
        "attribution": "Boreas dataset (UTIAS-ASRL). Cite Burnett et al. 2023; CC BY 4.0.",
    },
}


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def _list_keys(prefix: str, max_keys: int = 1000) -> list[tuple[str, int]]:
    """Paginated S3 v2 listing. Returns [(key, size), ...]."""
    out: list[tuple[str, int]] = []
    cont: str | None = None
    while True:
        url = f"{S3}/?list-type=2&prefix={prefix}&max-keys={max_keys}"
        if cont:
            url += f"&continuation-token={urllib.parse.quote(cont)}"
        xml = _fetch(url).decode()
        root = ET.fromstring(xml)
        for c in root.findall("s3:Contents", NS):
            k = c.findtext("s3:Key", default="", namespaces=NS)
            s = int(c.findtext("s3:Size", default="0", namespaces=NS) or 0)
            out.append((k, s))
        is_trunc = root.findtext("s3:IsTruncated", default="false", namespaces=NS) == "true"
        if not is_trunc:
            break
        cont = root.findtext("s3:NextContinuationToken", default="", namespaces=NS)
    return out


def _load_camera_poses(csv_path: Path) -> np.ndarray:
    """Parse camera_poses.csv → structured ndarray with GPSTime, easting, northing, heading.

    GPSTime is a microsecond integer that matches the camera frame filenames.
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return data


def _pick_snow_window(snow_poses: np.ndarray, t0: float, t1: float) -> tuple[int, int]:
    """Return (start_idx, end_idx) into snow_poses covering t0..t1 seconds since the first frame."""
    ts = snow_poses["GPSTime"].astype(np.int64)
    base = ts[0]
    rel = (ts - base) / 1e6  # seconds since first
    start = int(np.searchsorted(rel, t0))
    end = int(np.searchsorted(rel, t1))
    return start, end


def _align_summer_window(
    snow_poses: np.ndarray, snow_start: int, snow_end: int, summer_poses: np.ndarray
) -> tuple[int, int]:
    """Find the contiguous summer window whose (easting, northing) trajectory best matches
    the snow window."""
    snow_xy = np.stack([snow_poses["easting"][snow_start:snow_end], snow_poses["northing"][snow_start:snow_end]], axis=1)
    summer_xy = np.stack([summer_poses["easting"], summer_poses["northing"]], axis=1)

    # Snap each snow frame to the nearest summer frame in (x, y), then take the
    # contiguous range of summer indices that are reached.
    summer_idx_for_snow = []
    for s in snow_xy:
        d = np.sum((summer_xy - s) ** 2, axis=1)
        summer_idx_for_snow.append(int(np.argmin(d)))
    summer_idx_for_snow = np.array(summer_idx_for_snow, dtype=np.int64)

    s_start = int(summer_idx_for_snow.min())
    s_end = int(summer_idx_for_snow.max() + 1)

    # Sanity check: was the alignment monotonic? (If not, snow loop crossed
    # summer's path or we're nowhere near.)
    monotonic = bool(np.all(np.diff(summer_idx_for_snow) >= -3))  # tolerate small wobble
    return s_start, s_end, monotonic


def _download_one(args: tuple[str, Path], *, max_retries: int = 4) -> tuple[str, int]:
    url, dest = args
    if dest.exists() and dest.stat().st_size > 0:
        return url, dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            body = _fetch(url)
            dest.write_bytes(body)
            return url, len(body)
        except Exception as e:
            last_exc = e
            # Exponential back-off on DNS / transient HTTP errors. Don't blow
            # up the whole pull because one frame timed out.
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch {url} after {max_retries} retries: {last_exc}")


def _pull_window(seq: str, gpstimes: list[int], dest: Path, label: str) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    jobs = []
    for ts in gpstimes:
        url = f"{S3}/{seq}/camera/{ts}.png"
        jobs.append((url, dest / f"{ts}.png"))
    print(f"  {label}: downloading {len(jobs)} frames in parallel...", flush=True)
    total = 0
    failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed(ex.submit(_download_one, j) for j in jobs):
            try:
                url, size = fut.result()
                total += size
            except Exception as e:
                # Don't kill the whole pull because one frame failed.
                failed += 1
                print(f"    [FAIL] {e}", flush=True)
    msg = f"  {label}: {total / 1024 / 1024:.0f} MB in {time.time() - t0:.0f}s"
    if failed:
        msg += f"  ({failed} failed)"
    print(msg, flush=True)
    return total


def _bootstrap_metadata(seq: str, target_dir: Path) -> None:
    """Pull camera_poses.csv + calib/* from S3 if not already on disk."""
    target_dir.mkdir(parents=True, exist_ok=True)
    poses = target_dir / "camera_poses.csv"
    if not poses.exists() or poses.stat().st_size == 0:
        print(f"  pulling {seq}/applanix/camera_poses.csv...")
        body = _fetch(f"{S3}/{seq}/applanix/camera_poses.csv")
        poses.write_bytes(body)
    calib_dir = target_dir / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    for calib_name in ("P_camera.txt", "T_applanix_lidar.txt", "T_camera_lidar.txt", "camera0_intrinsics.yaml"):
        f = calib_dir / calib_name
        if f.exists() and f.stat().st_size > 0:
            continue
        try:
            f.write_bytes(_fetch(f"{S3}/{seq}/calib/{calib_name}"))
        except Exception as e:
            print(f"    skip {calib_name}: {e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True, choices=list(TRACKS.keys()))
    p.add_argument("--max-frames", type=int, default=350,
                   help="Cap on frames per sequence (default 350 ≈ 35 s @ 10 Hz)")
    args = p.parse_args()

    spec = TRACKS[args.track]
    track_dir = DATA / args.track
    snow_dir = track_dir / "snow"
    summer_dir = track_dir / "summer"
    snow_dir.mkdir(parents=True, exist_ok=True)
    summer_dir.mkdir(parents=True, exist_ok=True)

    # 0. Bootstrap metadata (camera_poses + calib) if missing.
    print(f"[{args.track}] bootstrapping metadata for {spec['snow_seq']} + {spec['summer_seq']}")
    _bootstrap_metadata(spec["snow_seq"], snow_dir)
    _bootstrap_metadata(spec["summer_seq"], summer_dir)

    # 1. Load camera_poses.csv for both.
    print(f"[{args.track}] reading camera poses...")
    snow_poses = _load_camera_poses(snow_dir / "camera_poses.csv")
    summer_poses = _load_camera_poses(summer_dir / "camera_poses.csv")
    print(f"  snow:   {len(snow_poses)} poses,   t0={snow_poses['GPSTime'][0]},   t-1={snow_poses['GPSTime'][-1]}")
    print(f"  summer: {len(summer_poses)} poses, t0={summer_poses['GPSTime'][0]}, t-1={summer_poses['GPSTime'][-1]}")

    # 2. Pick snow window.
    t0, t1 = spec["snow_window_seconds"]
    s_start, s_end = _pick_snow_window(snow_poses, t0, t1)
    s_end = min(s_end, s_start + args.max_frames)
    snow_indices = list(range(s_start, s_end))
    print(f"  snow window: indices [{s_start}, {s_end}) = {len(snow_indices)} frames "
          f"({t0:.1f}s .. {t1:.1f}s relative)")

    # 3. Find summer window via UTM nearest-neighbour.
    sm_start, sm_end, monotonic = _align_summer_window(snow_poses, s_start, s_end, summer_poses)
    sm_end = min(sm_end, sm_start + args.max_frames)
    summer_indices = list(range(sm_start, sm_end))
    print(f"  summer window: indices [{sm_start}, {sm_end}) = {len(summer_indices)} frames")
    print(f"  alignment monotonic: {monotonic}")
    if not monotonic:
        print("  WARNING: summer alignment was not monotonic — snow loop may not match summer at all.")
        print("  Investigate before proceeding to K.2.")

    # 4. Sanity print: snow start (x, y) vs summer start (x, y).
    print(f"  snow start UTM:   ({snow_poses['easting'][s_start]:+.2f}, {snow_poses['northing'][s_start]:+.2f})")
    print(f"  summer start UTM: ({summer_poses['easting'][sm_start]:+.2f}, {summer_poses['northing'][sm_start]:+.2f})")

    # 5. Pull the frames.
    snow_ts = [int(snow_poses["GPSTime"][i]) for i in snow_indices]
    summer_ts = [int(summer_poses["GPSTime"][i]) for i in summer_indices]

    _pull_window(spec["snow_seq"], snow_ts, snow_dir / "frames", "snow")
    _pull_window(spec["summer_seq"], summer_ts, summer_dir / "frames", "summer")

    # 6. Persist window metadata.
    snow_window = {
        "seq": spec["snow_seq"],
        "indices": [s_start, s_end],
        "gpstimes": snow_ts,
        "utm_bounds": {
            "easting": [float(snow_poses["easting"][snow_indices].min()), float(snow_poses["easting"][snow_indices].max())],
            "northing": [float(snow_poses["northing"][snow_indices].min()), float(snow_poses["northing"][snow_indices].max())],
        },
    }
    summer_window = {
        "seq": spec["summer_seq"],
        "indices": [sm_start, sm_end],
        "gpstimes": summer_ts,
        "utm_bounds": {
            "easting": [float(summer_poses["easting"][summer_indices].min()), float(summer_poses["easting"][summer_indices].max())],
            "northing": [float(summer_poses["northing"][summer_indices].min()), float(summer_poses["northing"][summer_indices].max())],
        },
        "alignment_monotonic": monotonic,
    }
    (snow_dir / "window.json").write_text(json.dumps(snow_window, indent=2))
    (summer_dir / "window.json").write_text(json.dumps(summer_window, indent=2))

    track = {
        "track_id": args.track,
        "snow": snow_window,
        "summer": summer_window,
        "license": spec["license"],
        "attribution": spec["attribution"],
    }
    (track_dir / "track.json").write_text(json.dumps(track, indent=2))
    print(f"\n[{args.track}] done. {len(snow_indices)} snow + {len(summer_indices)} summer frames on disk.")


if __name__ == "__main__":
    main()
