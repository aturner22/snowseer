"""Fetch a Mapillary winter sequence + per-frame summer priors into the
Boreas-compatible track structure expected by `src.video_runtime.Track`.

Bridges the gap between Mapillary's image API (lat/lng/sequence) and the
existing video pipeline (which expects camera_poses.csv with
GPSTime/easting/northing/heading + window.json).

Usage:
    uv run python -m data.fetch_mapillary_video \\
        --sequence-id <mapillary_seq_id> \\
        --track-id <local_name>

Produces:
    data/video/tracks/<local_name>/
        track.json
        snow/
            frames/<image_id>.png
            camera_poses.csv      (synthetic UTM via local equirectangular)
            window.json
        summer/
            frames/<image_id>.png
            camera_poses.csv      (deduplicated pool of summer priors)
            window.json

UTM is approximated via local equirectangular projection at the track's
mean latitude — accurate to < 1 m for tracks under a few km. The pipeline
only consumes (easting, northing) for KD-tree queries; absolute UTM zone
doesn't matter as long as both halves use the same origin.

For the oracle's prior-availability check, we query Mapillary `closeto`
per snow frame within `--radius-m` (default 100 m) for any-year summer
(default 2018-06 .. 2024-09). Hit rate is logged; frames with zero hits
get a placeholder summer pose (the snow pose itself, marked unusable
when the oracle runs — the segmenter will reject empty placeholders).

Calib files (P_camera.txt etc.) are NOT generated — the matching pipeline
only needs frames + poses. Mapillary doesn't provide calibration anyway.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKS = ROOT / "data/video/tracks"

GRAPH_URL = "https://graph.mapillary.com"

DEFAULT_SUMMER_RANGE = ("2018-06-01", "2024-09-30")
DEFAULT_RADIUS_M = 100


def _fetch_json(url: str, *, timeout: float = 60.0, retries: int = 3) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "snow-underlay-fetch/0.2"})
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            last = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise last


def _fetch_bytes(url: str, *, timeout: float = 60.0, retries: int = 2) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "snow-underlay-fetch/0.2"})
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise last


def _list_sequence(token: str, sequence_id: str) -> list[dict]:
    """Pull every image in a Mapillary sequence with metadata."""
    print(f"[fetch] listing sequence {sequence_id}...", flush=True)
    # The /image_ids endpoint returns just the IDs; we then query each for full metadata.
    # Cheaper: query the /images endpoint with sequence_ids filter.
    qs = urllib.parse.urlencode({
        "access_token": token,
        "sequence_ids": sequence_id,
        "fields": "id,sequence,geometry,captured_at,compass_angle,is_pano,thumb_2048_url",
        "limit": 2000,
    })
    j = _fetch_json(f"{GRAPH_URL}/images?{qs}")
    imgs = j.get("data", [])
    imgs.sort(key=lambda x: int(x.get("captured_at", 0)))
    print(f"[fetch]   found {len(imgs)} images in sequence", flush=True)
    return imgs


def _query_closeto(token: str, lng: float, lat: float, radius_m: int,
                   start_iso: str, end_iso: str, *, exclude_seq: str | None = None,
                   limit: int = 5) -> list[dict]:
    delta = radius_m / 111_000.0
    bbox = f"{lng-delta},{lat-delta},{lng+delta},{lat+delta}"
    qs = urllib.parse.urlencode({
        "access_token": token,
        "bbox": bbox,
        "start_captured_at": f"{start_iso}T00:00:00Z",
        "end_captured_at": f"{end_iso}T23:59:59Z",
        "fields": "id,sequence,geometry,captured_at,compass_angle,is_pano,thumb_2048_url",
        "limit": limit,
    })
    try:
        results = _fetch_json(f"{GRAPH_URL}/images?{qs}").get("data", [])
    except Exception:
        return []
    keep = []
    for r in results:
        if r.get("is_pano"):
            continue
        if exclude_seq and r.get("sequence") == exclude_seq:
            continue
        keep.append(r)
    return keep


def _get_thumb_url(token: str, image_id: str) -> str | None:
    """Get a fresh thumb_2048_url for an image (the cached URLs expire)."""
    qs = urllib.parse.urlencode({
        "access_token": token,
        "fields": "thumb_2048_url",
    })
    try:
        j = _fetch_json(f"{GRAPH_URL}/{image_id}?{qs}")
        return j.get("thumb_2048_url")
    except Exception:
        return None


def _equirectangular_xy(lng: float, lat: float, lat0: float) -> tuple[float, float]:
    """Local equirectangular projection (meters). Anchor at (0, 0) for the
    track's mean latitude. Accurate to < 1 m for tracks under a few km."""
    cos_lat0 = math.cos(math.radians(lat0))
    easting = lng * 111_320.0 * cos_lat0
    northing = lat * 110_540.0
    return easting, northing


def _download_pool(token: str, images: list[dict], target_dir: Path,
                   *, max_workers: int = 8) -> dict[str, dict]:
    """Download images concurrently. Filename = `<captured_at_us>.png` so
    `Track._build_meta` (which expects `<GPSTime>.png`) can find them.

    Returns map image_id → metadata + filepath. The captured_at field is
    in milliseconds; we multiply by 1000 to get microseconds (matches
    Boreas's GPSTime convention)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    def _one(img: dict) -> tuple[str, dict | None]:
        image_id = img["id"]
        gpstime_us = int(img["captured_at"]) * 1000
        out = target_dir / f"{gpstime_us}.png"
        if out.exists() and out.stat().st_size > 0:
            return image_id, {**img, "_path": out, "_skipped": True}
        url = img.get("thumb_2048_url") or _get_thumb_url(token, image_id)
        if not url:
            return image_id, None
        try:
            body = _fetch_bytes(url)
            out.write_bytes(body)
            return image_id, {**img, "_path": out, "_skipped": False}
        except Exception as e:
            print(f"  download failed for {image_id}: {e}", flush=True)
            return image_id, None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_one, img): img for img in images}
        n_done = 0
        for f in as_completed(futs):
            image_id, meta = f.result()
            n_done += 1
            if meta is not None:
                results[image_id] = meta
            if n_done % 25 == 0:
                print(f"  downloaded {n_done}/{len(images)}", flush=True)
    return results


def _write_camera_poses(target_dir: Path, images: list[dict], lat0: float) -> None:
    """Write Boreas-compatible camera_poses.csv with synthetic UTM."""
    target_dir.mkdir(parents=True, exist_ok=True)
    p = target_dir / "camera_poses.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["GPSTime", "easting", "northing", "altitude",
                    "vel_east", "vel_north", "vel_up",
                    "roll", "pitch", "heading", "ang_vel_z"])
        for img in images:
            lng, lat = img["geometry"]["coordinates"]
            easting, northing = _equirectangular_xy(lng, lat, lat0)
            heading = float(img.get("compass_angle") or 0.0)
            captured_us = int(img["captured_at"]) * 1000   # ms→μs to mimic Boreas GPSTime
            w.writerow([captured_us, f"{easting:.4f}", f"{northing:.4f}", "0.0",
                        "0.0", "0.0", "0.0", "0.0", "0.0", f"{heading:.4f}", "0.0"])


def _write_window_json(target_dir: Path, images: list[dict], seq_id: str) -> None:
    """Write window.json — pipeline_v.Track expects this for indices + gpstimes."""
    p = target_dir / "window.json"
    n = len(images)
    p.write_text(json.dumps({
        "seq": seq_id,
        "indices": [0, n],   # entire range; Track will load all
        "gpstimes": [int(img["captured_at"]) * 1000 for img in images],
    }, indent=2))


def fetch(token: str, sequence_id: str, track_id: str,
          *, radius_m: int = DEFAULT_RADIUS_M,
          summer_range: tuple[str, str] = DEFAULT_SUMMER_RANGE,
          max_frames: int | None = None) -> None:
    track_dir = TRACKS / track_id
    snow_dir = track_dir / "snow"
    summer_dir = track_dir / "summer"

    # 1. List the snow sequence
    snow_imgs = _list_sequence(token, sequence_id)
    if max_frames is not None:
        snow_imgs = snow_imgs[:max_frames]
        print(f"[fetch]   capped at {max_frames} snow frames", flush=True)

    if not snow_imgs:
        raise SystemExit(f"sequence {sequence_id} returned 0 images")

    # 2. Compute the track's anchor latitude for local equirectangular.
    lats = [float(i["geometry"]["coordinates"][1]) for i in snow_imgs]
    lngs = [float(i["geometry"]["coordinates"][0]) for i in snow_imgs]
    lat0 = sum(lats) / len(lats)
    lng0 = sum(lngs) / len(lngs)
    print(f"[fetch]   track anchor: lat0={lat0:.4f} lng0={lng0:.4f}  "
          f"({len(snow_imgs)} snow frames spanning lat=[{min(lats):.4f}..{max(lats):.4f}])",
          flush=True)

    # 3. Per-snow-frame summer-prior probe (with caching).
    print(f"[fetch] probing summer priors (radius={radius_m}m, "
          f"window={summer_range[0]}..{summer_range[1]}) ...", flush=True)
    summer_pool: dict[str, dict] = {}
    n_hits = 0
    for i, img in enumerate(snow_imgs):
        lng, lat = img["geometry"]["coordinates"]
        candidates = _query_closeto(
            token, lng, lat, radius_m,
            summer_range[0], summer_range[1],
            exclude_seq=sequence_id, limit=3,
        )
        if candidates:
            n_hits += 1
            for c in candidates:
                summer_pool.setdefault(c["id"], c)
        if (i + 1) % 50 == 0:
            print(f"  probed {i+1}/{len(snow_imgs)}  "
                  f"({n_hits} snow frames with ≥1 prior, "
                  f"{len(summer_pool)} unique summer images so far)", flush=True)
    print(f"[fetch]   {n_hits}/{len(snow_imgs)} snow frames had ≥1 summer prior  "
          f"(unique summer images: {len(summer_pool)})", flush=True)
    if n_hits == 0:
        raise SystemExit("no summer priors found — try a wider radius or different sequence")

    # 4. Download all snow frames + summer pool images.
    print(f"[fetch] downloading {len(snow_imgs)} snow frames...", flush=True)
    snow_results = _download_pool(token, snow_imgs, snow_dir / "frames")
    print(f"[fetch]   {len(snow_results)} snow images on disk", flush=True)

    print(f"[fetch] downloading {len(summer_pool)} summer pool images...", flush=True)
    summer_results = _download_pool(token, list(summer_pool.values()),
                                    summer_dir / "frames")
    print(f"[fetch]   {len(summer_results)} summer images on disk", flush=True)

    # 5. Filter to images we successfully downloaded (preserving order).
    snow_kept = [i for i in snow_imgs if i["id"] in snow_results]
    summer_kept = [i for i in summer_pool.values() if i["id"] in summer_results]
    summer_kept.sort(key=lambda x: int(x["captured_at"]))

    # 6. Write Boreas-compatible metadata.
    _write_camera_poses(snow_dir, snow_kept, lat0)
    _write_camera_poses(summer_dir, summer_kept, lat0)
    _write_window_json(snow_dir, snow_kept, sequence_id)
    _write_window_json(summer_dir, summer_kept, "summer_pool_per_frame_closeto")

    # 7. Top-level track.json.
    track_json = track_dir / "track.json"
    track_json.write_text(json.dumps({
        "track_id": track_id,
        "source": "mapillary_v4",
        "snow": {
            "seq": sequence_id,
            "indices": [0, len(snow_kept)],
            "lat0": lat0, "lng0": lng0,
            "n_frames": len(snow_kept),
            "first_capture_iso": time.strftime(
                "%Y-%m-%d %H:%M",
                time.gmtime(int(snow_kept[0]["captured_at"]) / 1000),
            ),
        },
        "summer": {
            "source": "per_frame_closeto",
            "radius_m": radius_m,
            "date_range": list(summer_range),
            "indices": [0, len(summer_kept)],
            "n_frames": len(summer_kept),
        },
        "notes": (
            "Mapillary v4 sequence + per-frame summer-prior closeto query. "
            "UTM is local equirectangular anchored at lat0."
        ),
    }, indent=2))

    # Calib dirs (empty — matching pipeline doesn't need calib)
    (snow_dir / "calib").mkdir(parents=True, exist_ok=True)
    (summer_dir / "calib").mkdir(parents=True, exist_ok=True)

    print(f"\n[fetch] done.")
    print(f"  track at:       {track_dir}")
    print(f"  snow frames:    {len(snow_kept)}")
    print(f"  summer pool:    {len(summer_kept)}")
    print(f"\nNext step:")
    print(f"  make oracle TRACK={track_id}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence-id", required=True,
                   help="Mapillary sequence_id (use data.find_snow_for_video to discover)")
    p.add_argument("--track-id", required=True,
                   help="local track name, e.g. 'mapillary_tromso_2025_winter'")
    p.add_argument("--radius-m", type=int, default=DEFAULT_RADIUS_M)
    p.add_argument("--summer-start", default=DEFAULT_SUMMER_RANGE[0])
    p.add_argument("--summer-end", default=DEFAULT_SUMMER_RANGE[1])
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap snow-frame count (for quick smoke tests)")
    args = p.parse_args()

    from data.fetch_mapillary import _load_dotenv
    _load_dotenv()
    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        raise SystemExit("MAPILLARY_TOKEN not set in environment or .env.")

    fetch(
        token, args.sequence_id, args.track_id,
        radius_m=args.radius_m,
        summer_range=(args.summer_start, args.summer_end),
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
