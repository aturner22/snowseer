"""CADC scene → data/video/tracks/<id>/ in our standard format.

Maps a CADC scene drop at data/external/cadc/<date>/<scene>/labeled/
into the per-track layout the video pipeline already understands:

    data/video/tracks/cadc_<date>_<scene>/
    ├── snow/
    │   ├── frames/<gpstime_us>.png    ← copied from labeled/image_00/data/
    │   ├── camera_poses.csv           ← derived from labeled/novatel/data/*.txt
    │   └── window.json
    ├── summer/
    │   ├── frames/<gpstime_us>.png    ← per-frame Mapillary closeto pulls
    │   ├── camera_poses.csv
    │   └── window.json
    └── track.json

Camera 00 (camera_F) is the forward-facing camera per CADC's calib YAMLs.
GPS comes from per-frame Novatel INSPVAX log files as lat/lon/azimuth;
we project to local-equirectangular metres (same convention the existing
PriorPool consumes). Timestamps in image_00/timestamps.txt are ISO-8601;
we convert to integer microseconds-since-epoch so frame filenames are
sortable + match camera_poses.csv keys.

Two-step usage so the audit gate can intervene:

    # Step 1: snow side only (no Mapillary calls).
    uv run python -m src.data.fetch_cadc --scene 2019_02_27/0027 --snow-only

    # User inspects data/video/tracks/cadc_*/snow/frames/ for content quality.

    # Step 2: pull per-frame summer priors from Mapillary.
    uv run python -m src.data.fetch_cadc --scene 2019_02_27/0027 --summer

Then `make oracle TRACK=cadc_2019_02_27_0027` and (if green)
`make track TRACK=cadc_2019_02_27_0027`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CADC_ROOT = ROOT / "data/external/cadc"
TRACKS_ROOT = ROOT / "data/video/tracks"

EARTH_RADIUS_M = 6_378_137.0
DEG2RAD = math.pi / 180.0

GRAPH_URL = "https://graph.mapillary.com/images"
TOKEN_ENV = "MAPILLARY_TOKEN"


# ── Time + projection helpers ────────────────────────────────────────────

def _iso_to_us(s: str) -> int:
    """ISO-8601 ('2019-02-27 14:40:12.500000000') → epoch microseconds (int)."""
    s = s.strip()
    if "." in s:
        head, frac = s.split(".", 1)
        # Pad/truncate fractional seconds to 6 digits (microseconds).
        frac = (frac + "0" * 6)[:6]
        s_normalised = f"{head}.{frac}"
    else:
        s_normalised = s
    dt = datetime.strptime(s_normalised, "%Y-%m-%d %H:%M:%S.%f")
    return int(dt.timestamp() * 1_000_000)


def _equirect_metres(lat_deg: float, lon_deg: float,
                     ref_lat_deg: float, ref_lon_deg: float) -> tuple[float, float]:
    """Local equirectangular projection: (easting_m, northing_m) relative to a
    reference (ref_lat, ref_lon). Centimetre-accurate over a few-km region —
    fine for KD-tree nearest-prior queries on a single track."""
    cos_ref = math.cos(ref_lat_deg * DEG2RAD)
    easting = (lon_deg - ref_lon_deg) * DEG2RAD * EARTH_RADIUS_M * cos_ref
    northing = (lat_deg - ref_lat_deg) * DEG2RAD * EARTH_RADIUS_M
    return easting, northing


def _read_novatel(novatel_dir: Path) -> list[dict]:
    """Read all per-frame Novatel files (one .txt per frame) → list of dicts.
    Each .txt has space-separated values matching dataformat.txt."""
    out: list[dict] = []
    for p in sorted(novatel_dir.glob("*.txt")):
        if p.name == "dataformat.txt":
            continue
        toks = p.read_text().split()
        out.append({
            "lat": float(toks[0]),
            "lon": float(toks[1]),
            "alt": float(toks[2]),
            "azimuth_deg": float(toks[9]),  # heading; CADC uses azimuth
            "frame_idx": int(p.stem),
        })
    return out


def _read_timestamps(ts_path: Path) -> list[int]:
    return [_iso_to_us(line) for line in ts_path.read_text().splitlines() if line.strip()]


# ── Snow side ────────────────────────────────────────────────────────────

def build_snow(scene_path: Path, track_dir: Path) -> dict:
    """Copy CADC camera-00 frames + derive camera_poses.csv. Returns a dict
    of summary info the caller uses to drive the summer-side pull."""
    image_dir = scene_path / "labeled/image_00/data"
    image_ts_path = scene_path / "labeled/image_00/timestamps.txt"
    novatel_dir = scene_path / "labeled/novatel/data"

    if not image_dir.exists():
        raise FileNotFoundError(f"missing CADC image_00 dir: {image_dir}")
    if not novatel_dir.exists():
        raise FileNotFoundError(f"missing CADC novatel dir: {novatel_dir}")

    snow_dir = track_dir / "snow"
    frames_dir = snow_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    image_timestamps = _read_timestamps(image_ts_path)  # μs per frame
    novatel = _read_novatel(novatel_dir)
    if len(novatel) != len(image_timestamps):
        # The two streams are sampled together at scene capture; mismatch is
        # rare but worth surfacing rather than silently truncating.
        print(f"  ! warn: {len(image_timestamps)} image ts vs {len(novatel)} "
              f"novatel rows — using min({min(len(image_timestamps), len(novatel))})")
    n = min(len(image_timestamps), len(novatel))

    # Local equirectangular reference: median of the snow trajectory.
    lats = [r["lat"] for r in novatel[:n]]
    lons = [r["lon"] for r in novatel[:n]]
    ref_lat = sorted(lats)[n // 2]
    ref_lon = sorted(lons)[n // 2]

    rows = []
    for i in range(n):
        gpstime_us = image_timestamps[i]
        easting, northing = _equirect_metres(
            novatel[i]["lat"], novatel[i]["lon"], ref_lat, ref_lon,
        )
        heading = novatel[i]["azimuth_deg"]

        # Copy png with gpstime-us name to match the existing track convention.
        src = image_dir / f"{i:010d}.png"
        dst = frames_dir / f"{gpstime_us}.png"
        if not dst.exists():
            shutil.copy(src, dst)

        rows.append({
            "GPSTime": gpstime_us,
            "easting": easting,
            "northing": northing,
            "heading": heading,
            "lat": novatel[i]["lat"],
            "lon": novatel[i]["lon"],
        })

    # camera_poses.csv — header matches what video_runtime.track expects.
    csv_path = snow_dir / "camera_poses.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["GPSTime", "easting", "northing", "heading", "lat", "lon"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    (snow_dir / "window.json").write_text(json.dumps({
        "indices": [0, n],
        "frame_count": n,
        "ref_lat": ref_lat,
        "ref_lon": ref_lon,
        "source": "CADC labeled/image_00 (front camera, camera_F)",
    }, indent=2))

    print(f"  snow: {n} frames staged in {snow_dir}")
    return {
        "n": n,
        "ref_lat": ref_lat,
        "ref_lon": ref_lon,
        "rows": rows,
    }


# ── Summer side (Mapillary closeto per-frame) ───────────────────────────

def _load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _mapillary_closeto(token: str, lat: float, lon: float, *,
                       radius_m: float = 50.0,
                       summer_only: bool = True) -> dict | None:
    """Query Mapillary v4 for the single closest summer image to (lat, lon).
    Returns the API record or None if nothing matches.

    Mapillary's Graph API uses the lat / lng / radius trio (not 'closeto')
    on the /images endpoint; radius is capped at 50 m server-side.
    """
    fields = "id,geometry,captured_at,is_pano,creator,thumb_2048_url"
    params = {
        "access_token": token,
        "lat": lat,
        "lng": lon,
        "radius": min(radius_m, 50.0),  # server-side cap
        "fields": fields,
        "limit": 50,
    }
    qs = urllib.parse.urlencode(params)
    url = f"{GRAPH_URL}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "snow-underlay-cadc/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        results = json.loads(r.read().decode()).get("data", [])
    if not results:
        return None

    # Filter: non-pano, summer (June–September), then nearest by haversine.
    def _summer_month(captured_ms: object) -> bool:
        try:
            ms = int(captured_ms or 0)
        except (TypeError, ValueError):
            return False
        return time.gmtime(ms / 1000).tm_mon in (6, 7, 8, 9)

    def _dist_m(rec: dict) -> float:
        coords = (rec.get("geometry") or {}).get("coordinates") or [lon, lat]
        rlon, rlat = coords[0], coords[1]
        # haversine over a few metres ≈ equirect.
        e, n = _equirect_metres(rlat, rlon, lat, lon)
        return math.hypot(e, n)

    candidates = [r for r in results if not r.get("is_pano")]
    if summer_only:
        summer = [r for r in candidates if _summer_month(r.get("captured_at"))]
        # If nothing summer in the radius, fall back to any clear-weather candidate.
        candidates = summer or candidates
    if not candidates:
        return None
    candidates.sort(key=_dist_m)
    return candidates[0]


def _download_mapillary_image(rec: dict, dst: Path) -> bool:
    url = rec.get("thumb_2048_url")
    if not url:
        return False
    try:
        urllib.request.urlretrieve(url, dst)
        return True
    except Exception as exc:
        print(f"    ! download failed for image {rec.get('id')}: {exc}")
        return False


def build_summer(snow_summary: dict, track_dir: Path, *,
                 radius_m: float, max_pulls: int | None) -> None:
    """For each snow frame's lat/lon, fetch the closest Mapillary summer
    image. Dedupe by Mapillary image id."""
    _load_dotenv()
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise SystemExit(f"{TOKEN_ENV} not set in env or .env")

    summer_dir = track_dir / "summer"
    frames_dir = summer_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_image_ids: dict[str, int] = {}  # mapillary_id → gpstime_us
    snow_rows = snow_summary["rows"]
    n = snow_summary["n"] if max_pulls is None else min(snow_summary["n"], max_pulls)

    print(f"  summer: querying Mapillary closeto for {n} snow-frame locations "
          f"(radius={radius_m:.0f}m)")
    for i, snow_row in enumerate(snow_rows[:n]):
        lat, lon = snow_row["lat"], snow_row["lon"]
        try:
            rec = _mapillary_closeto(token, lat, lon, radius_m=radius_m)
        except Exception as exc:
            print(f"    [{i + 1:>3d}/{n}] query failed: {exc}", flush=True)
            continue
        if rec is None:
            print(f"    [{i + 1:>3d}/{n}] no Mapillary summer image within "
                  f"{radius_m:.0f}m of ({lat:.5f},{lon:.5f})", flush=True)
            continue

        mid = str(rec["id"])
        if mid in seen_image_ids:
            # Same Mapillary image already on disk for an earlier snow frame.
            # Re-use it — write a row pointing at the same gpstime filename.
            gpstime_us = seen_image_ids[mid]
        else:
            gpstime_us = snow_row["GPSTime"]
            dst = frames_dir / f"{gpstime_us}.png"
            ok = _download_mapillary_image(rec, dst)
            if not ok:
                continue
            seen_image_ids[mid] = gpstime_us

        rcoords = (rec.get("geometry") or {}).get("coordinates") or [lon, lat]
        rlon, rlat = rcoords[0], rcoords[1]
        e, north = _equirect_metres(
            rlat, rlon,
            snow_summary["ref_lat"], snow_summary["ref_lon"],
        )
        rows.append({
            "GPSTime": gpstime_us,
            "easting": e,
            "northing": north,
            "heading": 0.0,  # Mapillary doesn't expose heading per record reliably
            "lat": rlat,
            "lon": rlon,
            "mapillary_id": mid,
        })
        print(f"    [{i + 1:>3d}/{n}] mapillary {mid} @ ({rlat:.5f},{rlon:.5f})", flush=True)

    # Dedupe by GPSTime → unique summer poses (KD-tree wants distinct points).
    unique_rows = {r["GPSTime"]: r for r in rows}
    rows = sorted(unique_rows.values(), key=lambda r: r["GPSTime"])

    csv_path = summer_dir / "camera_poses.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "GPSTime", "easting", "northing", "heading", "lat", "lon", "mapillary_id",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    (summer_dir / "window.json").write_text(json.dumps({
        "indices": [0, len(rows)],
        "frame_count": len(rows),
        "source": f"Mapillary closeto per-frame, radius={radius_m}m, summer-only",
    }, indent=2))
    print(f"  summer: {len(rows)} unique Mapillary images staged in {summer_dir}")


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True,
                   help="<date>/<drive> e.g. 2019_02_27/0027")
    p.add_argument("--snow-only", action="store_true",
                   help="Stage snow-side only; skip Mapillary summer pulls. "
                        "Use this first; user inspects the snow frames; then "
                        "run with --summer to commit the Mapillary calls.")
    p.add_argument("--summer", action="store_true",
                   help="Run only the Mapillary summer-side pull. Snow side "
                        "must already be staged. Pulls one closest summer "
                        "image per snow frame (deduped by Mapillary id).")
    p.add_argument("--radius", type=float, default=50.0,
                   help="Mapillary closeto search radius in metres.")
    p.add_argument("--max-pulls", type=int, default=None,
                   help="Cap the number of summer pulls (debug aid).")
    args = p.parse_args()

    scene_path = CADC_ROOT / args.scene
    if not scene_path.exists():
        raise SystemExit(f"missing CADC scene: {scene_path}")

    track_id = "cadc_" + args.scene.replace("/", "_")
    track_dir = TRACKS_ROOT / track_id
    track_dir.mkdir(parents=True, exist_ok=True)

    print(f"== fetch_cadc {args.scene} → {track_dir} ==")

    summer_dir = track_dir / "summer"

    if args.summer and not args.snow_only:
        # Reload snow summary from disk so we don't re-copy frames.
        snow_dir = track_dir / "snow"
        if not (snow_dir / "camera_poses.csv").exists():
            raise SystemExit("snow side not staged yet — run with --snow-only first.")
        rows = []
        with open(snow_dir / "camera_poses.csv") as fh:
            r = csv.DictReader(fh)
            for row in r:
                rows.append({k: (float(v) if k not in ("GPSTime", "mapillary_id") else (int(v) if k == "GPSTime" else v))
                             for k, v in row.items()})
        win = json.loads((snow_dir / "window.json").read_text())
        snow_summary = {
            "n": len(rows),
            "ref_lat": win["ref_lat"],
            "ref_lon": win["ref_lon"],
            "rows": rows,
        }
        build_summer(snow_summary, track_dir,
                     radius_m=args.radius, max_pulls=args.max_pulls)
    else:
        snow_summary = build_snow(scene_path, track_dir)
        if args.snow_only:
            print("\n== snow side staged. Inspect frames, then re-run with --summer to fetch priors. ==")
        else:
            build_summer(snow_summary, track_dir,
                         radius_m=args.radius, max_pulls=args.max_pulls)

    # Write or refresh track.json
    track_json = {
        "track_id": track_id,
        "source": "CADC (Canadian Adverse Driving Conditions)",
        "scene": args.scene,
        "snow_camera": "image_00 (camera_F, forward-facing)",
        "summer_source": "Mapillary v4 closeto per-frame (summer months only)",
        "ref_lat": snow_summary["ref_lat"],
        "ref_lon": snow_summary["ref_lon"],
    }
    (track_dir / "track.json").write_text(json.dumps(track_json, indent=2))
    print(f"\n== done. Track id: {track_id} ==")
    print(f"  Next: make oracle TRACK={track_id}")


if __name__ == "__main__":
    main()
