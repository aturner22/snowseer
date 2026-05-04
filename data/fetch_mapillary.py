"""Fetch paired snowy/clear image pairs from Mapillary.

Usage:
    export MAPILLARY_TOKEN=<your token from https://www.mapillary.com/dashboard/developers>
    uv run python -m data.fetch_mapillary

Pipeline:
    1. For each region in REGIONS, query Mapillary for images captured in:
        - winter months (`WINTER_MONTHS`)  -> "snow candidates"
        - summer months (`SUMMER_MONTHS`)  -> "clear candidates"
    2. For each winter image, find the nearest summer image by lat/lng/heading.
    3. Save matched pairs with their thumbnails to data/pairs/<id>/.

Snow appears only at INFERENCE TIME as the runtime input. No model weights are
fine-tuned on snowy images anywhere in this codebase.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from sklearn.neighbors import BallTree
from tqdm import tqdm

API = "https://graph.mapillary.com/images"
TOKEN_ENV = "MAPILLARY_TOKEN"

REGIONS: list[dict] = [
    # Reliably snowy in DJF, drivable streets. Widened bboxes for under-covered
    # regions; do NOT relax the per-pair distance/heading thresholds — those
    # are not the bottleneck (audit confirmed). Mapillary's bbox max is 0.01
    # sq degrees, comfortably above any radius below ~2.5 km at these latitudes.
    {"name": "kiruna_se", "lat": 67.8558, "lng": 20.2253, "radius_m": 600},
    {"name": "rovaniemi_fi", "lat": 66.5039, "lng": 25.7294, "radius_m": 600},
    {"name": "ostersund_se", "lat": 63.1792, "lng": 14.6357, "radius_m": 1500},
    {"name": "ostersund_e45_se", "lat": 63.8512, "lng": 15.5543, "radius_m": 1500},  # user's flagship
    {"name": "tromso_no", "lat": 69.6489, "lng": 18.9551, "radius_m": 1500},
    {"name": "anchorage_ak", "lat": 61.2181, "lng": -149.9003, "radius_m": 1500},
    {"name": "sundsvall_se", "lat": 62.3908, "lng": 17.3069, "radius_m": 1500},
    {"name": "gallivare_se", "lat": 67.1354, "lng": 20.6543, "radius_m": 1500},
    {"name": "lulea_se", "lat": 65.5848, "lng": 22.1547, "radius_m": 1500},
    {"name": "trondheim_no", "lat": 63.4305, "lng": 10.3951, "radius_m": 1500},
    {"name": "bodo_no", "lat": 67.2804, "lng": 14.4049, "radius_m": 1500},
]

WINTER_MONTHS = {12, 1, 2, 3}
SUMMER_MONTHS = {5, 6, 7, 8, 9}
YEARS = (2018, 2026)  # inclusive range to scan

DIST_M_THRESH = 5.0
HEADING_DEG_THRESH = 20.0
TARGET_PAIRS_PER_REGION = 12

OUT_DIR = Path("data/pairs")

FIELDS = [
    "id",
    "geometry",
    "captured_at",
    "compass_angle",
    "computed_compass_angle",
    "thumb_1024_url",
    "thumb_2048_url",
    "is_pano",
    "width",
    "height",
    "sequence",
]


@dataclass
class ImageMeta:
    id: str
    lng: float
    lat: float
    heading: float  # degrees, [0, 360)
    captured_at: int  # ms since epoch
    thumb_url: str
    is_pano: bool

    @property
    def captured_dt(self) -> datetime:
        return datetime.fromtimestamp(self.captured_at / 1000.0, tz=timezone.utc)

    @property
    def month(self) -> int:
        return self.captured_dt.month


def _bbox_for_region(lat: float, lng: float, radius_m: float) -> tuple[float, float, float, float]:
    """Approximate bbox of side ~2*radius_m around (lat, lng).

    Mapillary requires bbox area <= 0.01 sq degrees. At lat 65, 1 deg lat ~111km,
    1 deg lng ~47km, so 0.01 sq deg ~ 5km x 2km area. 600m radius is comfortably under.
    """
    dlat = radius_m / 111_000.0
    dlng = radius_m / (111_000.0 * np.cos(np.radians(lat)))
    return (lng - dlng, lat - dlat, lng + dlng, lat + dlat)


def _query_bbox(
    bbox: tuple[float, float, float, float],
    token: str,
    start: datetime,
    end: datetime,
    *,
    limit: int = 2000,
) -> list[dict]:
    params = {
        "fields": ",".join(FIELDS),
        "bbox": ",".join(f"{x:.6f}" for x in bbox),
        "start_captured_at": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_captured_at": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": str(limit),
    }
    r = requests.get(
        API, headers={"Authorization": f"OAuth {token}"}, params=params, timeout=60
    )
    r.raise_for_status()
    return r.json().get("data", [])


def _to_meta(raw: dict) -> ImageMeta | None:
    geom = raw.get("geometry")
    if not geom or geom.get("type") != "Point":
        return None
    lng, lat = geom["coordinates"]
    heading = raw.get("computed_compass_angle")
    if heading is None:
        heading = raw.get("compass_angle")
    if heading is None:
        return None
    thumb = raw.get("thumb_2048_url") or raw.get("thumb_1024_url")
    if not thumb:
        return None
    return ImageMeta(
        id=str(raw["id"]),
        lng=float(lng),
        lat=float(lat),
        heading=float(heading) % 360.0,
        captured_at=int(raw["captured_at"]),
        thumb_url=thumb,
        is_pano=bool(raw.get("is_pano", False)),
    )


def _collect_images(
    region: dict, token: str, months_filter: set[int]
) -> list[ImageMeta]:
    bbox = _bbox_for_region(region["lat"], region["lng"], region["radius_m"])
    out: list[ImageMeta] = []
    for year in range(YEARS[0], YEARS[1] + 1):
        # Query each year as one window to stay under the 2000-image cap.
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        try:
            raw = _query_bbox(bbox, token, start, end)
        except requests.HTTPError as e:
            print(f"  ! HTTP {e.response.status_code} for {region['name']} year {year}", file=sys.stderr)
            continue
        except (requests.Timeout, requests.ConnectionError) as e:
            print(f"  ! network error for {region['name']} year {year}: {type(e).__name__}", file=sys.stderr)
            continue
        for r in raw:
            meta = _to_meta(r)
            if meta is None:
                continue
            if meta.is_pano:
                continue
            if meta.month not in months_filter:
                continue
            out.append(meta)
        time.sleep(0.05)  # be polite under the 10k/min search limit
    return out


def _heading_delta(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _meters_per_deg_lat() -> float:
    return 111_000.0


def _meters_per_deg_lng(lat: float) -> float:
    return 111_000.0 * float(np.cos(np.radians(lat)))


def _pair_winter_to_summer(
    winter: list[ImageMeta], summer: list[ImageMeta]
) -> list[tuple[ImageMeta, ImageMeta, float]]:
    """For each winter image, find the closest summer image within DIST_M_THRESH and HEADING_DEG_THRESH.

    Returns list of (winter, summer, distance_m).
    """
    if not winter or not summer:
        return []
    summer_arr = np.array([[s.lat, s.lng] for s in summer])
    # BallTree with haversine expects radians
    tree = BallTree(np.radians(summer_arr), metric="haversine")
    earth_r_m = 6_371_008.8
    radius_rad = DIST_M_THRESH / earth_r_m

    pairs: list[tuple[ImageMeta, ImageMeta, float]] = []
    seen_summer: set[str] = set()
    for w in winter:
        q = np.radians([[w.lat, w.lng]])
        idxs = tree.query_radius(q, r=radius_rad)[0]
        if len(idxs) == 0:
            continue
        # Pick the nearest summer image (heading-compatible) not yet used.
        best: tuple[float, ImageMeta] | None = None
        for i in idxs:
            s = summer[int(i)]
            if s.id in seen_summer:
                continue
            if _heading_delta(w.heading, s.heading) > HEADING_DEG_THRESH:
                continue
            # haversine distance in metres
            dlat_m = (s.lat - w.lat) * _meters_per_deg_lat()
            dlng_m = (s.lng - w.lng) * _meters_per_deg_lng(w.lat)
            dm = float(np.hypot(dlat_m, dlng_m))
            if best is None or dm < best[0]:
                best = (dm, s)
        if best is not None:
            pairs.append((w, best[1], best[0]))
            seen_summer.add(best[1].id)
    return pairs


def _download_image(url: str, path: Path) -> None:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)


def _save_pair(
    region_name: str, winter: ImageMeta, summer: ImageMeta, dist_m: float
) -> Path:
    pair_dir = OUT_DIR / f"{region_name}__{winter.id}__{summer.id}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    _download_image(winter.thumb_url, pair_dir / "snow.jpg")
    _download_image(summer.thumb_url, pair_dir / "clear.jpg")
    meta = {
        "region": region_name,
        "distance_m": round(dist_m, 2),
        "heading_delta_deg": round(_heading_delta(winter.heading, summer.heading), 2),
        "snow": asdict(winter),
        "clear": asdict(summer),
    }
    (pair_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return pair_dir


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny .env loader. Tolerates values containing '|' which shell `source` mishandles."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def main() -> None:
    _load_dotenv()
    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(f"!! Set {TOKEN_ENV} in your environment or in .env.", file=sys.stderr)
        print("   Get one at https://www.mapillary.com/dashboard/developers", file=sys.stderr)
        sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for region in REGIONS:
        print(f"\n[{region['name']}] querying...")
        winter = _collect_images(region, token, WINTER_MONTHS)
        summer = _collect_images(region, token, SUMMER_MONTHS)
        print(f"  winter candidates: {len(winter)}   summer candidates: {len(summer)}")
        pairs = _pair_winter_to_summer(winter, summer)
        # Rank by distance (tighter pairs first), keep the top K
        pairs.sort(key=lambda t: t[2])
        pairs = pairs[:TARGET_PAIRS_PER_REGION]
        print(f"  matched pairs (top {TARGET_PAIRS_PER_REGION}): {len(pairs)}")
        for w, s, d in tqdm(pairs, desc=f"  downloading {region['name']}"):
            try:
                pair_dir = _save_pair(region["name"], w, s, d)
                summary.append({"pair_dir": str(pair_dir), "distance_m": d})
            except Exception as e:
                print(f"    skip ({w.id} <-> {s.id}): {e}", file=sys.stderr)

    (OUT_DIR / "index.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. {len(summary)} pairs saved under {OUT_DIR}/")


if __name__ == "__main__":
    main()
