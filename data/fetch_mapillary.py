"""Fetch paired snowy/clear image pairs from Mapillary.

Two modes:
  - `--curated-only` (default for `make stills`): pull *only* the pairs
    declared in `data/demo_pairs.json` by their image IDs. This is the
    canonical reproducibility path; produces the exact demo set on a fresh
    clone.
  - exploration mode (no flag): query each REGIONS entry by bbox+date and
    pair winter+summer images. Use this when looking for additional
    candidate pairs; the output goes into `data/pairs/` for offline review.

Usage:
    export MAPILLARY_TOKEN=<your token from https://www.mapillary.com/dashboard/developers>
    uv run python -m data.fetch_mapillary --curated-only      # the demo set
    uv run python -m data.fetch_mapillary                     # exploration

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
    # Cold-climate cities with both winter and summer Mapillary uploads.
    # Geographic restriction has been deliberately broadened — quality of the
    # snow query frame matters more than place. Mapillary bbox max is 0.01
    # sq degrees, comfortably above any radius below ~2.5 km at these latitudes.
    {"name": "kiruna_se", "lat": 67.8558, "lng": 20.2253, "radius_m": 600},
    {"name": "rovaniemi_fi", "lat": 66.5039, "lng": 25.7294, "radius_m": 600},
    {"name": "ostersund_e45_se", "lat": 63.8512, "lng": 15.5543, "radius_m": 1500},
    {"name": "tromso_no", "lat": 69.6489, "lng": 18.9551, "radius_m": 1500},
    {"name": "lulea_se", "lat": 65.5848, "lng": 22.1547, "radius_m": 1500},
    {"name": "gallivare_se", "lat": 67.1354, "lng": 20.6543, "radius_m": 1500},
    {"name": "trondheim_no", "lat": 63.4305, "lng": 10.3951, "radius_m": 1500},
    {"name": "bodo_no", "lat": 67.2804, "lng": 14.4049, "radius_m": 1500},
    {"name": "stockholm_se", "lat": 59.3293, "lng": 18.0686, "radius_m": 1500},
    {"name": "helsinki_fi", "lat": 60.1699, "lng": 24.9384, "radius_m": 1500},
    {"name": "oslo_no", "lat": 59.9139, "lng": 10.7522, "radius_m": 1500},
    {"name": "reykjavik_is", "lat": 64.1466, "lng": -21.9426, "radius_m": 1500},
    {"name": "boston_us", "lat": 42.3601, "lng": -71.0589, "radius_m": 1500},
    {"name": "minneapolis_us", "lat": 44.9778, "lng": -93.2650, "radius_m": 1500},
    {"name": "buffalo_us", "lat": 42.8864, "lng": -78.8784, "radius_m": 1500},
    {"name": "quebec_ca", "lat": 46.8139, "lng": -71.2080, "radius_m": 1500},
    {"name": "edmonton_ca", "lat": 53.5461, "lng": -113.4938, "radius_m": 1500},
    {"name": "calgary_ca", "lat": 51.0447, "lng": -114.0719, "radius_m": 1500},
    {"name": "sapporo_jp", "lat": 43.0618, "lng": 141.3545, "radius_m": 1500},
    {"name": "innsbruck_at", "lat": 47.2692, "lng": 11.4041, "radius_m": 1500},
    {"name": "vladivostok_ru", "lat": 43.1198, "lng": 131.8869, "radius_m": 1500},
]

WINTER_MONTHS = {12, 1, 2, 3}
SUMMER_MONTHS = {5, 6, 7, 8, 9}
YEARS = (2018, 2026)  # inclusive range to scan

DIST_M_THRESH = 5.0
HEADING_DEG_THRESH = 20.0
TARGET_PAIRS_PER_REGION = 20

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


def _query_image(image_id: str, token: str) -> dict | None:
    """Fetch a single Mapillary image's metadata + thumbnail URL by ID."""
    url = f"https://graph.mapillary.com/{image_id}"
    params = {"fields": ",".join(FIELDS)}
    try:
        r = requests.get(
            url, headers={"Authorization": f"OAuth {token}"}, params=params, timeout=60
        )
        r.raise_for_status()
        return r.json()
    except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as e:
        print(f"  ! image {image_id}: {type(e).__name__}", file=sys.stderr)
        return None


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
    # Skip download if we've already pulled this pair (allows re-running the
    # fetcher to pick up new regions without re-downloading existing pairs).
    if (pair_dir / "snow.jpg").exists() and (pair_dir / "clear.jpg").exists() and (pair_dir / "meta.json").exists():
        return pair_dir
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


DEMO_PAIRS_PATH = Path("data/demo_pairs.json")


def _save_multi_prior_pair(pair_id: str, region: str, snow: ImageMeta,
                            priors: list[ImageMeta]) -> Path:
    """Multi-prior layout:
        data/pairs/<pair_id>/
            snow.jpg
            clear.jpg               (= priors/00_<id>.jpg, kept for back-compat)
            priors/
                00_<id>.jpg         (primary prior — matches the v1 clear_id)
                01_<id>.jpg
                …
            meta.json               (with `priors: [...]`)
    """
    pair_dir = OUT_DIR / pair_id
    priors_dir = pair_dir / "priors"
    pair_dir.mkdir(parents=True, exist_ok=True)
    priors_dir.mkdir(parents=True, exist_ok=True)

    snow_path = pair_dir / "snow.jpg"
    if not snow_path.exists():
        _download_image(snow.thumb_url, snow_path)

    saved_priors: list[dict] = []
    for idx, p in enumerate(priors):
        prior_path = priors_dir / f"{idx:02d}_{p.id}.jpg"
        if not prior_path.exists():
            try:
                _download_image(p.thumb_url, prior_path)
            except Exception as e:
                print(f"  ! prior {p.id}: {e}", file=sys.stderr)
                continue
        # Distances + heading delta for record.
        dlat_m = (p.lat - snow.lat) * _meters_per_deg_lat()
        dlng_m = (p.lng - snow.lng) * _meters_per_deg_lng(snow.lat)
        d_m = float(np.hypot(dlat_m, dlng_m))
        saved_priors.append({
            **asdict(p),
            "file": str(prior_path.relative_to(pair_dir)),
            "distance_m": round(d_m, 2),
            "heading_delta_deg": round(_heading_delta(snow.heading, p.heading), 2),
        })

    # Back-compat: copy primary to clear.jpg (used by older code paths)
    if saved_priors:
        primary_path = pair_dir / saved_priors[0]["file"]
        clear_back = pair_dir / "clear.jpg"
        if not clear_back.exists() and primary_path.exists():
            clear_back.write_bytes(primary_path.read_bytes())

    meta = {
        "region": region,
        "snow": asdict(snow),
        "priors": saved_priors,
        # Back-compat: meta.clear == priors[0]
        "clear": saved_priors[0] if saved_priors else None,
    }
    (pair_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return pair_dir


def _fetch_curated(token: str) -> int:
    """Fetch only the pairs declared in data/demo_pairs.json. Each entry's
    Mapillary IDs are queried fresh (URLs are signed and expire), so this
    works on a clean clone.

    Supports both v1 (single clear_id) and v2 (prior_ids list) schemas.
    """
    if not DEMO_PAIRS_PATH.exists():
        print(f"!! {DEMO_PAIRS_PATH} not found.", file=sys.stderr)
        return 1
    spec = json.loads(DEMO_PAIRS_PATH.read_text())
    pairs = spec.get("pairs", [])
    schema_v2 = spec.get("version", "").startswith("v2")
    print(f"[curated] {len(pairs)} pairs ({'v2 multi-prior' if schema_v2 else 'v1 single-prior'})")
    n_ok = 0

    for entry in tqdm(pairs, desc="curated"):
        pair_id = entry["pair_id"]
        region = entry["region"]
        snow_id = entry["snow_id"]
        prior_ids = entry.get("prior_ids") or [entry.get("clear_id")]
        prior_ids = [pid for pid in prior_ids if pid]

        # Snow image metadata
        snow_raw = _query_image(snow_id, token)
        if snow_raw is None:
            print(f"  ! {pair_id}: could not fetch snow metadata", file=sys.stderr)
            continue
        snow_meta = _to_meta(snow_raw)
        if snow_meta is None:
            print(f"  ! {pair_id}: could not parse snow metadata", file=sys.stderr)
            continue

        # Prior metadata for each id
        prior_metas: list[ImageMeta] = []
        for pid in prior_ids:
            p_raw = _query_image(pid, token)
            if p_raw is None:
                continue
            p_meta = _to_meta(p_raw)
            if p_meta is not None:
                prior_metas.append(p_meta)
        if not prior_metas:
            print(f"  ! {pair_id}: no priors fetchable; skipping", file=sys.stderr)
            continue

        try:
            _save_multi_prior_pair(pair_id, region, snow_meta, prior_metas)
            n_ok += 1
        except Exception as e:
            print(f"  ! {pair_id}: {e}", file=sys.stderr)

    print(f"\nDone. {n_ok}/{len(pairs)} curated pairs available under {OUT_DIR}/")
    return 0


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated-only", action="store_true",
                    help="Pull only pairs listed in data/demo_pairs.json (the demo set). "
                         "Default behaviour explores all REGIONS by bbox+date.")
    args = ap.parse_args()

    _load_dotenv()
    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(f"!! Set {TOKEN_ENV} in your environment or in .env.", file=sys.stderr)
        print("   Get one at https://www.mapillary.com/dashboard/developers", file=sys.stderr)
        sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.curated_only:
        sys.exit(_fetch_curated(token))

    summary: list[dict] = []

    for region in REGIONS:
        existing = sum(
            1 for d in OUT_DIR.iterdir()
            if d.is_dir() and d.name.startswith(region["name"] + "__")
        ) if OUT_DIR.exists() else 0
        if existing >= TARGET_PAIRS_PER_REGION:
            print(f"\n[{region['name']}] skipping — already have {existing} pairs")
            continue
        print(f"\n[{region['name']}] querying... (have {existing}, target {TARGET_PAIRS_PER_REGION})")
        winter = _collect_images(region, token, WINTER_MONTHS)
        summer = _collect_images(region, token, SUMMER_MONTHS)
        print(f"  winter candidates: {len(winter)}   summer candidates: {len(summer)}")
        pairs = _pair_winter_to_summer(winter, summer)
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
