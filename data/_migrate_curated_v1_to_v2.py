"""One-shot migration: data/curated_pairs.json v1 -> v2.

v1 entries have a single `clear_id`. v2 entries have `prior_ids: [str, ...]`,
adaptive 1-5 priors each, sourced by querying Mapillary for additional summer
neighbours within 30 m / 30° of the snow image. The original `clear_id`
becomes the *primary* (idx 0) prior; up to 4 additional priors are appended.

Usage:
    export MAPILLARY_TOKEN=<token>
    uv run python -m data._migrate_curated_v1_to_v2

Idempotent: if the input is already v2, the script reports and exits.
The new file is written to data/curated_pairs.json (overwriting v1).
A backup of the input is written to data/curated_pairs.v1.json.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

CURATED = Path("data/curated_pairs.json")
CURATED_V1_BACKUP = Path("data/curated_pairs.v1.json")
PAIRS_DIR = Path("data/pairs")

API = "https://graph.mapillary.com/images"
TOKEN_ENV = "MAPILLARY_TOKEN"

# Adaptive K: take up to MAX_PRIORS within these tolerances.
MAX_PRIORS = 5
DIST_M_THRESH = 30.0
HEADING_DEG_THRESH = 30.0
SUMMER_MONTHS = {5, 6, 7, 8, 9}

FIELDS = [
    "id", "geometry", "captured_at", "compass_angle", "computed_compass_angle",
    "thumb_1024_url", "thumb_2048_url", "is_pano", "width", "height", "sequence",
]


def _heading_delta(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _meters_per_deg_lat() -> float:
    return 111_000.0


def _meters_per_deg_lng(lat: float) -> float:
    return 111_000.0 * float(np.cos(np.radians(lat)))


def _bbox(lat: float, lng: float, radius_m: float) -> tuple[float, float, float, float]:
    dlat = radius_m / _meters_per_deg_lat()
    dlng = radius_m / _meters_per_deg_lng(lat)
    return (lng - dlng, lat - dlat, lng + dlng, lat + dlat)


def _query_summer_neighbours(
    snow_lat: float, snow_lng: float, snow_heading: float, token: str,
) -> list[dict]:
    """Query Mapillary for summer images within 30 m + 30° of the snow image."""
    bbox = _bbox(snow_lat, snow_lng, DIST_M_THRESH * 1.5)  # slight pad for the bbox query
    candidates: list[dict] = []
    # Scan a generous span of years; each year is one API call.
    for year in range(2017, 2027):
        # Summer span: May-Sep
        start = datetime(year, 5, 1, tzinfo=timezone.utc)
        end = datetime(year, 10, 1, tzinfo=timezone.utc)
        params = {
            "fields": ",".join(FIELDS),
            "bbox": ",".join(f"{x:.6f}" for x in bbox),
            "start_captured_at": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_captured_at": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": "2000",
        }
        try:
            r = requests.get(API, headers={"Authorization": f"OAuth {token}"},
                             params=params, timeout=60)
            r.raise_for_status()
            for raw in r.json().get("data", []):
                geom = raw.get("geometry")
                if not geom or geom.get("type") != "Point":
                    continue
                lng, lat = geom["coordinates"]
                heading = raw.get("computed_compass_angle") or raw.get("compass_angle")
                if heading is None:
                    continue
                if raw.get("is_pano"):
                    continue
                # Distance
                dlat_m = (float(lat) - snow_lat) * _meters_per_deg_lat()
                dlng_m = (float(lng) - snow_lng) * _meters_per_deg_lng(snow_lat)
                dist_m = float(np.hypot(dlat_m, dlng_m))
                if dist_m > DIST_M_THRESH:
                    continue
                # Heading
                hdelta = _heading_delta(snow_heading, float(heading) % 360.0)
                if hdelta > HEADING_DEG_THRESH:
                    continue
                candidates.append({
                    "id": str(raw["id"]),
                    "lat": float(lat),
                    "lng": float(lng),
                    "heading": float(heading) % 360.0,
                    "captured_at": int(raw["captured_at"]),
                    "distance_m": round(dist_m, 2),
                    "heading_delta_deg": round(hdelta, 2),
                })
        except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as e:
            print(f"    ! year {year}: {type(e).__name__}", file=sys.stderr)
            continue
        time.sleep(0.05)
    # Dedupe by id
    seen: set[str] = set()
    deduped = []
    for c in candidates:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        deduped.append(c)
    return deduped


def _rank_priors(neighbours: list[dict], primary_id: str) -> list[dict]:
    """Order priors with the primary (existing clear_id) first, then by closeness."""
    primary = next((n for n in neighbours if n["id"] == primary_id), None)
    others = sorted(
        (n for n in neighbours if n["id"] != primary_id),
        key=lambda n: (n["distance_m"], n["heading_delta_deg"]),
    )
    out = []
    if primary is not None:
        out.append(primary)
    out.extend(others)
    return out[:MAX_PRIORS]


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    _load_dotenv()
    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(f"!! Set {TOKEN_ENV} in your environment or .env.", file=sys.stderr)
        sys.exit(2)

    if not CURATED.exists():
        print(f"!! {CURATED} not found.", file=sys.stderr)
        sys.exit(2)

    spec = json.loads(CURATED.read_text())
    if spec.get("version", "").startswith("v2"):
        print(f"{CURATED} is already v2 ({spec.get('version')}). Nothing to do.")
        return

    # Backup v1
    if not CURATED_V1_BACKUP.exists():
        shutil.copy2(CURATED, CURATED_V1_BACKUP)
        print(f"backed up v1 to {CURATED_V1_BACKUP}")

    pairs_v1 = spec.get("pairs", [])
    print(f"migrating {len(pairs_v1)} pairs to v2 ...")

    pairs_v2 = []
    for entry in pairs_v1:
        pid = entry["pair_id"]
        snow_id = entry["snow_id"]
        clear_id = entry["clear_id"]
        # Pull snow lat/lng/heading from local meta.json
        meta_path = PAIRS_DIR / pid / "meta.json"
        if not meta_path.exists():
            print(f"  ! {pid}: missing meta.json; skipping", file=sys.stderr)
            continue
        m = json.loads(meta_path.read_text())
        snow = m.get("snow", {})
        snow_lat, snow_lng, snow_h = snow.get("lat"), snow.get("lng"), snow.get("heading")
        if snow_lat is None or snow_lng is None or snow_h is None:
            print(f"  ! {pid}: snow geometry missing; skipping", file=sys.stderr)
            continue
        print(f"  {pid[:60]}  querying neighbours …")
        neighbours = _query_summer_neighbours(snow_lat, snow_lng, float(snow_h), token)
        priors = _rank_priors(neighbours, clear_id)
        if not priors:
            # Fall back to using the existing clear (we know it works because it's curated).
            existing_clear = m.get("clear", {})
            priors = [{
                "id": clear_id,
                "lat": existing_clear.get("lng", snow_lng),  # safe defaults
                "lng": existing_clear.get("lng", snow_lng),
                "heading": float(existing_clear.get("heading", snow_h)),
                "captured_at": int(existing_clear.get("captured_at", 0)),
                "distance_m": float(entry.get("distance_m", 0.0)),
                "heading_delta_deg": float(entry.get("heading_delta_deg", 0.0)),
            }]
        # Build v2 entry — drop clear_id, add prior_ids list.
        v2 = {
            "pair_id": pid,
            "region": entry.get("region"),
            "rating": entry.get("rating"),
            "snow_id": snow_id,
            "prior_ids": [p["id"] for p in priors],
            "priors": priors,
            "place": entry.get("place"),
            "condition": entry.get("condition"),
            "snow_captured": entry.get("snow_captured"),
            "snow_captured_at": entry.get("snow_captured_at"),
            # Drop clear_id, clear_captured, clear_captured_at — they're now
            # part of the priors list (priors[0] is the canonical primary).
        }
        pairs_v2.append(v2)
        print(f"    -> {len(priors)} priors")

    out = {
        "version": "v2.0",
        "description": (
            "Demo set with adaptive 1-5 clear-season priors per snow query, "
            "fused via mask voting. priors[0] is the canonical primary (the v1 clear_id)."
        ),
        "fusion": {
            "max_priors": MAX_PRIORS,
            "search_radius_m": DIST_M_THRESH,
            "search_heading_deg": HEADING_DEG_THRESH,
        },
        "pairs": pairs_v2,
    }
    CURATED.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {CURATED} (v2.0, {len(pairs_v2)} pairs)")


if __name__ == "__main__":
    main()
