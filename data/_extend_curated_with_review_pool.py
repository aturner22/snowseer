"""Extend data/curated_pairs.json to include the wider 'review pool' —
all manually-snow-accepted pairs that aren't already in the demo set.

Used once after the multi-prior migration so we can re-evaluate the 13
pairs that were rated NOT_GOOD/AWFUL in v1.1 to see whether multi-prior
fusion promotes any of them to GREAT/OKAY.

Adds entries with `rating="review_pool"` so they're distinguishable from
the canonical demo. The pipeline's `_load_curated_pair_ids` already
returns all pair_ids in the file, so they get processed alongside the
demo set on the next pipeline run.

Usage:
    export MAPILLARY_TOKEN=<token>
    uv run python -m data._extend_curated_with_review_pool
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

CURATED = Path("data/curated_pairs.json")
PAIRS_DIR = Path("data/pairs")
SNOW_CURATION = Path("data/manual_snow_curation.json")

API = "https://graph.mapillary.com/images"
TOKEN_ENV = "MAPILLARY_TOKEN"
MAX_PRIORS = 5
DIST_M_THRESH = 30.0
HEADING_DEG_THRESH = 30.0

FIELDS = [
    "id", "geometry", "captured_at", "compass_angle", "computed_compass_angle",
    "thumb_1024_url", "thumb_2048_url", "is_pano", "width", "height", "sequence",
]

PLACE = {
    "gallivare_se": "Gällivare, Sweden",
    "kiruna_se": "Kiruna, Sweden",
    "lulea_se": "Luleå, Sweden",
    "rovaniemi_fi": "Rovaniemi, Finland",
    "ostersund_e45_se": "Östersund, Sweden",
    "tromso_no": "Tromsø, Norway",
    "bodo_no": "Bodø, Norway",
}


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


def _query_summer_neighbours(snow_lat: float, snow_lng: float, snow_h: float, token: str) -> list[dict]:
    bbox = _bbox(snow_lat, snow_lng, DIST_M_THRESH * 1.5)
    candidates: list[dict] = []
    for year in range(2017, 2027):
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
                if heading is None or raw.get("is_pano"):
                    continue
                dlat_m = (float(lat) - snow_lat) * _meters_per_deg_lat()
                dlng_m = (float(lng) - snow_lng) * _meters_per_deg_lng(snow_lat)
                dist_m = float(np.hypot(dlat_m, dlng_m))
                if dist_m > DIST_M_THRESH:
                    continue
                hdelta = _heading_delta(snow_h, float(heading) % 360.0)
                if hdelta > HEADING_DEG_THRESH:
                    continue
                candidates.append({
                    "id": str(raw["id"]),
                    "lat": float(lat), "lng": float(lng),
                    "heading": float(heading) % 360.0,
                    "captured_at": int(raw["captured_at"]),
                    "distance_m": round(dist_m, 2),
                    "heading_delta_deg": round(hdelta, 2),
                })
        except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as e:
            print(f"    ! year {year}: {type(e).__name__}", file=sys.stderr)
            continue
        time.sleep(0.05)
    seen: set[str] = set()
    deduped = []
    for c in candidates:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        deduped.append(c)
    return deduped


def _rank(neighbours: list[dict], primary_id: str) -> list[dict]:
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
        print(f"!! Set {TOKEN_ENV}", file=sys.stderr)
        sys.exit(2)

    spec = json.loads(CURATED.read_text())
    if not spec.get("version", "").startswith("v2"):
        print("!! curated_pairs.json is not v2; run _migrate_curated_v1_to_v2 first", file=sys.stderr)
        sys.exit(2)
    existing_ids = {p["pair_id"] for p in spec.get("pairs", [])}

    snow = json.loads(SNOW_CURATION.read_text())
    accepted_ids = [k for k, v in snow.items() if v.get("verdict") == "accept"]
    extras = [pid for pid in accepted_ids if pid not in existing_ids]
    print(f"adding {len(extras)} review-pool pairs to {CURATED}")

    new_entries = []
    for pid in extras:
        meta_path = PAIRS_DIR / pid / "meta.json"
        if not meta_path.exists():
            print(f"  ! {pid}: no meta.json", file=sys.stderr)
            continue
        m = json.loads(meta_path.read_text())
        snow_obj = m.get("snow") or {}
        snow_lat, snow_lng, snow_h = snow_obj.get("lat"), snow_obj.get("lng"), snow_obj.get("heading")
        if None in (snow_lat, snow_lng, snow_h):
            print(f"  ! {pid}: snow geometry missing", file=sys.stderr)
            continue
        clear_obj = m.get("clear") or {}
        primary_clear_id = str(clear_obj.get("id", ""))

        # Region from pair_id (e.g. 'gallivare_se__...')
        region = pid.split("__", 1)[0]

        print(f"  {pid[:60]}  querying neighbours …")
        neighbours = _query_summer_neighbours(snow_lat, snow_lng, float(snow_h), token)
        priors = _rank(neighbours, primary_clear_id)
        if not priors:
            # Fall back: use the existing clear as the only prior.
            priors = [{
                "id": primary_clear_id,
                "lat": clear_obj.get("lat", snow_lat),
                "lng": clear_obj.get("lng", snow_lng),
                "heading": float(clear_obj.get("heading", snow_h)),
                "captured_at": int(clear_obj.get("captured_at", 0)),
                "distance_m": float(m.get("distance_m", 0)),
                "heading_delta_deg": float(m.get("heading_delta_deg", 0)),
            }]
        print(f"    -> {len(priors)} priors")

        snow_t = ""
        if snow_obj.get("captured_at"):
            snow_t = datetime.fromtimestamp(snow_obj["captured_at"]/1000, tz=timezone.utc).strftime("%B %Y")

        new_entries.append({
            "pair_id": pid,
            "region": region,
            "rating": "review_pool",
            "snow_id": str(snow_obj.get("id", "")),
            "prior_ids": [p["id"] for p in priors],
            "priors": priors,
            "place": PLACE.get(region, region),
            "condition": "(review pool — not yet rated under multi-prior)",
            "snow_captured": snow_t,
            "snow_captured_at": int(snow_obj.get("captured_at", 0)),
        })

    spec["pairs"].extend(new_entries)
    spec["description"] = (
        "Demo set with adaptive 1-5 clear-season priors per snow query, "
        "fused via mask voting. priors[0] is the canonical primary. "
        "Entries with rating='review_pool' are extra pairs (snow-accepted, "
        "not in the GREAT+OKAY demo) included for re-evaluation under "
        "multi-prior; demote them once rated."
    )
    CURATED.write_text(json.dumps(spec, indent=2))
    print(f"\nwrote {CURATED} ({len(spec['pairs'])} pairs total)")


if __name__ == "__main__":
    main()
