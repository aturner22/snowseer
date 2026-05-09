"""Fetch the snow + clear image pairs from Mapillary.

Reads `data/demo_pairs.json`, queries each entry's Mapillary image IDs
fresh (URLs are signed and expire), downloads the snow + paired clear-
season images, writes them under `data/pairs/<pair_id>/`.

Usage:
    export MAPILLARY_TOKEN=<token from https://www.mapillary.com/dashboard/developers>
    uv run python -m src.data.fetch_mapillary
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

API = "https://graph.mapillary.com/images"
TOKEN_ENV = "MAPILLARY_TOKEN"

OUT_DIR = Path("data/pairs")
DEMO_PAIRS_PATH = Path(__file__).resolve().parent / "demo_pairs.json"

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


def _heading_delta(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _meters_per_deg_lat() -> float:
    return 111_000.0


def _meters_per_deg_lng(lat: float) -> float:
    return 111_000.0 * float(np.cos(np.radians(lat)))


def _download_image(url: str, path: Path) -> None:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)


def _save_pair(pair_id: str, region: str, snow: ImageMeta,
               priors: list[ImageMeta]) -> Path:
    """Write a pair directory:

        data/pairs/<pair_id>/
            snow.jpg
            clear.jpg               (= priors/00_<id>.jpg, kept as primary)
            priors/
                00_<id>.jpg         (primary prior)
                01_<id>.jpg
                …
            meta.json
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
        dlat_m = (p.lat - snow.lat) * _meters_per_deg_lat()
        dlng_m = (p.lng - snow.lng) * _meters_per_deg_lng(snow.lat)
        d_m = float(np.hypot(dlat_m, dlng_m))
        saved_priors.append({
            **asdict(p),
            "file": str(prior_path.relative_to(pair_dir)),
            "distance_m": round(d_m, 2),
            "heading_delta_deg": round(_heading_delta(snow.heading, p.heading), 2),
        })

    if saved_priors:
        primary_path = pair_dir / saved_priors[0]["file"]
        clear_back = pair_dir / "clear.jpg"
        if not clear_back.exists() and primary_path.exists():
            clear_back.write_bytes(primary_path.read_bytes())

    meta = {
        "region": region,
        "snow": asdict(snow),
        "priors": saved_priors,
        "clear": saved_priors[0] if saved_priors else None,
    }
    (pair_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return pair_dir


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ. Tolerates
    values containing '|' which shell `source` mishandles.
    """
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


def _fetch_selected(token: str) -> int:
    """Fetch every pair declared in `demo_pairs.json`. Idempotent: existing
    files are not re-downloaded.
    """
    if not DEMO_PAIRS_PATH.exists():
        print(f"!! {DEMO_PAIRS_PATH} not found.", file=sys.stderr)
        return 1
    spec = json.loads(DEMO_PAIRS_PATH.read_text())
    pairs = spec.get("pairs", [])
    print(f"[fetch] {len(pairs)} selected pairs", flush=True)
    n_ok = 0

    for entry in tqdm(pairs, desc="pairs"):
        pair_id = entry["pair_id"]
        region = entry["region"]
        snow_id = entry["snow_id"]
        prior_ids = entry.get("prior_ids") or [entry.get("clear_id")]
        prior_ids = [pid for pid in prior_ids if pid]

        snow_raw = _query_image(snow_id, token)
        if snow_raw is None:
            print(f"  ! {pair_id}: could not fetch snow metadata", file=sys.stderr)
            continue
        snow_meta = _to_meta(snow_raw)
        if snow_meta is None:
            print(f"  ! {pair_id}: could not parse snow metadata", file=sys.stderr)
            continue

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
            _save_pair(pair_id, region, snow_meta, prior_metas)
            n_ok += 1
        except Exception as e:
            print(f"  ! {pair_id}: {e}", file=sys.stderr)

    print(f"\nDone. {n_ok}/{len(pairs)} pairs available under {OUT_DIR}/", flush=True)
    return 0


def main() -> None:
    import argparse
    argparse.ArgumentParser(
        description="Fetch the selected snow + clear pairs declared in demo_pairs.json."
    ).parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    _load_dotenv()
    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(f"!! Set {TOKEN_ENV} in your environment or in .env.", file=sys.stderr)
        print("   Get one at https://www.mapillary.com/dashboard/developers", file=sys.stderr)
        sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.exit(_fetch_selected(token))


if __name__ == "__main__":
    main()
