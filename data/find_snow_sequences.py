"""Mapillary winter-sequence reconnaissance using the Graph API.

Scans pre-defined city bboxes within a winter date window and ranks
candidate sequences by:
  - frame count in the bbox (longer = more demo material)
  - non-pano (we want flat dashcam/roof, not 360°)
  - cadence (median time between frames; ideally 1–4 s)
  - existence of a clear-season counterpart sequence at the same coords

Usage:
    uv run python -m data.find_snow_sequences --city tromso
    uv run python -m data.find_snow_sequences --city all

Outputs a ranked CSV under data/video/recon/<city>__<date_range>.csv plus
a summary JSON.

Requires MAPILLARY_TOKEN in env (already used by the existing fetcher).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/video/recon"

GRAPH_URL = "https://graph.mapillary.com/images"


# Bboxes (W, S, E, N) for ~1 km winter snow city centres.
CITIES = {
    "tromso":         (18.93, 69.64, 19.00, 69.68),     # NO
    "reykjavik":      (-21.96, 64.13, -21.91, 64.16),   # IS
    "bergen":         ( 5.31, 60.39,  5.34, 60.41),     # NO
    "quebec":         (-71.21, 46.80, -71.20, 46.82),   # CA
    "innsbruck":      (11.38, 47.26, 11.41, 47.28),     # AT
    "sapporo":        (141.34, 43.05, 141.36, 43.07),   # JP
    "kiruna":         (20.21, 67.85, 20.24, 67.87),     # SE
    # broader sweep (added 2026-05)
    "trondheim":      (10.39, 63.42, 10.43, 63.44),     # NO
    "helsinki":       (24.93, 60.16, 24.97, 60.18),     # FI
    "anchorage":      (-149.92, 61.20, -149.86, 61.23), # US-AK
    "saskatoon":      (-106.68, 52.12, -106.64, 52.14), # CA
    "winnipeg":       (-97.16, 49.88, -97.12, 49.90),   # CA
    "longyearbyen":   (15.62, 78.21, 15.66, 78.24),     # SJ — Svalbard
    "yellowknife":    (-114.38, 62.45, -114.34, 62.47), # CA
    "torshavn":       (-6.78, 62.00, -6.76, 62.02),     # FO — Faroe Islands
    "kiruna_wide":    (20.18, 67.84, 20.27, 67.89),     # SE — broader
    "bergen_wide":    ( 5.28, 60.37,  5.40, 60.42),     # NO — broader
    "reykjavik_wide": (-22.05, 64.10, -21.85, 64.18),   # IS — broader
    "sapporo_wide":   (141.30, 43.04, 141.40, 43.10),   # JP — broader
}

# Winter ranges to scan. Two recent winters covers shifting Mapillary contributor activity.
WINTER_RANGES = [
    ("2024-01-15", "2024-03-15"),
    ("2025-01-15", "2025-03-15"),
]

# Summer range to verify a clear-season counterpart sequence exists in the bbox.
SUMMER_RANGE = ("2024-06-01", "2024-09-15")


def _fetch_json(params: dict) -> dict:
    qs = urllib.parse.urlencode(params)
    url = f"{GRAPH_URL}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "snow-underlay-recon/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _query_window(token: str, bbox: tuple[float, float, float, float],
                  start_iso: str, end_iso: str, *, limit: int = 2000) -> list[dict]:
    bs = ",".join(f"{x}" for x in bbox)
    params = {
        "access_token": token,
        "bbox": bs,
        "start_captured_at": f"{start_iso}T00:00:00Z",
        "end_captured_at": f"{end_iso}T23:59:59Z",
        "fields": "id,sequence,geometry,captured_at,compass_angle,is_pano,creator",
        "limit": limit,
    }
    j = _fetch_json(params)
    return j.get("data", [])


def _aggregate_sequences(images: list[dict]) -> dict[str, dict]:
    """Group images by sequence_id, summarise."""
    by_seq: dict[str, list[dict]] = defaultdict(list)
    for img in images:
        sid = img.get("sequence")
        if not sid:
            continue
        by_seq[sid].append(img)

    out: dict[str, dict] = {}
    for sid, imgs in by_seq.items():
        imgs.sort(key=lambda x: int(x.get("captured_at", 0)))
        n = len(imgs)
        is_pano = any(bool(i.get("is_pano")) for i in imgs)
        creators = sorted({((i.get("creator") or {}).get("username") or "") for i in imgs})
        # Cadence: median ms between consecutive frames.
        if n >= 2:
            ts = [int(i.get("captured_at") or 0) for i in imgs]
            deltas = sorted(ts[i + 1] - ts[i] for i in range(n - 1))
            median_ms = deltas[len(deltas) // 2]
        else:
            median_ms = 0
        out[sid] = {
            "sequence_id": sid,
            "n_images": n,
            "is_pano": is_pano,
            "creators": creators,
            "median_frame_dt_ms": median_ms,
            "first_capture_iso": _iso(imgs[0].get("captured_at")),
            "last_capture_iso": _iso(imgs[-1].get("captured_at")),
            "first_image_id": imgs[0].get("id"),
            "last_image_id": imgs[-1].get("id"),
            "first_lng": _lng(imgs[0]),
            "first_lat": _lat(imgs[0]),
            "image_ids": [i["id"] for i in imgs],
        }
    return out


def _iso(ms_str: object) -> str:
    if not ms_str:
        return ""
    try:
        ms = int(ms_str)
    except (TypeError, ValueError):
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ms / 1000))


def _lng(img: dict) -> float:
    g = img.get("geometry") or {}
    coords = g.get("coordinates") or [None, None]
    return float(coords[0]) if coords[0] is not None else 0.0


def _lat(img: dict) -> float:
    g = img.get("geometry") or {}
    coords = g.get("coordinates") or [None, None]
    return float(coords[1]) if coords[1] is not None else 0.0


def _rank(seq: dict) -> float:
    """Higher is better. Penalise pano. Reward length, normal cadence (1–10s)."""
    score = 0.0
    if seq["is_pano"]:
        return -1.0
    score += min(seq["n_images"], 200) / 10.0
    dt = seq["median_frame_dt_ms"]
    if 500 <= dt <= 10_000:
        score += 5.0
    elif 200 <= dt <= 30_000:
        score += 2.0
    return score


def scan_city(token: str, city: str, bbox: tuple[float, float, float, float]) -> dict:
    print(f"\n=== {city}  bbox={bbox} ===")
    winter_seqs: dict[str, dict] = {}
    for s, e in WINTER_RANGES:
        try:
            imgs = _query_window(token, bbox, s, e)
        except Exception as exc:
            print(f"  query failed ({s}..{e}): {exc}")
            continue
        if not imgs:
            print(f"  {s}..{e}: 0 images")
            continue
        seqs = _aggregate_sequences(imgs)
        for sid, info in seqs.items():
            info["winter_range"] = f"{s}..{e}"
            winter_seqs[sid] = info
        print(f"  {s}..{e}: {len(imgs)} images → {len(seqs)} sequences")

    # Summer counterpart presence in the bbox.
    summer_seq_count = 0
    try:
        summer_imgs = _query_window(token, bbox, *SUMMER_RANGE)
        summer_seqs = _aggregate_sequences(summer_imgs)
        summer_seq_count = len(summer_seqs)
    except Exception as exc:
        print(f"  summer query failed: {exc}")
        summer_seqs = {}

    # Rank winter sequences.
    ranked = sorted(winter_seqs.values(), key=_rank, reverse=True)

    return {
        "city": city,
        "bbox": list(bbox),
        "summer_sequences_in_bbox": summer_seq_count,
        "n_winter_sequences": len(winter_seqs),
        "top_winter_sequences": ranked[:10],
        "summer_sequences_sample": list(summer_seqs.values())[:5],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True, help="city key from CITIES, or 'all'")
    args = p.parse_args()

    # Mirror the existing fetcher's env loading.
    from data.fetch_mapillary import _load_dotenv
    _load_dotenv()
    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        raise SystemExit("MAPILLARY_TOKEN not set in environment or .env.")

    OUT.mkdir(parents=True, exist_ok=True)

    cities = list(CITIES.keys()) if args.city == "all" else [args.city]
    if args.city != "all" and args.city not in CITIES:
        raise SystemExit(f"Unknown city {args.city!r}. Try one of: {list(CITIES)}")

    summary = {}
    for c in cities:
        report = scan_city(token, c, CITIES[c])
        summary[c] = report
        # Per-city CSV with one row per winter sequence.
        csv_path = OUT / f"{c}.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sequence_id", "n_images", "is_pano", "creators",
                        "median_frame_dt_ms", "first_capture_iso", "last_capture_iso",
                        "first_image_id", "first_lng", "first_lat",
                        "score", "summer_seqs_in_bbox"])
            for s in report["top_winter_sequences"]:
                w.writerow([
                    s["sequence_id"], s["n_images"], s["is_pano"],
                    ";".join(s["creators"][:3]), s["median_frame_dt_ms"],
                    s["first_capture_iso"], s["last_capture_iso"],
                    s["first_image_id"], f"{s['first_lng']:.5f}", f"{s['first_lat']:.5f}",
                    f"{_rank(s):.1f}", report["summer_sequences_in_bbox"],
                ])
        print(f"  wrote {csv_path}")

    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
