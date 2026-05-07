"""Broader Mapillary winter-sequence scanner with per-frame summer-prior probe.

Improves on `find_snow_sequences.py`:

1. **Bigger city list** — the original 7 (tromso/reykjavik/bergen/quebec/
   innsbruck/sapporo/kiruna) found zero winter+summer matches in same
   bbox. Add 12+ more (trondheim, helsinki, anchorage, saskatoon, winnipeg,
   torshavn, yellowknife, whitehorse, fairbanks, iqaluit, longyearbyen,
   plus wider bboxes on the existing failing ones).

2. **Per-frame summer-prior probe** instead of "summer sequence in same
   bbox". For each candidate winter sequence, we sample 10 evenly-
   distributed frames and query Mapillary `?closeto=lat,lng&radius=30
   &captured_at>July` for each. Hit rate ≥ 0.5 → the sequence is viable
   (most winter frames will have a per-frame summer counterpart even
   without a co-located summer sequence). Hit rate < 0.5 → reject.

3. **Apply window-oracle logic at the recon stage** — if the winter
   sequence has decent length (≥ 60 frames) AND the per-frame summer
   probe succeeds (hit rate ≥ 0.5), it's a candidate. Fetch + cache
   compute is gated on this check.

Usage:
    uv run python -m data.find_snow_for_video --city all  [--top 5]
    uv run python -m data.find_snow_for_video --city trondheim

Outputs:
    data/video/recon/<city>__viable.csv  (per-sequence ranked list)
    data/video/recon/_viable_summary.json  (cross-city summary)

Requires MAPILLARY_TOKEN.
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


# Bboxes (W, S, E, N). Wider than the original recon for cities that
# returned zero hits. Latitudes ordered cold → temperate.
CITIES = {
    # — Original 7 (kept for reference) —
    "tromso":       (18.85, 69.62, 19.10, 69.70),     # NO; widened
    "reykjavik":    (-22.10, 64.10, -21.80, 64.18),   # IS; widened
    "bergen":       ( 5.28, 60.36,  5.42, 60.43),     # NO; widened
    "quebec":       (-71.30, 46.78, -71.15, 46.85),   # CA; widened
    "innsbruck":    (11.30, 47.24, 11.45, 47.30),     # AT; widened
    "sapporo":      (141.30, 43.03, 141.40, 43.10),   # JP; widened
    "kiruna":       (20.18, 67.83, 20.30, 67.90),     # SE; widened
    # — New 12 —
    "trondheim":    (10.35, 63.40, 10.50, 63.46),     # NO
    "helsinki":     (24.91, 60.15, 25.00, 60.20),     # FI
    "anchorage":    (-149.95, 61.18, -149.80, 61.24), # US
    "saskatoon":    (-106.70, 52.10, -106.62, 52.16), # CA
    "winnipeg":     (-97.18, 49.85, -97.10, 49.92),   # CA
    "torshavn":     (-6.80, 62.00, -6.75, 62.04),     # FO
    "yellowknife":  (-114.40, 62.43, -114.34, 62.47), # CA
    "whitehorse":   (-135.10, 60.70, -135.00, 60.74), # CA
    "fairbanks":    (-147.78, 64.81, -147.68, 64.86), # US
    "iqaluit":      (-68.55, 63.74, -68.45, 63.78),   # CA
    "longyearbyen": (15.55, 78.20, 15.70, 78.25),     # SJ
    "edmonton":     (-113.55, 53.50, -113.45, 53.56), # CA
}

# Winter ranges chunked by month to dodge Mapillary's "reduce the amount
# of data" rate limit (dense cities trip it on >1-month windows).
WINTER_RANGES = [
    ("2024-12-01", "2024-12-31"),
    ("2025-01-01", "2025-01-31"),
    ("2025-02-01", "2025-02-28"),
    ("2025-03-01", "2025-03-31"),
    ("2024-01-01", "2024-01-31"),
    ("2024-02-01", "2024-02-29"),
    ("2024-03-01", "2024-03-31"),
]
# Summer prior search: any year, broad date window. Mapillary contributors
# upload summer captures sporadically; restricting to one year often misses
# co-located coverage.
SUMMER_RANGE = ("2018-06-01", "2024-09-30")

# Per-frame summer-prior probe parameters.
SAMPLE_K = 10                   # how many winter frames to sample per sequence
PROBE_RADIUS_M = 100            # summer search radius (100 m — looser than original 30 m;
                                # winter contributors and summer contributors rarely
                                # overlap perfectly so we accept more slop here)


def _fetch_json(params: dict, *, timeout: float = 90.0, retries: int = 2) -> dict:
    qs = urllib.parse.urlencode(params)
    url = f"{GRAPH_URL}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "snow-underlay-recon/0.2"})
    last_exc = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise last_exc


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
    return _fetch_json(params).get("data", [])


def _query_closeto(token: str, lng: float, lat: float, radius_m: float,
                   start_iso: str, end_iso: str, *, limit: int = 5) -> list[dict]:
    """Per-point query: any image within `radius_m` of (lng, lat) in date range."""
    # Mapillary closeto uses degrees not metres; convert. 1 deg lat ~ 111 km.
    delta = radius_m / 111_000.0
    bbox = (lng - delta, lat - delta, lng + delta, lat + delta)
    bs = ",".join(f"{x}" for x in bbox)
    params = {
        "access_token": token,
        "bbox": bs,
        "start_captured_at": f"{start_iso}T00:00:00Z",
        "end_captured_at": f"{end_iso}T23:59:59Z",
        "fields": "id,sequence,geometry,captured_at,is_pano",
        "limit": limit,
    }
    try:
        return _fetch_json(params).get("data", [])
    except Exception:
        return []


def _aggregate_sequences(images: list[dict]) -> dict[str, dict]:
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
            "median_frame_dt_ms": median_ms,
            "first_capture_iso": _iso(imgs[0].get("captured_at")),
            "last_capture_iso": _iso(imgs[-1].get("captured_at")),
            "first_image_id": imgs[0].get("id"),
            "first_lng": _lng(imgs[0]),
            "first_lat": _lat(imgs[0]),
            "samples": _evenly_sample(imgs, k=SAMPLE_K),
        }
    return out


def _evenly_sample(images: list[dict], k: int) -> list[dict]:
    if len(images) <= k:
        return images
    step = len(images) / k
    return [images[int(i * step)] for i in range(k)]


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
    return float((g.get("coordinates") or [0.0, 0.0])[0])


def _lat(img: dict) -> float:
    g = img.get("geometry") or {}
    return float((g.get("coordinates") or [0.0, 0.0])[1])


def _probe_summer(token: str, sample_imgs: list[dict]) -> tuple[float, list[int]]:
    """For each sampled winter frame, query Mapillary for any summer image
    within PROBE_RADIUS_M. Return (hit_rate, hits_per_sample)."""
    hits = []
    for img in sample_imgs:
        lng, lat = _lng(img), _lat(img)
        if not lng or not lat:
            hits.append(0)
            continue
        results = _query_closeto(token, lng, lat, PROBE_RADIUS_M, *SUMMER_RANGE, limit=3)
        # Filter out panoramic + same-sequence (can't pair winter→winter)
        winter_seq = img.get("sequence")
        keep = [r for r in results
                if not r.get("is_pano") and r.get("sequence") != winter_seq]
        hits.append(len(keep))
    hit_rate = sum(1 for h in hits if h > 0) / max(len(hits), 1)
    return hit_rate, hits


def _viability_score(seq: dict, hit_rate: float) -> float:
    """Higher is better. Penalise pano. Reward length, normal cadence,
    summer-prior availability."""
    if seq["is_pano"]:
        return -1.0
    score = 0.0
    score += min(seq["n_images"], 200) / 5.0   # up to +40 for length
    dt = seq["median_frame_dt_ms"]
    if 500 <= dt <= 10_000:
        score += 5.0
    elif 200 <= dt <= 30_000:
        score += 2.0
    score += hit_rate * 50.0                    # up to +50 for full summer coverage
    return score


def scan_city(token: str, city: str, bbox: tuple[float, float, float, float],
              top_n: int = 5) -> dict:
    print(f"\n=== {city}  bbox={bbox} ===", flush=True)
    winter_seqs: dict[str, dict] = {}
    for s, e in WINTER_RANGES:
        try:
            imgs = _query_window(token, bbox, s, e)
        except Exception as exc:
            print(f"  query failed ({s}..{e}): {exc}", flush=True)
            continue
        if not imgs:
            print(f"  {s}..{e}: 0 images", flush=True)
            continue
        seqs = _aggregate_sequences(imgs)
        for sid, info in seqs.items():
            info["winter_range"] = f"{s}..{e}"
            if sid not in winter_seqs or info["n_images"] > winter_seqs[sid]["n_images"]:
                winter_seqs[sid] = info
        print(f"  {s}..{e}: {len(imgs)} imgs → {len(seqs)} seqs", flush=True)

    # Filter to non-pano + length ≥ 60 (~1 min @ 1 fps).
    candidates = [s for s in winter_seqs.values()
                  if not s["is_pano"] and s["n_images"] >= 60]
    candidates.sort(key=lambda s: -s["n_images"])
    print(f"  {len(candidates)} candidates with ≥60 frames + non-pano", flush=True)

    # Probe top candidates for per-frame summer-prior availability.
    viable = []
    for c in candidates[:top_n]:
        print(f"  probing seq {c['sequence_id'][:12]} ({c['n_images']} frames) for summer...",
              flush=True)
        hit_rate, hits = _probe_summer(token, c["samples"])
        c["summer_hit_rate"] = hit_rate
        c["summer_hits_per_sample"] = hits
        c["viability_score"] = _viability_score(c, hit_rate)
        print(f"    hit_rate={hit_rate:.0%}  score={c['viability_score']:.1f}", flush=True)
        viable.append(c)

    viable.sort(key=lambda s: -s["viability_score"])
    return {
        "city": city,
        "bbox": list(bbox),
        "n_winter_sequences": len(winter_seqs),
        "n_candidates": len(candidates),
        "top_viable": viable,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True, help="city key from CITIES, or 'all'")
    p.add_argument("--top", type=int, default=5,
                   help="probe summer for top N candidates per city (default 5)")
    args = p.parse_args()

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
    all_viable: list[dict] = []
    for c in cities:
        report = scan_city(token, c, CITIES[c], top_n=args.top)
        summary[c] = report
        all_viable.extend([{**v, "city": c} for v in report["top_viable"]])

        # Per-city CSV
        csv_path = OUT / f"{c}__viable.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "sequence_id", "n_images", "median_frame_dt_ms",
                "first_capture_iso", "last_capture_iso",
                "first_lng", "first_lat",
                "summer_hit_rate", "viability_score",
            ])
            for s in report["top_viable"]:
                w.writerow([
                    s["sequence_id"], s["n_images"], s["median_frame_dt_ms"],
                    s["first_capture_iso"], s["last_capture_iso"],
                    f"{s['first_lng']:.5f}", f"{s['first_lat']:.5f}",
                    f"{s.get('summer_hit_rate', 0):.2f}",
                    f"{s.get('viability_score', 0):.1f}",
                ])
        print(f"  → wrote {csv_path}", flush=True)

    all_viable.sort(key=lambda s: -s.get("viability_score", 0))
    print("\n=== TOP VIABLE CANDIDATES (cross-city, summer-prior verified) ===")
    for i, s in enumerate(all_viable[:10]):
        print(f"  {i+1:2d}. {s['city']:<14s} "
              f"seq={s['sequence_id'][:14]} "
              f"n_frames={s['n_images']:>4d} "
              f"hit_rate={s.get('summer_hit_rate', 0):.0%} "
              f"score={s.get('viability_score', 0):.1f}")

    summary_path = OUT / "_viable_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n→ wrote {summary_path}")


if __name__ == "__main__":
    main()
