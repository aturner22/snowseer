"""Per-snow-frame quality scoring.

A snow plough's actual camera produces sharp, well-framed captures with the
road clearly visible. Mapillary contributor uploads include heavy motion blur,
night-with-no-lighting, and windshield-blocked frames that are not in the
plough's operating regime. This module scores each snow frame on three cheap
metrics so we can pre-filter junk before the human curator sees them.

Metrics (all computed over the lower 70% of the image, where the road sits):
    - sharpness: variance of the Laplacian. Higher = sharper.
    - brightness: median V channel of HSV. Drop very dark frames.
    - edge_density: Canny edge count / pixel count. Drops featureless frames.

Each metric is normalised to [0, 1] via per-metric percentile rank across the
image set, then averaged into a `composite` score. A `pass: bool` is set by
applying configurable thresholds.

Usage:
    uv run python -m src.snow_quality                # score all pairs
    uv run python -m src.snow_quality --top 100      # show top 100 by score
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

PAIRS_DIR = Path("data/pairs")
ROAD_REGION_TOP_FRAC = 0.30  # ignore top 30% of image (sky / building tops)
ROAD_REGION_BOTTOM_FRAC = 1.00  # include all the way down; dashboard handled elsewhere

DEFAULT_THRESHOLDS = {
    # Conservative defaults — designed to keep the top ~50% of frames.
    "sharpness_min": 60.0,    # raw Laplacian variance
    "brightness_min": 35.0,   # 0-255 V channel median
    "edge_density_min": 0.010,  # fraction of road-region pixels that are Canny edges
}


@dataclass
class SnowQuality:
    sharpness: float
    brightness: float
    edge_density: float
    composite: float | None = None  # populated after batch normalisation
    sharpness_pass: bool = False
    brightness_pass: bool = False
    edge_pass: bool = False
    overall_pass: bool = False


def _road_region(img_bgr: np.ndarray) -> np.ndarray:
    h = img_bgr.shape[0]
    y0 = int(round(h * ROAD_REGION_TOP_FRAC))
    y1 = int(round(h * ROAD_REGION_BOTTOM_FRAC))
    return img_bgr[y0:y1, :, :]


def score_image(img_bgr: np.ndarray) -> SnowQuality:
    """Compute the three raw metrics on one image (BGR uint8)."""
    region = _road_region(img_bgr)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    brightness = float(np.median(hsv[..., 2]))
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.mean() / 255.0)
    return SnowQuality(
        sharpness=round(sharpness, 2),
        brightness=round(brightness, 1),
        edge_density=round(edge_density, 4),
    )


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    rank = np.empty_like(order, dtype=np.float64)
    n = len(values)
    rank[order] = np.linspace(0.0, 1.0, n)
    return rank


def score_all(pairs_dir: Path = PAIRS_DIR, thresholds: dict | None = None) -> list[dict]:
    """Score every snow.jpg under pairs_dir; persist per-pair JSON; return summary."""
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    pair_dirs = sorted(p for p in pairs_dir.iterdir() if p.is_dir())
    raw: list[tuple[Path, SnowQuality]] = []
    for d in pair_dirs:
        snow = d / "snow.jpg"
        if not snow.exists():
            continue
        img = cv2.imread(str(snow), cv2.IMREAD_COLOR)
        if img is None:
            continue
        sq = score_image(img)
        sq.sharpness_pass = sq.sharpness >= thresholds["sharpness_min"]
        sq.brightness_pass = sq.brightness >= thresholds["brightness_min"]
        sq.edge_pass = sq.edge_density >= thresholds["edge_density_min"]
        sq.overall_pass = sq.sharpness_pass and sq.brightness_pass and sq.edge_pass
        raw.append((d, sq))

    if not raw:
        return []

    sharp_arr = np.array([sq.sharpness for _, sq in raw])
    bright_arr = np.array([sq.brightness for _, sq in raw])
    edge_arr = np.array([sq.edge_density for _, sq in raw])
    sharp_rank = _percentile_rank(sharp_arr)
    bright_rank = _percentile_rank(bright_arr)
    edge_rank = _percentile_rank(edge_arr)

    summary: list[dict] = []
    for i, (pair_dir, sq) in enumerate(raw):
        sq.composite = round(float((sharp_rank[i] + bright_rank[i] + edge_rank[i]) / 3.0), 4)
        out = asdict(sq)
        out["thresholds"] = thresholds
        (pair_dir / "snow_quality.json").write_text(json.dumps(out, indent=2))
        summary.append({"pair_id": pair_dir.name, **out})

    summary.sort(key=lambda s: -(s["composite"] or 0))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default=str(PAIRS_DIR))
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--sharpness-min", type=float, default=DEFAULT_THRESHOLDS["sharpness_min"])
    ap.add_argument("--brightness-min", type=float, default=DEFAULT_THRESHOLDS["brightness_min"])
    ap.add_argument("--edge-density-min", type=float, default=DEFAULT_THRESHOLDS["edge_density_min"])
    args = ap.parse_args()

    thresholds = {
        "sharpness_min": args.sharpness_min,
        "brightness_min": args.brightness_min,
        "edge_density_min": args.edge_density_min,
    }
    summary = score_all(Path(args.pairs_dir), thresholds=thresholds)

    print(f"Scored {len(summary)} snow frames.")
    n_pass = sum(1 for s in summary if s["overall_pass"])
    print(f"Passing all three thresholds: {n_pass}  ({n_pass/max(1,len(summary)):.0%})")
    print(f"Thresholds: {thresholds}")
    print()
    print(f"Top {args.top} by composite score:")
    print(f"  {'pair_id':62s}  composite  sharp  bright  edge   pass")
    for s in summary[: args.top]:
        flags = "".join(["S" if s["sharpness_pass"] else "-",
                          "B" if s["brightness_pass"] else "-",
                          "E" if s["edge_pass"] else "-"])
        print(f"  {s['pair_id']:62s}  {s['composite']:.3f}     {s['sharpness']:6.1f}  {s['brightness']:5.1f}  {s['edge_density']:.3f}  {flags}")


if __name__ == "__main__":
    main()
