"""End-to-end pipeline: snow query + clear prior -> overlay figure.

Caches intermediate model outputs so reruns are fast.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .homography import HomographyResult, estimate
from .matching import Matcher, MatchResult, draw_matches
from .overlay import alpha_blend, three_panel_figure, warp_mask
from .segmentation import RoadSegmenter

DATA_PAIRS_DIR = Path("data/pairs")
OUT_DIR = Path("outputs/heroes")


@dataclass
class PairResult:
    pair_id: str
    snow_path: Path
    clear_path: Path
    n_matches: int
    n_inliers: int
    used_ground_plane_restriction: bool
    figure_path: Path | None
    snow_overlay_path: Path | None
    naive_baseline_path: Path | None
    H: np.ndarray | None


def _load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize_to(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return img
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def run_pair(
    pair_dir: Path,
    matcher: Matcher,
    segmenter: RoadSegmenter,
    *,
    out_dir: Path = OUT_DIR,
    max_dim: int = 1024,
) -> PairResult:
    snow_path = pair_dir / "snow.jpg"
    clear_path = pair_dir / "clear.jpg"
    snow = _resize_to(_load_rgb(snow_path), max_dim)
    clear = _resize_to(_load_rgb(clear_path), max_dim)

    # 1. Match snow (img0) <-> clear (img1)
    matches = matcher.match(snow, clear)

    # 2. Estimate homography snow -> clear with ground-plane bias
    homo: HomographyResult = estimate(matches, snow.shape[:2], clear.shape[:2])

    # Save correspondences viz (sanity)
    pair_id = pair_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_path = out_dir / f"{pair_id}__matches.png"
    draw_matches(snow, clear, matches, inlier_mask=homo.inlier_mask, out_path=viz_path)

    if homo.H is None:
        return PairResult(
            pair_id=pair_id,
            snow_path=snow_path,
            clear_path=clear_path,
            n_matches=len(matches.kpts0),
            n_inliers=0,
            used_ground_plane_restriction=homo.used_ground_plane_restriction,
            figure_path=None,
            snow_overlay_path=None,
            naive_baseline_path=None,
            H=None,
        )

    # 3. Road mask on the *clear* prior (the model knows clear-weather roads)
    road_mask_clear = segmenter.segment_road(clear)

    # 4. Warp the road mask: clear -> snow.
    # We have H: snow -> clear. We need H^-1: clear -> snow.
    H_inv = np.linalg.inv(homo.H)
    road_mask_snow = warp_mask(road_mask_clear, H_inv, snow.shape[:2])

    # 5. Compose figures
    snow_overlay = alpha_blend(snow, road_mask_snow, color=(0, 255, 100), alpha=0.45)
    snow_overlay_path = out_dir / f"{pair_id}__overlay.png"
    cv2.imwrite(str(snow_overlay_path), cv2.cvtColor(snow_overlay, cv2.COLOR_RGB2BGR))

    figure_path = out_dir / f"{pair_id}__panel.png"
    three_panel_figure(
        snow, clear, road_mask_clear, snow_overlay,
        title=f"{pair_id}    inliers={homo.n_inliers}    ground-plane bias={homo.used_ground_plane_restriction}",
        out_path=figure_path,
    )

    # 6. Naive baseline: run the same Cityscapes segmenter directly on the snow frame.
    #    Expected to produce a fragmented / shifted / collapsed road prediction.
    road_mask_snow_naive = segmenter.segment_road(snow)
    snow_naive = alpha_blend(snow, road_mask_snow_naive, color=(255, 80, 80), alpha=0.45)
    naive_path = out_dir / f"{pair_id}__naive_baseline.png"
    cv2.imwrite(str(naive_path), cv2.cvtColor(snow_naive, cv2.COLOR_RGB2BGR))

    return PairResult(
        pair_id=pair_id,
        snow_path=snow_path,
        clear_path=clear_path,
        n_matches=len(matches.kpts0),
        n_inliers=homo.n_inliers,
        used_ground_plane_restriction=homo.used_ground_plane_restriction,
        figure_path=figure_path,
        snow_overlay_path=snow_overlay_path,
        naive_baseline_path=naive_path,
        H=homo.H,
    )


def run_all(
    pairs_dir: Path = DATA_PAIRS_DIR,
    out_dir: Path = OUT_DIR,
    max_dim: int = 1024,
) -> list[PairResult]:
    pair_dirs = sorted(p for p in pairs_dir.iterdir() if p.is_dir())
    if not pair_dirs:
        raise SystemExit(f"No pairs under {pairs_dir}. Run `uv run python -m data.fetch_mapillary` first.")
    matcher = Matcher()
    segmenter = RoadSegmenter()
    results: list[PairResult] = []
    summary: list[dict] = []
    for d in pair_dirs:
        try:
            res = run_pair(d, matcher, segmenter, out_dir=out_dir, max_dim=max_dim)
        except Exception as e:
            print(f"  ! {d.name}: {e}")
            continue
        results.append(res)
        summary.append(
            {
                "pair_id": res.pair_id,
                "n_matches": int(res.n_matches),
                "n_inliers": int(res.n_inliers),
                "ground_plane_used": bool(res.used_ground_plane_restriction),
                "figure": str(res.figure_path) if res.figure_path else None,
                "naive_baseline": str(res.naive_baseline_path) if res.naive_baseline_path else None,
            }
        )
        print(f"  {res.pair_id}: matches={res.n_matches} inliers={res.n_inliers} (ground-plane={res.used_ground_plane_restriction})")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return results


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", default=str(DATA_PAIRS_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--pair-id", default=None, help="Run a single pair by directory name.")
    args = parser.parse_args()

    pairs_dir = Path(args.pairs_dir)
    out_dir = Path(args.out_dir)

    if args.pair_id:
        single = pairs_dir / args.pair_id
        if not single.exists():
            raise SystemExit(f"No such pair: {single}")
        matcher = Matcher()
        segmenter = RoadSegmenter()
        res = run_pair(single, matcher, segmenter, out_dir=out_dir, max_dim=args.max_dim)
        print(json.dumps(
            {"pair_id": res.pair_id, "n_matches": res.n_matches, "n_inliers": res.n_inliers},
            indent=2,
        ))
    else:
        run_all(pairs_dir=pairs_dir, out_dir=out_dir, max_dim=args.max_dim)


if __name__ == "__main__":
    _cli()
