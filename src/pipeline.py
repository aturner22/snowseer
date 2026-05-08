"""End-to-end static-stills pipeline.

Loads a snow + clear-prior pair, runs DISK + LightGlue matching, fits a
homography, segments the prior, warps the road mask onto the snow frame,
and saves the raw constituent images. The website and writeup compose
those into 2x2 grids at the HTML / markdown layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .fuse import crop_foreground
from .homography import HomographyResult, estimate, refine_iteratively
from .matching import Matcher, draw_matches
from .overlay import alpha_blend, keep_largest_component, warp_mask
from .segmentation import RoadSegmenter

# When the initial homography has fewer than this many inliers, run a
# segmentation-guided refinement pass before generating overlays. Below this
# we are likely in a tunnel / off-road / drift situation where the generic
# lower-image-half ground-plane bias is too coarse.
REFINEMENT_INLIER_TRIGGER = 25

DATA_PAIRS_DIR = Path("data/pairs")
OUT_DIR = Path("outputs/nordic_stills")
DEMO_PAIRS_PATH = Path(__file__).resolve().parent / "data" / "demo_pairs.json"


def _display_strings(pair_id: str) -> tuple[str, str]:
    """(title, subtitle) for a pair_id, sourced from data/demo_pairs.json.

    Title:    'Place, Country, condition phrase'
    Subtitle: 'Snow capture (month yyyy)  ↔  Clear capture (month yyyy)'
    """
    if not DEMO_PAIRS_PATH.exists():
        return (pair_id.replace("__", "  ·  "), "")
    spec = json.loads(DEMO_PAIRS_PATH.read_text())
    entry = next((p for p in spec.get("pairs", []) if p.get("pair_id") == pair_id), None)
    if entry is None:
        return (pair_id.replace("__", "  ·  "), "")
    place = entry.get("place") or entry.get("region", "")
    condition = entry.get("condition") or ""
    title = f"{place}, {condition}" if condition else place
    snow_t = entry.get("snow_captured", "")
    clear_t = entry.get("clear_captured", "")
    subtitle = f"{snow_t}  ↔  {clear_t}" if snow_t and clear_t else ""
    return (title, subtitle)


def _load_demo_pair_ids() -> set[str]:
    """Pair IDs the demo manifest commits to. Empty set if the file is missing."""
    if not DEMO_PAIRS_PATH.exists():
        return set()
    spec = json.loads(DEMO_PAIRS_PATH.read_text())
    return {p["pair_id"] for p in spec.get("pairs", [])}


# Inlier threshold below which a pair is flagged as content-mismatched in
# the per-pair summary (different scenes despite tight GPS+heading). Pairs
# remain on disk regardless; the flag is for downstream review.
ACCEPT_INLIER_MIN = 15


@dataclass
class PairResult:
    pair_id: str
    snow_path: Path
    clear_path: Path
    n_matches: int
    n_inliers: int
    used_ground_plane_restriction: bool
    snow_overlay_path: Path | None
    naive_baseline_path: Path | None
    H: np.ndarray | None
    iou_overlay_vs_naive: float | None = None
    iou_overlay_vs_identity: float | None = None
    refined: bool = False


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = (a & b).sum()
    union = (a | b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


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


def _process_one_prior(
    snow: np.ndarray, prior: np.ndarray,
    matcher: Matcher, segmenter: RoadSegmenter,
) -> dict | None:
    """Match snow against one prior, segment + warp the prior road mask
    into snow space, return per-prior outputs. None if matching fails.
    """
    matches = matcher.match(snow, prior)
    homo: HomographyResult = estimate(matches, snow.shape[:2], prior.shape[:2])
    if homo.H is None:
        return None
    road_mask_clear = keep_largest_component(segmenter.segment_road(prior))
    refined = False
    if homo.n_inliers < REFINEMENT_INLIER_TRIGGER:
        refined_homo = refine_iteratively(
            matches, homo.H, snow.shape[:2], prior.shape[:2], road_mask_clear,
        )
        if refined_homo.H is not None and refined_homo.n_inliers > homo.n_inliers:
            homo = refined_homo
            refined = True
    H_inv = np.linalg.inv(homo.H)
    road_mask_snow = keep_largest_component(
        warp_mask(road_mask_clear, H_inv, snow.shape[:2])
    )
    valid_region = warp_mask(
        np.ones(prior.shape[:2], dtype=np.uint8), H_inv, snow.shape[:2]
    )
    return {
        "matches": matches,
        "homo": homo,
        "refined": refined,
        "road_mask_clear": road_mask_clear,
        "road_mask_snow": road_mask_snow,
        "valid_region": valid_region,
        "prior": prior,
    }


def run_pair(
    pair_dir: Path,
    matcher: Matcher,
    segmenter: RoadSegmenter,
    *,
    out_dir: Path = OUT_DIR,
    max_dim: int = 1024,
) -> PairResult:
    """Run the cross-season pipeline on one pair_dir.

    Reads the first prior from `meta.json` (or `clear.jpg` as fallback),
    matches snow → prior, fits a homography, segments the prior, warps
    the road mask back into snow space, foreground-crops, and saves:

        <pair_id>__snow.png              (raw snow query)
        <pair_id>__clear.png             (raw clear prior)
        <pair_id>__clear_with_mask.png   (clear with green road mask)
        <pair_id>__overlay.png           (cross-season overlay on snow)
        <pair_id>__naive_baseline.png    (Cityscapes direct on snow)
        <pair_id>__matches.png           (correspondence visualisation)
    """
    snow_path = pair_dir / "snow.jpg"
    snow = _resize_to(_load_rgb(snow_path), max_dim)
    pair_id = pair_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = pair_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    prior_specs = meta.get("priors")
    if not prior_specs:
        prior_specs = [{"file": "clear.jpg", "id": meta.get("clear", {}).get("id", "")}]
    ps = prior_specs[0]

    prior_path = pair_dir / ps["file"]
    if not prior_path.exists():
        print(f"  ! {pair_id}: prior {ps['file']} missing", flush=True)
        return PairResult(
            pair_id=pair_id, snow_path=snow_path, clear_path=pair_dir / "clear.jpg",
            n_matches=0, n_inliers=0, used_ground_plane_restriction=False,
            snow_overlay_path=None, naive_baseline_path=None, H=None,
        )
    prior = _resize_to(_load_rgb(prior_path), max_dim)
    primary = _process_one_prior(snow, prior, matcher, segmenter)
    if primary is None:
        return PairResult(
            pair_id=pair_id, snow_path=snow_path, clear_path=prior_path,
            n_matches=0, n_inliers=0, used_ground_plane_restriction=False,
            snow_overlay_path=None, naive_baseline_path=None, H=None,
        )
    primary["prior_id"] = ps.get("id", "")
    primary["prior_file"] = ps["file"]

    # Matches visualisation.
    draw_matches(
        snow, primary["prior"], primary["matches"],
        inlier_mask=primary["homo"].inlier_mask,
        out_path=out_dir / f"{pair_id}__matches.png",
    )

    # Naive baseline: Cityscapes segmenter applied directly to the snow frame.
    road_mask_snow_naive = segmenter.segment_road(snow)
    snow_naive = alpha_blend(snow, road_mask_snow_naive, color=(220, 60, 50), alpha=0.55)
    naive_path = out_dir / f"{pair_id}__naive_baseline.png"
    cv2.imwrite(str(naive_path), cv2.cvtColor(snow_naive, cv2.COLOR_RGB2BGR))

    # Cross-season overlay: warp prior road mask, foreground-crop, alpha-blend.
    overlay_mask = keep_largest_component(
        crop_foreground(primary["road_mask_snow"], foreground_y_frac=0.45)
    )
    snow_overlay = alpha_blend(snow, overlay_mask, color=(46, 156, 86), alpha=0.50)
    cv2.imwrite(
        str(out_dir / f"{pair_id}__overlay.png"),
        cv2.cvtColor(snow_overlay, cv2.COLOR_RGB2BGR),
    )

    # Raw constituents for HTML / markdown to compose.
    primary_clear = primary["prior"]
    primary_road_mask_clear = primary["road_mask_clear"]
    clear_with_mask = alpha_blend(primary_clear, primary_road_mask_clear,
                                  color=(46, 156, 86), alpha=0.50)
    cv2.imwrite(str(out_dir / f"{pair_id}__snow.png"),
                cv2.cvtColor(snow, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{pair_id}__clear.png"),
                cv2.cvtColor(primary_clear, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{pair_id}__clear_with_mask.png"),
                cv2.cvtColor(clear_with_mask, cv2.COLOR_RGB2BGR))

    # IoU metrics: overlay vs naive (how different is the cross-season
    # output from the naive baseline?), and overlay vs identity-warped
    # prior mask (how much did the homography warp it?).
    iou_naive = _iou(overlay_mask, road_mask_snow_naive)
    sh, sw = snow.shape[:2]
    ch, cw = primary_road_mask_clear.shape[:2]
    if (sh, sw) == (ch, cw):
        identity_mask = primary_road_mask_clear
    else:
        identity_mask = cv2.resize(primary_road_mask_clear, (sw, sh), interpolation=cv2.INTER_NEAREST)
    iou_identity = _iou(overlay_mask, identity_mask)

    return PairResult(
        pair_id=pair_id,
        snow_path=snow_path,
        clear_path=pair_dir / primary["prior_file"],
        n_matches=len(primary["matches"].kpts0),
        n_inliers=primary["homo"].n_inliers,
        used_ground_plane_restriction=primary["homo"].used_ground_plane_restriction,
        snow_overlay_path=out_dir / f"{pair_id}__overlay.png",
        naive_baseline_path=naive_path,
        H=primary["homo"].H,
        iou_overlay_vs_naive=iou_naive,
        iou_overlay_vs_identity=iou_identity,
        refined=primary["refined"],
    )


def run_all(
    pairs_dir: Path = DATA_PAIRS_DIR,
    out_dir: Path = OUT_DIR,
    max_dim: int = 1024,
    *,
    require_demo_manifest: bool = True,
) -> list[PairResult]:
    """Run the pipeline on every pair in the demo manifest.

    The demo set is the pair IDs in `data/demo_pairs.json`. When
    `require_demo_manifest` is True (default), the pipeline refuses to
    run unless the manifest exists, so the only pairs ever processed
    are the curated 18.
    """
    pair_dirs = sorted(p for p in pairs_dir.iterdir() if p.is_dir())
    if not pair_dirs:
        raise SystemExit(
            f"No pairs under {pairs_dir}. "
            f"Run `uv run python -m src.data.fetch_mapillary --curated-only` first."
        )

    demo_ids = _load_demo_pair_ids()
    if require_demo_manifest:
        if not demo_ids:
            raise SystemExit(
                f"No demo manifest found at {DEMO_PAIRS_PATH}. "
                f"Run `uv run python -m src.data.fetch_mapillary --curated-only` to populate it."
            )
        before = len(pair_dirs)
        pair_dirs = [d for d in pair_dirs if d.name in demo_ids]
        print(f"demo set active: {len(pair_dirs)} / {before} pairs ({DEMO_PAIRS_PATH})", flush=True)
    elif demo_ids:
        before = len(pair_dirs)
        pair_dirs = [d for d in pair_dirs if d.name in demo_ids]
        print(f"demo set (advisory): {len(pair_dirs)} / {before} pairs", flush=True)

    matcher = Matcher()
    segmenter = RoadSegmenter()
    results: list[PairResult] = []
    summary: list[dict] = []
    for d in pair_dirs:
        try:
            res = run_pair(d, matcher, segmenter, out_dir=out_dir, max_dim=max_dim)
        except Exception as e:
            print(f"  ! {d.name}: {e}", flush=True)
            continue
        results.append(res)
        accept = res.n_inliers >= ACCEPT_INLIER_MIN
        summary.append(
            {
                "pair_id": res.pair_id,
                "n_matches": int(res.n_matches),
                "n_inliers": int(res.n_inliers),
                "ground_plane_used": bool(res.used_ground_plane_restriction),
                "refined": bool(res.refined),
                "iou_overlay_vs_naive": (
                    None if res.iou_overlay_vs_naive is None else round(res.iou_overlay_vs_naive, 4)
                ),
                "iou_overlay_vs_identity": (
                    None if res.iou_overlay_vs_identity is None else round(res.iou_overlay_vs_identity, 4)
                ),
                "accept": bool(accept),
                "naive_baseline": str(res.naive_baseline_path) if res.naive_baseline_path else None,
            }
        )
        verdict = "ACCEPT" if accept else "reject"
        print(
            f"  [{verdict}] {res.pair_id}: matches={res.n_matches} "
            f"inliers={res.n_inliers} (ground-plane={res.used_ground_plane_restriction})",
            flush=True,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return results


def _cli() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", default=str(DATA_PAIRS_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--pair-id", default=None, help="Run a single pair by directory name.")
    parser.add_argument(
        "--allow-uncurated", action="store_true",
        help="Bypass the demo-manifest gate (default: required). "
             "With this flag, run on every pair-directory on disk.",
    )
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
        run_all(
            pairs_dir=pairs_dir, out_dir=out_dir, max_dim=args.max_dim,
            require_demo_manifest=not args.allow_uncurated,
        )


if __name__ == "__main__":
    _cli()
