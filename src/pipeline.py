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

from .homography import HomographyResult, estimate, refine_iteratively
from .matching import Matcher, MatchResult, draw_matches
from .overlay import alpha_blend, keep_largest_component, three_panel_figure, warp_mask
from .segmentation import RoadSegmenter

# When the initial homography has fewer than this many inliers, run a
# segmentation-guided refinement pass before generating overlays. Below this
# we are likely in a tunnel / off-road / drift situation where the generic
# lower-image-half ground-plane bias is too coarse.
REFINEMENT_INLIER_TRIGGER = 25

DATA_PAIRS_DIR = Path("data/pairs")
OUT_DIR = Path("outputs/heroes")

# Content-level curation threshold. Pairs below this are considered
# content-mismatched (different scenes despite tight GPS+heading) and are
# excluded from the curated demo set. They remain on disk and are visible in
# `outputs/heroes/summary.json` with `accept=false`.
ACCEPT_INLIER_MIN = 15


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

    # 3. Road mask on the *clear* prior (the model knows clear-weather roads).
    #    Reduce to the single largest connected component — the road in front
    #    of the camera. Kills disconnected sidewalks / distant road patches /
    #    far-end visible road on the other side of an intersection.
    road_mask_clear = segmenter.segment_road(clear)
    road_mask_clear = keep_largest_component(road_mask_clear)

    # 3b. If the initial fit is shaky, run iterative segmentation-guided
    #     refinement. This rescues drift cases where the lower-image-half
    #     ground-plane bias was too coarse to dominate (tunnels, scenes where
    #     most of the lower half is wall, not road).
    refined = False
    if homo.n_inliers < REFINEMENT_INLIER_TRIGGER:
        refined_homo = refine_iteratively(
            matches, homo.H, snow.shape[:2], clear.shape[:2], road_mask_clear,
        )
        if refined_homo.H is not None and refined_homo.n_inliers > homo.n_inliers:
            homo = refined_homo
            refined = True

    # 4. Warp the road mask: clear -> snow.
    # We have H: snow -> clear. We need H^-1: clear -> snow.
    H_inv = np.linalg.inv(homo.H)
    road_mask_snow = warp_mask(road_mask_clear, H_inv, snow.shape[:2])
    # Drop small "island" regions left behind by warp aliasing or pixels that
    # got clipped at the destination boundary; keep only the dominant road in
    # front of the plough.
    road_mask_snow = keep_largest_component(road_mask_snow)

    # 5. Compose figures
    snow_overlay = alpha_blend(snow, road_mask_snow, color=(0, 255, 100), alpha=0.45)
    snow_overlay_path = out_dir / f"{pair_id}__overlay.png"
    cv2.imwrite(str(snow_overlay_path), cv2.cvtColor(snow_overlay, cv2.COLOR_RGB2BGR))

    figure_path = out_dir / f"{pair_id}__panel.png"
    three_panel_figure(
        snow, clear, road_mask_clear, snow_overlay,
        title=(
            f"{pair_id}    inliers={homo.n_inliers}    "
            f"ground-plane bias={homo.used_ground_plane_restriction}    "
            f"refined={refined}"
        ),
        out_path=figure_path,
    )

    # 6. Naive baseline: run the same Cityscapes segmenter directly on the snow frame.
    #    Expected to produce a fragmented / shifted / collapsed road prediction.
    road_mask_snow_naive = segmenter.segment_road(snow)
    snow_naive = alpha_blend(snow, road_mask_snow_naive, color=(255, 80, 80), alpha=0.45)
    naive_path = out_dir / f"{pair_id}__naive_baseline.png"
    cv2.imwrite(str(naive_path), cv2.cvtColor(snow_naive, cv2.COLOR_RGB2BGR))

    # 7. Identity-warp baseline: trust the prior road mask without any
    #    matching/registration. This is "what if we just overlaid the clear
    #    road mask onto snow without doing the cross-season alignment". For
    #    any pair where snow and clear differ at all in framing, this is wrong.
    sh, sw = snow.shape[:2]
    ch, cw = road_mask_clear.shape[:2]
    if (sh, sw) == (ch, cw):
        identity_mask = road_mask_clear
    else:
        identity_mask = cv2.resize(
            road_mask_clear, (sw, sh), interpolation=cv2.INTER_NEAREST
        )

    iou_naive = _iou(road_mask_snow, road_mask_snow_naive)
    iou_identity = _iou(road_mask_snow, identity_mask)

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
        iou_overlay_vs_naive=iou_naive,
        iou_overlay_vs_identity=iou_identity,
        refined=refined,
    )


MANUAL_CURATION_PATH = Path("data/manual_snow_curation.json")


def _load_manual_curation() -> dict[str, str]:
    """pair_id -> verdict ('accept'|'reject'|'skip'). Empty if file missing."""
    if not MANUAL_CURATION_PATH.exists():
        return {}
    try:
        raw = json.loads(MANUAL_CURATION_PATH.read_text())
    except json.JSONDecodeError:
        return {}
    return {k: v.get("verdict", "skip") for k, v in raw.items() if isinstance(v, dict)}


def run_all(
    pairs_dir: Path = DATA_PAIRS_DIR,
    out_dir: Path = OUT_DIR,
    max_dim: int = 1024,
    *,
    require_manual_curation: bool = True,
) -> list[PairResult]:
    """Run the pipeline on all pairs accepted by manual snow-quality curation.

    require_manual_curation (default True): the pipeline refuses to run unless
    `data/manual_snow_curation.json` exists and has at least one accept. This
    guards against accidentally producing overlays for the long tail of
    motion-blurred / windshield-blocked snow frames that the user has already
    decided not to demo. Pass False (or --allow-uncurated on the CLI) to
    bypass.
    """
    pair_dirs = sorted(p for p in pairs_dir.iterdir() if p.is_dir())
    if not pair_dirs:
        raise SystemExit(f"No pairs under {pairs_dir}. Run `uv run python -m data.fetch_mapillary` first.")

    manual = _load_manual_curation()
    if require_manual_curation:
        if not MANUAL_CURATION_PATH.exists():
            raise SystemExit(
                f"{MANUAL_CURATION_PATH} not found. Run the snow curator first:\n"
                f"  uv run streamlit run demo/curate_snow.py\n"
                f"Or pass --allow-uncurated to run on every pair."
            )
        accepted_ids = {pid for pid, v in manual.items() if v == "accept"}
        if not accepted_ids:
            raise SystemExit(
                f"{MANUAL_CURATION_PATH} has no accepts. Run the snow curator and "
                f"accept at least one pair, or pass --allow-uncurated."
            )
        before = len(pair_dirs)
        pair_dirs = [d for d in pair_dirs if d.name in accepted_ids]
        print(f"manual curation active: {len(pair_dirs)} / {before} pairs (manually accepted)")
    elif manual:
        before = len(pair_dirs)
        pair_dirs = [d for d in pair_dirs if manual.get(d.name) == "accept"]
        print(f"manual curation present (advisory): {len(pair_dirs)} / {before} pairs")

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
        accept = res.n_inliers >= ACCEPT_INLIER_MIN
        summary.append(
            {
                "pair_id": res.pair_id,
                "n_matches": int(res.n_matches),
                "n_inliers": int(res.n_inliers),
                "ground_plane_used": bool(res.used_ground_plane_restriction),
                "refined": bool(res.refined),
                "manual_verdict": manual.get(res.pair_id, "—"),
                "iou_overlay_vs_naive": (
                    None if res.iou_overlay_vs_naive is None else round(res.iou_overlay_vs_naive, 4)
                ),
                "iou_overlay_vs_identity": (
                    None if res.iou_overlay_vs_identity is None else round(res.iou_overlay_vs_identity, 4)
                ),
                "accept": bool(accept),
                "figure": str(res.figure_path) if res.figure_path else None,
                "naive_baseline": str(res.naive_baseline_path) if res.naive_baseline_path else None,
            }
        )
        verdict = "ACCEPT" if accept else "reject"
        print(f"  [{verdict}] {res.pair_id}: matches={res.n_matches} inliers={res.n_inliers} (ground-plane={res.used_ground_plane_restriction})")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return results


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", default=str(DATA_PAIRS_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--pair-id", default=None, help="Run a single pair by directory name.")
    parser.add_argument("--allow-uncurated", action="store_true",
                        help="Bypass the manual snow curation gate (default: required).")
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
            require_manual_curation=not args.allow_uncurated,
        )


if __name__ == "__main__":
    _cli()
