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

from .fuse import (
    crop_foreground,
    majority_vote,
    union_with_edge_erosion,
    weighted_soft_average,
)
from .homography import HomographyResult, estimate, refine_iteratively
from .matching import Matcher, MatchResult, draw_matches
from .overlay import alpha_blend, keep_largest_component, panel_figure, warp_mask
from .segmentation import RoadSegmenter

# When the initial homography has fewer than this many inliers, run a
# segmentation-guided refinement pass before generating overlays. Below this
# we are likely in a tunnel / off-road / drift situation where the generic
# lower-image-half ground-plane bias is too coarse.
REFINEMENT_INLIER_TRIGGER = 25

DATA_PAIRS_DIR = Path("data/pairs")
OUT_DIR = Path("outputs/heroes")
DEMO_PAIRS_PATH = Path("data/demo_pairs.json")


def _display_strings(pair_id: str) -> tuple[str, str]:
    """(title, subtitle) for a pair_id, sourced from data/demo_pairs.json.

    Title:    'Place, Country — condition phrase'
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
    title = f"{place} — {condition}" if condition else place
    snow_t = entry.get("snow_captured", "")
    clear_t = entry.get("clear_captured", "")
    subtitle = f"{snow_t}  ↔  {clear_t}" if snow_t and clear_t else ""
    return (title, subtitle)


def _load_demo_pair_ids() -> set[str]:
    """Return the pair IDs the static-stills demo uses (from data/demo_pairs.json).

    The file is the demo manifest — a list of (Mapillary snow image id +
    paired clear-season images) the fetcher pulls and the pipeline runs
    against. Empty set if the file is missing.
    """
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


def _process_one_prior(
    snow: np.ndarray, prior: np.ndarray,
    matcher: Matcher, segmenter: RoadSegmenter,
) -> dict | None:
    """Match snow against a single prior, segment + warp the prior road mask
    into snow space, return per-prior outputs. None if matching fails entirely.
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
    max_priors: int | None = 1,
) -> PairResult:
    """Run the cross-season pipeline on a single pair_dir.

    Reads `meta.json` to discover priors and processes up to `max_priors` of
    them. The K resulting road masks are fused via three strategies (union
    with edge-erosion, weighted soft-average, hard majority vote),
    foreground-cropped, and saved.

    Modes:
    - **`max_priors=1` (default, the v1.x narrative)**: only the first prior
      is used. The fusion variants degenerate to a single mask, so they are
      skipped — only the v1 outputs (`__matches.png`, `__naive_baseline.png`,
      `__overlay.png`, `__panel.png`) are written.
    - **`max_priors=N>1` (multi-prior, the Phase J ablation)**: all priors
      up to N are processed; the three fusion variants and a per-prior
      strip are written in addition to the v1 outputs.
    - **`max_priors=None`**: use every prior in `meta.priors`.

    Falls back to single-prior behaviour if `meta.priors` is missing.
    """
    snow_path = pair_dir / "snow.jpg"
    snow = _resize_to(_load_rgb(snow_path), max_dim)
    pair_id = pair_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = pair_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    prior_specs = meta.get("priors")
    if not prior_specs:
        # Back-compat: single clear.jpg path.
        prior_specs = [{"file": "clear.jpg", "id": meta.get("clear", {}).get("id", "")}]
    # Apply max_priors cap. Default 1 = single-prior (v1.x narrative) which
    # produces only the v1 outputs (matches/naive/overlay/panel) and skips
    # the multi-prior fusion variants. Set to 5 (or None) to recover Phase J.
    if max_priors is not None:
        prior_specs = prior_specs[:max_priors]

    # Process each prior independently.
    per_prior: list[dict] = []
    for ps in prior_specs:
        prior_path = pair_dir / ps["file"]
        if not prior_path.exists():
            print(f"  ! {pair_id}: prior {ps['file']} missing", flush=True)
            continue
        prior = _resize_to(_load_rgb(prior_path), max_dim)
        result = _process_one_prior(snow, prior, matcher, segmenter)
        if result is None:
            continue
        result["prior_id"] = ps.get("id", "")
        result["prior_file"] = ps["file"]
        per_prior.append(result)

    if not per_prior:
        return PairResult(
            pair_id=pair_id, snow_path=snow_path, clear_path=pair_dir / "clear.jpg",
            n_matches=0, n_inliers=0, used_ground_plane_restriction=False,
            figure_path=None, snow_overlay_path=None, naive_baseline_path=None,
            H=None,
        )

    # The 'displayed' primary is priors[0] (the v1 canonical clear_id) so the
    # panel's clear+mask column matches the v1.1 hero scene people remember.
    # Multi-prior fusion still uses all K priors for the snow-side overlay.
    primary = per_prior[0]
    # Save matches.png from the highest-inlier prior (best diagnostic).
    best_for_matches = max(per_prior, key=lambda r: r["homo"].n_inliers)
    matches_canvas = draw_matches(
        snow, best_for_matches["prior"], best_for_matches["matches"],
        inlier_mask=best_for_matches["homo"].inlier_mask,
        out_path=out_dir / f"{pair_id}__matches.png",
    )

    # Naive baseline (direct on snow). Same as v1.1.
    road_mask_snow_naive = segmenter.segment_road(snow)
    snow_naive = alpha_blend(snow, road_mask_snow_naive, color=(220, 60, 50), alpha=0.55)
    naive_path = out_dir / f"{pair_id}__naive_baseline.png"
    cv2.imwrite(str(naive_path), cv2.cvtColor(snow_naive, cv2.COLOR_RGB2BGR))

    # Fuse the K warped masks via three strategies.
    masks = [r["road_mask_snow"] for r in per_prior]
    valids = [r["valid_region"] for r in per_prior]
    weights = [float(r["homo"].n_inliers) for r in per_prior]

    fused = {
        "union": union_with_edge_erosion(masks, valids, erosion_px=20),
        "weighted": weighted_soft_average(masks, weights, valids, threshold=0.4),
        "majority": majority_vote(masks, valids),
    }
    # Foreground crop + largest-component cleanup on each fused mask.
    for k, m in fused.items():
        m = crop_foreground(m, foreground_y_frac=0.45)
        fused[k] = keep_largest_component(m)

    # In single-prior mode (max_priors=1) the three fusion strategies are
    # degenerate (all == the single mask), so skip the per-fusion overlays
    # and the priors strip. The headline `__overlay.png` is still written.
    is_single_prior = len(per_prior) == 1
    # Default fusion in multi-prior mode: weighted soft-average (best on the
    # 27-pair manifest by inspection; others are kept available for ablation).
    chosen = "weighted"
    if not is_single_prior:
        # Save per-fusion snow overlays.
        for name, mask in fused.items():
            snow_overlay = alpha_blend(snow, mask, color=(46, 156, 86), alpha=0.50)
            cv2.imwrite(
                str(out_dir / f"{pair_id}__overlay_{name}.png"),
                cv2.cvtColor(snow_overlay, cv2.COLOR_RGB2BGR),
            )
    # Default `__overlay.png`: in single-prior mode this IS the only mask;
    # in multi-prior mode it's the user-picked fusion (or `weighted` default).
    cv2.imwrite(
        str(out_dir / f"{pair_id}__overlay.png"),
        cv2.cvtColor(
            alpha_blend(snow, fused[chosen], color=(46, 156, 86), alpha=0.50),
            cv2.COLOR_RGB2BGR,
        ),
    )

    # Per-prior thumbnails strip — only meaningful for K > 1.
    if not is_single_prior:
        _save_priors_strip(snow, per_prior, out_dir / f"{pair_id}__priors.png")

    # Headline 2x2 panel uses the user-chosen fusion (or weighted default)
    # + the canonical primary prior's clear+mask. (The clear+mask column
    # shows what *one* prior says; the overlay column shows the fused
    # multi-prior result.)
    primary_clear = primary["prior"]
    primary_road_mask_clear = primary["road_mask_clear"]
    panel_overlay = alpha_blend(snow, fused[chosen], color=(46, 156, 86), alpha=0.50)
    figure_path = out_dir / f"{pair_id}__panel.png"
    title, subtitle = _display_strings(pair_id)
    panel_figure(
        snow, primary_clear, primary_road_mask_clear, panel_overlay,
        snowy_naive=snow_naive,
        title=title, subtitle=subtitle, out_path=figure_path,
    )

    from src.layouts import save_extra_layouts
    save_extra_layouts(
        snow=snow,
        clear=primary_clear,
        road_mask_clear=primary_road_mask_clear,
        snow_naive=snow_naive,
        snow_overlay=panel_overlay,
        matches_canvas=matches_canvas,
        out_dir=out_dir,
        pair_id=pair_id,
    )

    # IoU metrics — measured against the chosen fusion (the one we ship).
    iou_naive = _iou(fused[chosen], road_mask_snow_naive)
    sh, sw = snow.shape[:2]
    ch, cw = primary_road_mask_clear.shape[:2]
    if (sh, sw) == (ch, cw):
        identity_mask = primary_road_mask_clear
    else:
        identity_mask = cv2.resize(primary_road_mask_clear, (sw, sh), interpolation=cv2.INTER_NEAREST)
    iou_identity = _iou(fused[chosen], identity_mask)

    return PairResult(
        pair_id=pair_id,
        snow_path=snow_path,
        clear_path=pair_dir / primary["prior_file"],
        n_matches=len(primary["matches"].kpts0),
        n_inliers=primary["homo"].n_inliers,
        used_ground_plane_restriction=primary["homo"].used_ground_plane_restriction,
        figure_path=figure_path,
        snow_overlay_path=out_dir / f"{pair_id}__overlay.png",
        naive_baseline_path=naive_path,
        H=primary["homo"].H,
        iou_overlay_vs_naive=iou_naive,
        iou_overlay_vs_identity=iou_identity,
        refined=primary["refined"],
    )


def _save_priors_strip(snow: np.ndarray, per_prior: list[dict], out_path: Path) -> None:
    """One-row strip showing the K priors with per-prior overlay tints, for inspection."""
    n = len(per_prior)
    if n == 0:
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n + 1, figsize=(4.5 * (n + 1), 4.0), facecolor="#f6f3ee")
    if n + 1 == 1:
        axes = [axes]
    axes[0].imshow(snow); axes[0].set_axis_off()
    axes[0].set_title("Snow query", fontfamily="Inter", fontsize=11, color="#1c1c1c", pad=8, loc="left")
    for i, r in enumerate(per_prior, start=1):
        ovl = alpha_blend(snow, r["road_mask_snow"], color=(46, 156, 86), alpha=0.50)
        axes[i].imshow(ovl); axes[i].set_axis_off()
        axes[i].set_title(
            f"prior {i-1}  ·  inliers={r['homo'].n_inliers}",
            fontfamily="Inter", fontsize=11, color="#1c1c1c", pad=8, loc="left",
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="#f6f3ee")
    plt.close(fig)


def run_all(
    pairs_dir: Path = DATA_PAIRS_DIR,
    out_dir: Path = OUT_DIR,
    max_dim: int = 1024,
    *,
    require_demo_manifest: bool = True,
    max_priors: int | None = 1,
) -> list[PairResult]:
    """Run the pipeline on the demo manifest.

    The demo set is the pair IDs in `data/demo_pairs.json` — a list of
    Mapillary snow + clear-season image pairs the fetcher pulls and the
    pipeline runs against.

    require_demo_manifest (default True): the pipeline refuses to run unless
    `demo_pairs.json` exists. Pass False (or --allow-uncurated on the CLI)
    to bypass and run on every pair-directory on disk.

    max_priors (default 1): cap on priors per pair. 1 = single-prior (v1.x
    narrative); 5 / None = multi-prior (Phase J ablation). See run_pair.
    """
    pair_dirs = sorted(p for p in pairs_dir.iterdir() if p.is_dir())
    if not pair_dirs:
        raise SystemExit(f"No pairs under {pairs_dir}. Run `uv run python -m data.fetch_mapillary` first.")

    demo_ids = _load_demo_pair_ids()
    if require_demo_manifest:
        if not demo_ids:
            raise SystemExit(
                f"No demo manifest found at {DEMO_PAIRS_PATH}. Either:\n"
                f"  - run `uv run python -m data.fetch_mapillary --curated-only` to populate it, or\n"
                f"  - pass --allow-uncurated to run on every pair-directory on disk."
            )
        before = len(pair_dirs)
        pair_dirs = [d for d in pair_dirs if d.name in demo_ids]
        print(f"demo set active: {len(pair_dirs)} / {before} pairs ({DEMO_PAIRS_PATH})")
    elif demo_ids:
        before = len(pair_dirs)
        pair_dirs = [d for d in pair_dirs if d.name in demo_ids]
        print(f"demo set (advisory): {len(pair_dirs)} / {before} pairs")

    matcher = Matcher()
    segmenter = RoadSegmenter()
    results: list[PairResult] = []
    summary: list[dict] = []
    for d in pair_dirs:
        try:
            res = run_pair(
                d, matcher, segmenter,
                out_dir=out_dir, max_dim=max_dim, max_priors=max_priors,
            )
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
                        help="Bypass the demo-manifest gate (default: required). "
                             "With this flag, run on every pair-directory on disk.")
    parser.add_argument(
        "--max-priors", type=int, default=1,
        help="Priors per pair (default 1 = single-prior v1.x narrative). "
             "Set to 5 (or 0 for unlimited) to enable Phase J multi-prior fusion.",
    )
    args = parser.parse_args()

    pairs_dir = Path(args.pairs_dir)
    out_dir = Path(args.out_dir)
    max_priors = None if args.max_priors == 0 else args.max_priors

    if args.pair_id:
        single = pairs_dir / args.pair_id
        if not single.exists():
            raise SystemExit(f"No such pair: {single}")
        matcher = Matcher()
        segmenter = RoadSegmenter()
        res = run_pair(
            single, matcher, segmenter,
            out_dir=out_dir, max_dim=args.max_dim, max_priors=max_priors,
        )
        print(json.dumps(
            {"pair_id": res.pair_id, "n_matches": res.n_matches, "n_inliers": res.n_inliers},
            indent=2,
        ))
    else:
        run_all(
            pairs_dir=pairs_dir, out_dir=out_dir, max_dim=args.max_dim,
            require_demo_manifest=not args.allow_uncurated,
            max_priors=max_priors,
        )


if __name__ == "__main__":
    _cli()
