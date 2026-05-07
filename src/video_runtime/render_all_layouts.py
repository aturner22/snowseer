"""Build matching cache + augmentation + matches sidecar + render all six layouts.

Usage:
    uv run python -m src.video_runtime.render_all_layouts \\
        --track <track_id> --cache-tag <tag> \\
        [--start N --end N --stride N --K K --ema-alpha A]

Produces outputs/video/<track>/{
    _cache_<tag>.pkl, _aug_<tag>.pkl, _matches_<tag>.pkl,
    overlay.mp4, sidebyside.mp4, matches.mp4,
    snow_naive_overlay.mp4, snow_overlay_naive.mp4, quad.mp4,
}.

Cache, aug pass, and matches sidecar are built automatically if missing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n")
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0:
        sys.exit(rc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--cache-tag", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=350)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--ema-alpha", type=float, default=0.4)
    p.add_argument("--max-dim", type=int, default=1024)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--seg-prob-threshold", type=float, default=None,
                   help="optional road-class probability threshold for the "
                        "summer segmentation. Default = argmax.")
    p.add_argument("--seg-morph-radius", type=int, default=0,
                   help="optional morphology radius post-threshold. 0 = off.")
    p.add_argument("--min-spatial-diversity", type=float, default=None,
                   help="optional spatial-diversity inlier-bbox threshold; "
                        "see render.py --min-spatial-diversity for details.")
    p.add_argument("--weight-strategy", default="inliers",
                   choices=["inliers", "inliers_x_diversity"],
                   help="see render.py --weight-strategy")
    p.add_argument("--outlier-drop", action="store_true",
                   help="see render.py --outlier-drop")
    p.add_argument("--min-frame-quality", type=float, default=None,
                   help="see render.py --min-frame-quality")
    args = p.parse_args()

    aug_path = ROOT / f"outputs/video/{args.track}/_aug_{args.cache_tag}.pkl"
    cache_path = ROOT / f"outputs/video/{args.track}/_cache_{args.cache_tag}.pkl"
    matches_path = ROOT / f"outputs/video/{args.track}/_matches_{args.cache_tag}.pkl"

    seg_args = []
    if args.seg_prob_threshold is not None:
        seg_args += ["--seg-prob-threshold", str(args.seg_prob_threshold)]
    if args.seg_morph_radius > 0:
        seg_args += ["--seg-morph-radius", str(args.seg_morph_radius)]
    if args.min_spatial_diversity is not None:
        seg_args += ["--min-spatial-diversity", str(args.min_spatial_diversity)]
    if args.weight_strategy != "inliers":
        seg_args += ["--weight-strategy", args.weight_strategy]
    if args.outlier_drop:
        seg_args += ["--outlier-drop"]
    if args.min_frame_quality is not None:
        seg_args += ["--min-frame-quality", str(args.min_frame_quality)]

    if not cache_path.exists():
        print(f"[render-all] matching cache missing; building...")
        _run([
            "uv", "run", "python", "-m", "src.video_runtime.render",
            "--track", args.track,
            "--start", str(args.start), "--end", str(args.end), "--stride", str(args.stride),
            "--K", str(args.K), "--max-dim", str(args.max_dim),
            "--temporal", "none",
            "--cache-tag", args.cache_tag,
            "--mode", "cache-only",
            *seg_args,
        ])

    if not aug_path.exists():
        print(f"[render-all] augmentation cache missing; building...")
        _run([
            "uv", "run", "python", "-m", "src.video_runtime.augment",
            "--track", args.track, "--cache-tag", args.cache_tag,
            "--K", str(args.K), "--max-dim", str(args.max_dim),
        ])

    if not matches_path.exists():
        print(f"[render-all] matches sidecar missing; building...")
        _run([
            "uv", "run", "python", "-m", "src.video_runtime.matches_pass",
            "--track", args.track, "--cache-tag", args.cache_tag,
            "--K", str(args.K), "--max-dim", str(args.max_dim),
        ])

    common = [
        "uv", "run", "python", "-m", "src.video_runtime.render",
        "--track", args.track,
        "--start", str(args.start), "--end", str(args.end), "--stride", str(args.stride),
        "--K", str(args.K),
        "--temporal", "ema", "--ema-alpha", str(args.ema_alpha),
        "--cache-tag", args.cache_tag,
        "--max-dim", str(args.max_dim), "--fps", str(args.fps),
    ]

    # Output names match the layout, no cache-tag suffix. Each track gets its
    # own outputs/video/<track>/ directory so per-tag disambiguation isn't
    # needed; the slide deck + GitHub Pages reference these stable names.
    layouts = [
        ("overlay", "overlay.mp4"),
        ("sidebyside", "sidebyside.mp4"),
        ("matches", "matches.mp4"),
        ("snow_naive_overlay", "snow_naive_overlay.mp4"),
        ("snow_overlay_naive", "snow_overlay_naive.mp4"),
        ("quad", "quad.mp4"),
    ]
    for mode, out_name in layouts:
        _run(common + ["--mode", mode, "--out-name", out_name])

    print(f"\n[render-all] done — wrote {len(layouts)} mp4s under outputs/video/{args.track}/")


if __name__ == "__main__":
    main()
