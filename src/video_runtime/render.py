"""Single-layout video renderer.

Builds (or loads) a `_cache_<tag>.pkl` matching cache for the track, then
renders one mp4 layout from it. Invoked as a subprocess by
`render_all_layouts.py`; all modes share the same cache. Always processes
every frame the fetcher pulled for the track.

Usage:
    uv run python -m src.video_runtime.render --track <id> --mode <overlay|sidebyside|snow_naive_overlay|quad|matches|cache-only>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.video_runtime.pipeline_v import run_track
from src.video_runtime.overlay_render import (
    render_matches,
    render_overlay,
    render_quad,
    render_sidebyside,
    render_three_panel,
)
from src.video_runtime.pipeline_v import make_smoother


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--mode", choices=["overlay", "sidebyside", "snow_naive_overlay",
                                       "quad", "matches", "cache-only"],
                   default="overlay")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--K", type=int, default=3, help="number of summer priors per snow frame")
    p.add_argument("--max-dim", type=int, default=1024)
    p.add_argument("--fps", type=float, default=10.0,
                   help="output video frame rate (Boreas camera is 10 Hz)")
    p.add_argument("--out-name", default="overlay_v0.mp4")
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--foreground-y-frac", type=float, default=0.30,
                   help="cut everything above this fraction of image height "
                        "(roof-mounted Boreas camera ≈ 0.30)")
    p.add_argument("--temporal", choices=["none", "ema"], default="none",
                   help="temporal smoothing strategy. Default 'none'.")
    p.add_argument("--ema-alpha", type=float, default=0.5,
                   help="EMA weight on the current raw mask (0..1). Default 0.5.")
    p.add_argument("--cache-tag", default="default",
                   help="Identifier for the matching cache. Renders that share "
                        "(track, stride, K, max_dim, foreground_y_frac) but "
                        "differ only in smoother should share the same tag. "
                        "Synthetic-prior runs use a different tag because they "
                        "change the matching itself.")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Force re-matching even if a cache file exists.")
    p.add_argument("--seg-prob-threshold", type=float, default=None,
                   help="If set (e.g. 0.6), keep summer road-class pixels only "
                        "where the per-pixel road score exceeds this threshold. "
                        "Default = argmax. Useful when the segmenter over-claims "
                        "road class on a given track.")
    p.add_argument("--seg-morph-radius", type=int, default=0,
                   help="Open+close morphology radius applied to summer road masks "
                        "after thresholding. 0 = off. Suppresses one- or two-pixel "
                        "jaggies that warp into amplified jitter.")
    p.add_argument("--debug-strip", action="store_true",
                   help="overlay mode: include the diagnostic strip "
                        "(frame index + priors_used). Off for final renders.")
    p.add_argument("--label-panels", action="store_true", default=True,
                   help="sidebyside / quad modes: caption each panel with its "
                        "role. Default ON (panel labels are useful context).")
    p.add_argument("--no-label-panels", dest="label_panels", action="store_false",
                   help="disable per-panel captions for a totally bare render")
    args = p.parse_args()


    smoother = make_smoother(args.temporal, alpha=args.ema_alpha)

    cache_path = Path(f"outputs/toronto_video/{args.track}/_cache_{args.cache_tag}.pkl")

    results = run_track(
        args.track,
        K=args.K,
        max_dim=args.max_dim,
        start=0,
        end=None,
        stride=args.stride,
        foreground_y_frac=args.foreground_y_frac,
        smoother=smoother,
        cache_path=cache_path,
        rebuild_cache=args.rebuild_cache,
        seg_prob_threshold=args.seg_prob_threshold,
        seg_morph_radius=args.seg_morph_radius,
    )

    if args.mode == "cache-only":
        print(f"[render] cache built at {cache_path}; no mp4 written")
        return

    if args.mode == "overlay":
        out = render_overlay(
            results, args.track,
            fps=args.fps,
            out_name=args.out_name,
            keep_frames=args.keep_frames,
            debug_strip=args.debug_strip,
        )
    elif args.mode == "sidebyside":
        out = render_sidebyside(
            results, args.track,
            fps=args.fps,
            out_name=args.out_name,
            keep_frames=args.keep_frames,
            label_panels=args.label_panels,
        )
    elif args.mode == "matches":
        import pickle as _pickle
        sidecar_path = Path(f"outputs/toronto_video/{args.track}/_matches_{args.cache_tag}.pkl")
        if not sidecar_path.exists():
            raise SystemExit(
                f"missing matches sidecar {sidecar_path}. "
                f"Run `uv run python -m src.video_runtime.matches_pass "
                f"--track {args.track} --cache-tag {args.cache_tag}` first."
            )
        with open(sidecar_path, "rb") as fh:
            sidecar = _pickle.load(fh)
        out = render_matches(
            results, args.track,
            matches_frames=sidecar["frames"],
            fps=args.fps,
            out_name=args.out_name,
            keep_frames=args.keep_frames,
        )
    elif args.mode in ("snow_naive_overlay", "quad"):
        import pickle as _pickle
        aug_path = Path(f"outputs/toronto_video/{args.track}/_aug_{args.cache_tag}.pkl")
        if not aug_path.exists():
            raise SystemExit(
                f"missing augmentation cache {aug_path}. "
                f"Run `uv run python -m src.video_runtime.augment "
                f"--track {args.track} --cache-tag {args.cache_tag}` first."
            )
        with open(aug_path, "rb") as fh:
            aug = _pickle.load(fh)
        if args.mode == "quad":
            out = render_quad(
                results, args.track,
                summer_panels=aug["summer_panels"],
                naive_masks=aug["naive_masks"],
                fps=args.fps,
                out_name=args.out_name,
                keep_frames=args.keep_frames,
                label_panels=args.label_panels,
            )
        else:
            out = render_three_panel(
                results, args.track,
                naive_masks=aug["naive_masks"],
                fps=args.fps,
                out_name=args.out_name,
                keep_frames=args.keep_frames,
                label_panels=args.label_panels,
            )
    else:
        raise AssertionError(f"unreachable mode: {args.mode}")
    print(f"[render] wrote {out}")


if __name__ == "__main__":
    main()
