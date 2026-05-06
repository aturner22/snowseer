"""CLI entry point for `make video-render`.

Usage:
    uv run python -m src.video_runtime.render --track <id> --mode <overlay|sidebyside|quad> [--start N --end N --stride N]

K.2 baseline only implements `overlay`. The other modes raise NotImplementedError
with a pointer to K.5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pathlib import Path

from src.video_runtime.pipeline_v import run_track
from src.video_runtime.overlay_render import render_overlay, render_sidebyside, render_quad
from src.video_runtime.temporal import make_smoother


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--mode", choices=["overlay", "sidebyside", "quad"], default="overlay")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
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
    p.add_argument("--temporal", choices=["none", "ema", "flow"], default="none",
                   help="temporal smoothing strategy (K.4 ablation). Default 'none'.")
    p.add_argument("--ema-alpha", type=float, default=0.5,
                   help="EMA weight on the current raw mask (0..1). Default 0.5.")
    p.add_argument("--flow-weight", type=float, default=0.5,
                   help="Flow propagation weight (0..1). Default 0.5.")
    p.add_argument("--synthetic-priors", type=int, default=0,
                   help="K.3 — number of past frames to surface as synthetic "
                        "priors per current frame. 0 disables.")
    p.add_argument("--cache-tag", default="default",
                   help="Identifier for the matching cache. Renders that share "
                        "(track, start, end, stride, K, max_dim, foreground_y_frac) "
                        "but differ only in smoother should share the same tag. "
                        "Synthetic-prior runs use a different tag because they "
                        "change the matching itself.")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Force re-matching even if a cache file exists.")
    args = p.parse_args()


    smoother = make_smoother(args.temporal, alpha=args.ema_alpha,
                             flow_weight=args.flow_weight)

    cache_path = Path(f"outputs/video/{args.track}/_cache_{args.cache_tag}.pkl")

    results = run_track(
        args.track,
        K=args.K,
        max_dim=args.max_dim,
        start=args.start,
        end=args.end,
        stride=args.stride,
        foreground_y_frac=args.foreground_y_frac,
        smoother=smoother,
        cache_path=cache_path,
        rebuild_cache=args.rebuild_cache,
        synthetic_priors=args.synthetic_priors,
    )

    if args.mode == "overlay":
        out = render_overlay(
            results, args.track,
            fps=args.fps,
            out_name=args.out_name,
            keep_frames=args.keep_frames,
        )
    elif args.mode == "sidebyside":
        out = render_sidebyside(
            results, args.track,
            fps=args.fps,
            out_name=args.out_name,
            keep_frames=args.keep_frames,
        )
    elif args.mode == "quad":
        import pickle as _pickle
        aug_path = Path(f"outputs/video/{args.track}/_aug_{args.cache_tag}.pkl")
        if not aug_path.exists():
            raise SystemExit(
                f"missing augmentation cache {aug_path}. "
                f"Run `uv run python -m src.video_runtime.augment "
                f"--track {args.track} --cache-tag {args.cache_tag}` first."
            )
        with open(aug_path, "rb") as fh:
            aug = _pickle.load(fh)
        out = render_quad(
            results, args.track,
            summer_panels=aug["summer_panels"],
            naive_masks=aug["naive_masks"],
            fps=args.fps,
            out_name=args.out_name,
            keep_frames=args.keep_frames,
        )
    else:
        raise AssertionError(f"unreachable mode: {args.mode}")
    print(f"[render] wrote {out}")


if __name__ == "__main__":
    main()
