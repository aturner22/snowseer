"""CLI entry point for `make video-render`.

Usage:
    uv run python -m src.video_runtime.render --track <id> --mode <overlay|sidebyside|quad> [--start N --end N --stride N]

K.2 baseline only implements `overlay`. The other modes raise NotImplementedError
with a pointer to K.5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.video_runtime.pipeline_v import run_track
from src.video_runtime.overlay_render import render_overlay


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
    args = p.parse_args()

    if args.mode != "overlay":
        raise NotImplementedError(
            f"--mode={args.mode} is K.5 work. K.2 baseline implements only 'overlay'."
        )

    results = run_track(
        args.track,
        K=args.K,
        max_dim=args.max_dim,
        start=args.start,
        end=args.end,
        stride=args.stride,
        foreground_y_frac=args.foreground_y_frac,
    )

    out = render_overlay(
        results, args.track,
        fps=args.fps,
        out_name=args.out_name,
        keep_frames=args.keep_frames,
    )
    print(f"[render] wrote {out}")


if __name__ == "__main__":
    main()
