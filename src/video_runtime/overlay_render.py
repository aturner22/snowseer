"""Per-frame overlay compositor + ffmpeg → mp4.

Inputs: list[FrameResult] from `pipeline_v.run_track`.
Outputs: per-frame PNG under `outputs/video/<track_id>/frames/<idx>.png`,
then a single mp4 stitched with ffmpeg.

K.2 baseline: only the `overlay` mode (alpha-blend of fused mask onto snow).
K.5 will add `sidebyside` and `quad`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.overlay import alpha_blend
from src.video_runtime.pipeline_v import FrameResult

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/video"

GREEN = (0, 220, 100)


def render_overlay(
    results: list[FrameResult],
    track_id: str,
    *,
    out_name: str = "overlay_v0.mp4",
    fps: float = 10.0,
    keep_frames: bool = False,
) -> Path:
    out_dir = OUT / track_id
    frame_dir = out_dir / "frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render] writing {len(results)} frames to {frame_dir}")
    for i, r in enumerate(results):
        canvas = r.snow_image.copy()
        if r.fused_mask is not None and int(r.fused_mask.sum()) > 0:
            canvas = alpha_blend(canvas, r.fused_mask, color=GREEN, alpha=0.45)
        # Tiny diagnostic strip (top-left): frame index + priors_used + first inlier count.
        label = f"{i + 1}/{len(results)}  priors={r.n_priors_used}"
        cv2.putText(canvas, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(frame_dir / f"f{i:04d}.png"),
                    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    out_path = out_dir / out_name
    print(f"[render] ffmpeg → {out_path} @ {fps} fps")
    # `-vf scale=...` with `:flags=lanczos` ensures even dimensions for yuv420p
    # (libx264 rejects odd-dim frames). `force_original_aspect_ratio=decrease`
    # plus padding to a multiple of 2 is the standard safe combo.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", f"{fps}",
        "-i", str(frame_dir / "f%04d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    if not keep_frames:
        shutil.rmtree(frame_dir)
    return out_path
