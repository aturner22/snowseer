"""Per-frame overlay compositor + ffmpeg → mp4.

Inputs: list[FrameResult] from `pipeline_v.run_track`.
Outputs: per-frame PNG under `outputs/toronto_video/<track_id>/frames/<idx>.png`,
then a single mp4 stitched with ffmpeg.

Modes:
  - `overlay`     : alpha-blend of fused mask onto snow.
  - `sidebyside`  : snow input | overlay output, two panels lockstepped.
  - `quad`        : 2x2 — snow query / naive (red) / summer + road / overlay.
  - `matches`     : snow + best summer prior with a small subset of
                    correspondence lines drawn between them.
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
OUT = ROOT / "outputs/toronto_video"

GREEN = (0, 220, 100)
RED = (220, 60, 60)


def _label(canvas: np.ndarray, text: str, *, position: str = "tl") -> np.ndarray:
    """Draw a small Inter-style label with a translucent dark bar behind it."""
    h, w = canvas.shape[:2]
    if position == "tl":
        x, y = 16, 36
    elif position == "tr":
        x, y = w - 380, 36
    elif position == "bl":
        x, y = 16, h - 16
    else:
        x, y = 16, 36
    # Black band underlay for readability.
    cv2.rectangle(canvas, (x - 6, y - 28), (x + len(text) * 13, y + 8),
                  (0, 0, 0), -1)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = target_h / h
    return cv2.resize(img, (int(round(w * s)), target_h), interpolation=cv2.INTER_AREA)


def _compose_overlay_panel(r: FrameResult, *, with_label: bool = True) -> np.ndarray:
    """Snow + green road mask. The headline panel."""
    canvas = r.snow_image.copy()
    if r.fused_mask is not None and int(r.fused_mask.sum()) > 0:
        canvas = alpha_blend(canvas, r.fused_mask, color=GREEN, alpha=0.45)
    if with_label:
        canvas = _label(canvas, "snow + cross-season road overlay")
    return canvas


def _compose_naive_panel(r: FrameResult, naive_mask: np.ndarray | None,
                        *, with_label: bool = True) -> np.ndarray:
    """Snow + (red) naive direct-on-snow Cityscapes segmentation. The failure."""
    canvas = r.snow_image.copy()
    if naive_mask is not None and int(naive_mask.sum()) > 0:
        canvas = alpha_blend(canvas, naive_mask, color=RED, alpha=0.45)
    if with_label:
        canvas = _label(canvas, "naive direct-on-snow segmentation")
    return canvas


def _compose_input_panel(r: FrameResult, *, with_label: bool = True) -> np.ndarray:
    canvas = r.snow_image.copy()
    if with_label:
        canvas = _label(canvas, "snow query (live frame)")
    return canvas


def _compose_summer_panel(summer_image: np.ndarray, summer_mask: np.ndarray,
                          *, with_label: bool = True) -> np.ndarray:
    canvas = summer_image.copy()
    if summer_mask is not None and int(summer_mask.sum()) > 0:
        canvas = alpha_blend(canvas, summer_mask, color=GREEN, alpha=0.45)
    if with_label:
        canvas = _label(canvas, "summer prior + road (Cityscapes)")
    return canvas


def render_overlay(
    results: list[FrameResult],
    track_id: str,
    *,
    out_name: str = "overlay_v0.mp4",
    fps: float = 10.0,
    keep_frames: bool = False,
    debug_strip: bool = False,
) -> Path:
    out_dir = OUT / track_id
    frame_dir = out_dir / "frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render] writing {len(results)} frames to {frame_dir}")
    for i, r in enumerate(results):
        canvas = _compose_overlay_panel(r, with_label=False)
        if debug_strip:
            # Tiny diagnostic strip (top-left): frame index + priors_used.
            label = f"{i + 1}/{len(results)}  priors={r.n_priors_used}"
            cv2.putText(canvas, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(frame_dir / f"f{i:04d}.png"),
                    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    out_path = out_dir / out_name
    _ffmpeg_concat(frame_dir, out_path, fps)
    if not keep_frames:
        shutil.rmtree(frame_dir)
    return out_path


def _ffmpeg_concat(frame_dir: Path, out_path: Path, fps: float) -> None:
    """ffmpeg pattern that survives odd dimensions + libx264/yuv420p."""
    print(f"[render] ffmpeg → {out_path} @ {fps} fps")
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


def render_sidebyside(
    results: list[FrameResult],
    track_id: str,
    *,
    out_name: str = "sidebyside.mp4",
    fps: float = 10.0,
    keep_frames: bool = False,
    label_panels: bool = False,
) -> Path:
    """Snow input | overlay output, two panels lockstepped.

    Both panels share the same height. The combined frame is hstacked.
    """
    out_dir = OUT / track_id
    frame_dir = out_dir / "frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render-sidebyside] writing {len(results)} frames")
    for i, r in enumerate(results):
        left = _compose_input_panel(r, with_label=label_panels)
        right = _compose_overlay_panel(r, with_label=label_panels)
        target_h = min(left.shape[0], right.shape[0])
        left = _resize_to_height(left, target_h)
        right = _resize_to_height(right, target_h)
        canvas = np.hstack([left, right])
        cv2.imwrite(str(frame_dir / f"f{i:04d}.png"),
                    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    out_path = out_dir / out_name
    _ffmpeg_concat(frame_dir, out_path, fps)
    if not keep_frames:
        shutil.rmtree(frame_dir)
    return out_path


def render_three_panel(
    results: list[FrameResult],
    track_id: str,
    naive_masks: list[np.ndarray | None],
    *,
    out_name: str = "three_panel.mp4",
    fps: float = 10.0,
    keep_frames: bool = False,
    label_panels: bool = True,
) -> Path:
    """Three-panel hstacked layout: input | naive (red) | overlay (green)."""
    out_dir = OUT / track_id
    frame_dir = out_dir / "frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render-three-panel] writing {len(results)} frames")
    for i, r in enumerate(results):
        target_h = r.snow_image.shape[0]
        snow = _resize_to_height(_compose_input_panel(r, with_label=label_panels), target_h)
        overlay = _resize_to_height(_compose_overlay_panel(r, with_label=label_panels), target_h)
        naive = _resize_to_height(_compose_naive_panel(r, naive_masks[i], with_label=label_panels), target_h)

        canvas = np.hstack([snow, naive, overlay])
        cv2.imwrite(str(frame_dir / f"f{i:04d}.png"),
                    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    out_path = out_dir / out_name
    _ffmpeg_concat(frame_dir, out_path, fps)
    if not keep_frames:
        shutil.rmtree(frame_dir)
    return out_path


def render_quad(
    results: list[FrameResult],
    track_id: str,
    summer_panels: list[dict | None],
    naive_masks: list[np.ndarray | None],
    *,
    out_name: str = "quad.mp4",
    fps: float = 10.0,
    keep_frames: bool = False,
    label_panels: bool = False,
) -> Path:
    """4-panel layout: snow query | naive (red) over snow / summer prior + road (green) | cross-season overlay (green).

    Reading order is the failure-then-recovery story:

        ┌─────────────┬─────────────┐
        │ snow query  │ naive (red) │   ← what arrives + what off-the-shelf
        │             │             │     segmentation says directly on snow
        ├─────────────┼─────────────┤
        │ clear prior │ overlay     │   ← what we know (same place, July)
        │ + green     │ + green     │     and the warp transferred onto snow
        └─────────────┴─────────────┘

    `summer_panels[i]` is a dict {image, road_mask, distance_m} for the
    closest summer prior at frame i, or None to fall back to a black panel.
    `naive_masks[i]` is the snow-direct Cityscapes mask (red overlay) or None.
    """
    out_dir = OUT / track_id
    frame_dir = out_dir / "frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render-quad] writing {len(results)} frames")
    for i, r in enumerate(results):
        target_h = r.snow_image.shape[0]
        p_snow = _compose_input_panel(r, with_label=label_panels)
        p_naive = _compose_naive_panel(r, naive_masks[i], with_label=label_panels)
        if summer_panels[i] is not None:
            sp = summer_panels[i]
            p_summer = _resize_to_height(
                _compose_summer_panel(sp["image"], sp["road_mask"], with_label=label_panels),
                target_h,
            )
        else:
            p_summer = np.zeros_like(r.snow_image)
        p_overlay = _compose_overlay_panel(r, with_label=label_panels)
        # 2x2 grid:
        #   top:    snow query   | naive (red)
        #   bottom: summer + road | cross-season overlay (green)
        top = np.hstack([p_snow, p_naive])
        bot = np.hstack([p_summer, p_overlay])
        # Equalise widths if the summer panel was a different aspect.
        if top.shape[1] != bot.shape[1]:
            tw, bw = top.shape[1], bot.shape[1]
            target = min(tw, bw)
            top = cv2.resize(top, (target, top.shape[0]), interpolation=cv2.INTER_AREA)
            bot = cv2.resize(bot, (target, bot.shape[0]), interpolation=cv2.INTER_AREA)
        canvas = np.vstack([top, bot])
        cv2.imwrite(str(frame_dir / f"f{i:04d}.png"),
                    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    out_path = out_dir / out_name
    _ffmpeg_concat(frame_dir, out_path, fps)
    if not keep_frames:
        shutil.rmtree(frame_dir)
    return out_path


def render_matches(
    results: list[FrameResult],
    track_id: str,
    matches_frames: list[dict | None],
    *,
    out_name: str = "matches.mp4",
    fps: float = 10.0,
    keep_frames: bool = False,
    max_inliers: int = 10,
) -> Path:
    """Snow | best-prior side-by-side with a subset of match lines drawn.

    `matches_frames[i]` is the per-frame entry from `_matches_<tag>.pkl`
    (kpts0, kpts1, inlier_mask, prior_image) or None if matching failed
    on that frame. Up to `max_inliers` correspondences are drawn per
    frame.
    """
    from src.matching import MatchResult, draw_matches

    out_dir = OUT / track_id
    frame_dir = out_dir / "frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render-matches] writing {len(results)} frames")
    last_canvas: np.ndarray | None = None
    for i, r in enumerate(results):
        m = matches_frames[i]
        if m is None:
            if last_canvas is not None:
                canvas = last_canvas
            else:
                canvas = np.hstack([r.snow_image, np.zeros_like(r.snow_image)])
        else:
            mr = MatchResult(kpts0=m["kpts0"], kpts1=m["kpts1"],
                             confidence=np.ones(len(m["kpts0"]), dtype=np.float32))
            canvas = draw_matches(
                r.snow_image, m["prior_image"], mr,
                inlier_mask=m["inlier_mask"],
                max_inliers=max_inliers,
            )
            last_canvas = canvas
        cv2.imwrite(str(frame_dir / f"f{i:04d}.png"),
                    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    out_path = out_dir / out_name
    _ffmpeg_concat(frame_dir, out_path, fps)
    if not keep_frames:
        shutil.rmtree(frame_dir)
    return out_path
