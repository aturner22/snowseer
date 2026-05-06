"""Compose the canonical clip with title cards + Bensound music bed.

Takes an existing overlay mp4 from `outputs/video/<track>/<clip>.mp4` and
wraps it with:
  - A 3 s opening title card (`Constants as the bridge — in motion`)
  - The clip itself (typically 10–30 s)
  - A 3 s closing card (`Find what stays the same and walk across`)
  - Bensound `Slow Motion` music bed under everything, faded in/out

Title cards are rendered from matplotlib using the same charcoal / cream /
rust palette as the static demo (`docs/style/style.md`).

Usage:
    uv run python -m src.video_runtime.compose_final \
        --track boreas_2021_01_26 --clip overlay_canonical_ema04.mp4 \
        --out canonical_final.mp4

Falls back gracefully (no audio if music.mp3 missing).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FONTS_DIR = ROOT / "assets/fonts"
AUDIO = ROOT / "assets/audio/music.mp3"
OUT = ROOT / "outputs/video"

BG = "#f6f3ee"
TEXT = "#1c1c1c"
ACCENT = "#b34a25"
MUTE = "#8a8780"


def _load_fonts() -> None:
    if FONTS_DIR.exists():
        for f in FONTS_DIR.glob("*.ttf"):
            try:
                fm.fontManager.addfont(str(f))
            except Exception:
                pass


def _render_title_card(
    eyebrow: str,
    title: str,
    subnote: str = "",
    *,
    width: int,
    height: int,
    out_path: Path,
) -> Path:
    """Render a single title card PNG matching the static demo style.

    Sizes scale with the canvas width so the same code works for 1024px
    Boreas frames and 1920px high-resolution exports.
    """
    _load_fonts()
    scale = width / 1920  # base sizes are calibrated for 1920px canvases

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_facecolor(BG); ax.set_axis_off()

    if eyebrow:
        # Letter-tracked eyebrow with single-space spacing (was double — too wide).
        spaced = " ".join(list(eyebrow))
        ax.text(8, 86, spaced, fontfamily="Inter", fontsize=int(28 * scale),
                color=MUTE, fontweight=400, va="top")
        ax.plot([8, 12], [83.5, 83.5], color=ACCENT, linewidth=2.0 * scale + 1)

    # Title font sized so a typical 18-char line fits to ~80% width.
    title_size = int(96 * scale)
    ax.text(8, 70, title, fontfamily="Inter", fontsize=title_size,
            color=TEXT, fontweight=500, va="top", linespacing=1.15)

    if subnote:
        ax.text(8, 12, subnote, fontfamily="EB Garamond",
                fontsize=int(36 * scale), color=MUTE, va="bottom", style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor=BG)
    plt.close(fig)
    return out_path


def _ffprobe_dims(mp4: Path) -> tuple[int, int]:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(mp4),
    ])
    w, h = (int(x) for x in out.decode().strip().split(","))
    return w, h


def _ffprobe_duration(mp4: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(mp4),
    ])
    return float(out.decode().strip())


def compose(track: str, clip_name: str, out_name: str = "canonical_final.mp4",
            opening_card: bool = True, closing_card: bool = True,
            card_seconds: float = 3.0) -> Path:
    track_dir = OUT / track
    clip = track_dir / clip_name
    if not clip.exists():
        raise SystemExit(f"clip not found: {clip}")
    out_path = track_dir / out_name
    work = track_dir / "_compose_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    width, height = _ffprobe_dims(clip)
    print(f"[compose] clip={clip.name}  {width}x{height}")

    # Render title cards as 1-frame mp4s of `card_seconds` each.
    card_paths: list[Path] = []
    if opening_card:
        png = _render_title_card(
            "Snow-Underlay",
            "In motion.",
            "Cross-season road overlay, frame by frame.\n"
            "Boreas (UTIAS), 2021-01-26 — Toronto, Canada.",
            width=width, height=height, out_path=work / "opening.png",
        )
        card_paths.append(_card_to_mp4(png, work / "opening.mp4", card_seconds))
    card_paths.append(clip)
    if closing_card:
        png = _render_title_card(
            "Constants as the bridge",
            "Find what stays the same\nand walk across.",
            "Snow-Underlay  ·  SoTA Commission I  ·  May 2026",
            width=width, height=height, out_path=work / "closing.png",
        )
        card_paths.append(_card_to_mp4(png, work / "closing.mp4", card_seconds))

    # ffmpeg concat filter (robust to fps/codec mismatches between cards
    # and clip, since we may render cards at 30 fps but the clip is at 10 fps).
    silent_path = work / "silent.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in card_paths:
        cmd.extend(["-i", str(p)])
    n = len(card_paths)
    streams = "".join(f"[{i}:v]" for i in range(n))
    filter_arg = (f"{streams}concat=n={n}:v=1:a=0[outv];"
                  f"[outv]fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2[v]")
    cmd.extend([
        "-filter_complex", filter_arg,
        "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(silent_path),
    ])
    subprocess.run(cmd, check=True)

    # Add music bed + faststart. Fade-out timing is computed from the
    # silent video's actual duration so it lands on the closing card.
    if AUDIO.exists():
        dur = _ffprobe_duration(silent_path)
        fade_out_dur = 1.8
        fade_out_start = max(0.0, dur - fade_out_dur)
        print(f"[compose] mixing {AUDIO.name}  (clip {dur:.1f}s, fade-out {fade_out_start:.1f}s)")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(silent_path),
            "-stream_loop", "-1", "-i", str(AUDIO),
            "-filter_complex",
            f"[1:a]volume=0.45,afade=t=in:st=0:d=1.5,"
            f"afade=t=out:st={fade_out_start:.2f}:d={fade_out_dur:.2f}[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-movflags", "+faststart",
            str(out_path),
        ], check=True)
    else:
        print("[compose] no music asset; output will be silent")
        shutil.move(silent_path, out_path)

    shutil.rmtree(work)
    print(f"[compose] wrote {out_path}")
    return out_path


def _card_to_mp4(png: Path, mp4: Path, seconds: float) -> Path:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", "30",
        "-t", f"{seconds}",
        "-i", str(png),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", "30",
        str(mp4),
    ], check=True)
    return mp4


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--clip", required=True, help="filename inside outputs/video/<track>/")
    p.add_argument("--out", default="canonical_final.mp4")
    p.add_argument("--no-opening", action="store_true")
    p.add_argument("--no-closing", action="store_true")
    p.add_argument("--card-seconds", type=float, default=3.0)
    args = p.parse_args()

    compose(args.track, args.clip, out_name=args.out,
            opening_card=not args.no_opening,
            closing_card=not args.no_closing,
            card_seconds=args.card_seconds)


if __name__ == "__main__":
    main()
