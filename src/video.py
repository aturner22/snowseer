"""Auto-rendered submission video.

Composes title cards + hero-panel scenes + dissolves into a ~3-min MP4.
Optional audio: if `assets/audio/music.mp3` (or `.wav`) and/or
`assets/audio/ambience.mp3` exist they are mixed under the captions; if
not, the video ships silent and the README documents the drop-in path.

The script is deterministic: same inputs, same outputs. The narrative beats
are defined in `SCENES` below; tweak there to re-cut.

Usage:
    uv run python -m src.video --out outputs/demo.mp4

Visual identity (charcoal · cream · rust) and typography (EB Garamond +
Inter + JetBrains Mono) follow `docs/style/style.md`.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

# Silence moviepy 'imageio' progress bars when run non-interactively.
os.environ.setdefault("IMAGEIO_FFMPEG_NO_PROGRESS", "1")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    afx,
    concatenate_videoclips,
    vfx,
)
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FONTS_DIR = ROOT / "assets/fonts"
AUDIO_DIR = ROOT / "assets/audio"
HEROES_DIR = ROOT / "outputs/heroes"

# Identity
BG = "#f6f3ee"
TEXT = "#1c1c1c"
ACCENT = "#b34a25"
MUTE = "#8a8780"

# Frame size: 1920x1080 (1080p) for upload clarity.
W, H = 1920, 1080
FPS = 30

# Register fonts so matplotlib renders title cards in the chosen typography.
for _f in FONTS_DIR.glob("*.ttf"):
    try:
        fm.fontManager.addfont(str(_f))
    except Exception:
        pass


@dataclass
class TitleCard:
    """A typography-only scene."""
    eyebrow: str = ""        # small uppercase label above the title
    title: str = ""          # large headline
    body: str = ""           # optional body sentence beneath
    subnote: str = ""        # optional small italic note at the bottom
    duration: float = 5.0    # seconds


@dataclass
class HeroScene:
    """A panel image with a narrative caption."""
    panel_filename: str      # filename in outputs/heroes/
    eyebrow: str = ""
    caption: str = ""
    duration: float = 5.0


SCENES: list[object] = [
    # 1 — opening hook
    TitleCard(
        eyebrow="SOTA COMMISSION I  ·  MINIMAL-SHOT AUTONOMY",
        title="Constants as the bridge",
        body="Minimal-shot autonomy, demonstrated on a snow plough.",
        duration=6.0,
    ),
    TitleCard(
        eyebrow="THE PROBLEM",
        title="A snow plough's job is short.",
        body="Keep the road clear. While the plough is doing it, the road is invisible.",
        duration=5.5,
    ),
    TitleCard(
        title="A self-driving stack trained on Cityscapes will report,\nwith calibrated confidence,\nthat the entire scene is sky.",
        duration=5.0,
    ),
    # 2 — the move
    TitleCard(
        eyebrow="THE MOVE",
        title="We are not going to label our way out.",
        body="27 million miles of road. The long tail of conditions any of them can be in is longer than the road itself.",
        duration=6.0,
    ),
    TitleCard(
        title="For every regime where autonomy fails for lack of data,\nthere is an adjacent regime where data exists,\nand where the parts that matter are the same.",
        duration=6.5,
    ),
    TitleCard(
        title="The plough's road is the same road it was last July.",
        body="The curb hasn't moved. The hydrant hasn't moved.\nThe road's appearance has changed completely; its position in space has not.",
        duration=6.5,
    ),
    TitleCard(
        eyebrow="THE PRINCIPLE",
        title="Constants as the bridge.",
        body="Identify what stays the same between the data-rich regime and the data-poor one. Transfer through the constants.",
        duration=6.5,
    ),
    # 3 — the architecture (one slide)
    TitleCard(
        eyebrow="THE EXAMPLE",
        title="Six steps.",
        body=(
            "1   Pull the live snowy frame.\n"
            "2   Pull a clear-season prior of the same coordinates.\n"
            "3   Match the two using a frozen feature matcher.\n"
            "4   Estimate a homography, biased toward the ground plane.\n"
            "5   Run a road segmenter on the clear prior — never on snow.\n"
            "6   Warp the road mask onto the snowy frame."
        ),
        duration=12.0,
    ),
    # 4 — heroes
    HeroScene(
        panel_filename="gallivare_se__1113124103239974__202392698419785__panel.png",
        eyebrow="HERO  ·  GÄLLIVARE",
        caption="Snow-banked road. The cleared lane is invisible to a model trained on dry asphalt. The cross-season overlay tracks it precisely.",
        duration=6.5,
    ),
    HeroScene(
        panel_filename="lulea_se__1235981388376274__771512076886521__panel.png",
        eyebrow="HERO  ·  LULEÅ",
        caption="A residential street, fully snow-covered. The clear prior knows where the road sits; the matcher anchors on the houses.",
        duration=6.5,
    ),
    HeroScene(
        panel_filename="gallivare_se__724743419870843__1232870027145826__panel.png",
        eyebrow="HERO  ·  GÄLLIVARE",
        caption="Direct front-of-camera view, road buried in snow. The overlay holds.",
        duration=6.5,
    ),
    HeroScene(
        panel_filename="kiruna_se__173943764513956__2572648156371424__panel.png",
        eyebrow="HERO  ·  KIRUNA",
        caption="Falun-red houses. The matcher anchors on the buildings; the road is recovered through the homography.",
        duration=6.5,
    ),
    # 5 — generalising
    TitleCard(
        eyebrow="THE STRUCTURE",
        title="A model trained on regime A.\nAn inference target in regime B.\nA known correspondence between the two.",
        body="Snow on a road is one instance.",
        duration=8.0,
    ),
    TitleCard(
        title="Low-light medical imaging without low-light training data.\nPolar earth observation without polar training data.\nA manipulator on Mars without Mars training data.",
        body="Each admits the same structure.",
        duration=8.0,
    ),
    # 6 — close
    TitleCard(
        title="Constants as the bridge.",
        body="Find what stays the same and walk across.",
        duration=6.0,
    ),
    TitleCard(
        eyebrow="REPRODUCIBLE FROM A CLEAN CLONE  ·  uv run make demo",
        title="Snow-Underlay",
        subnote="Submission to SoTA Commission I — Minimal-Shot Autonomy. May 2026.",
        duration=5.5,
    ),
]


# ─── Frame renderers ─────────────────────────────────────────────────────────


def _new_canvas() -> tuple[plt.Figure, plt.Axes]:
    """Create a 16:9 canvas in identity colours."""
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_facecolor(BG)
    ax.set_axis_off()
    return fig, ax


def _render_title_card(card: TitleCard, out_path: Path) -> Path:
    fig, ax = _new_canvas()
    # Eyebrow (small uppercase, sans, muted)
    if card.eyebrow:
        ax.text(8, 88, card.eyebrow, fontfamily="Inter", fontsize=11,
                color=MUTE, fontweight=400)
        # rust hairline under eyebrow
        ax.plot([8, 14], [85.5, 85.5], color=ACCENT, linewidth=1.5)
    # Title (Inter, large)
    if card.title:
        # Auto-size based on length
        n_lines = card.title.count("\n") + 1
        size = 60 if n_lines == 1 and len(card.title) < 40 else 44 if n_lines <= 2 else 34
        ax.text(8, 76 if card.eyebrow else 82, card.title, fontfamily="Inter",
                fontsize=size, color=TEXT, fontweight=500, va="top",
                linespacing=1.15)
    # Body (Garamond)
    if card.body:
        ax.text(8, 32, card.body, fontfamily="EB Garamond", fontsize=24,
                color=TEXT, va="top", linespacing=1.4)
    # Subnote (tiny italic)
    if card.subnote:
        ax.text(8, 8, card.subnote, fontfamily="EB Garamond", fontsize=14,
                color=MUTE, va="bottom", style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor=BG)
    plt.close(fig)
    return out_path


def _render_hero_scene(scene: HeroScene, out_path: Path) -> Path:
    panel_path = HEROES_DIR / scene.panel_filename
    if not panel_path.exists():
        # Fallback to a title card noting the missing asset.
        return _render_title_card(
            TitleCard(eyebrow=scene.eyebrow, title="(missing panel)",
                      subnote=str(panel_path), duration=scene.duration),
            out_path,
        )
    fig, ax = _new_canvas()
    img = np.array(Image.open(panel_path).convert("RGB"))
    ih, iw = img.shape[:2]

    # Reserve top 18% for eyebrow + caption; bottom 0; image fills the rest.
    img_left, img_right = 6, 94
    img_top = 16
    img_bottom = 95

    # Compute image rect to fit aspect.
    img_w_pct = img_right - img_left
    img_h_pct = img_bottom - img_top
    panel_aspect = iw / ih
    canvas_aspect = (W * img_w_pct / 100) / (H * img_h_pct / 100)
    if panel_aspect > canvas_aspect:
        # Image is wider — fit to width
        scale = (W * img_w_pct / 100) / iw
    else:
        scale = (H * img_h_pct / 100) / ih
    rendered_w_px = iw * scale
    rendered_h_px = ih * scale
    rendered_w_pct = rendered_w_px / W * 100
    rendered_h_pct = rendered_h_px / H * 100
    cx = (img_left + img_right) / 2
    img_x_left = cx - rendered_w_pct / 2
    img_y_top = (img_top + img_bottom) / 2 - rendered_h_pct / 2

    ax.imshow(img, extent=[img_x_left, img_x_left + rendered_w_pct,
                            100 - (img_y_top + rendered_h_pct), 100 - img_y_top],
              aspect="auto", interpolation="bilinear")

    # Eyebrow + accent rule
    if scene.eyebrow:
        ax.text(6, 96, scene.eyebrow, fontfamily="Inter", fontsize=11,
                color=MUTE, fontweight=400, va="top")
        ax.plot([6, 12], [94.5, 94.5], color=ACCENT, linewidth=1.5)

    # Caption
    if scene.caption:
        # Word-wrap caption to ~110 chars per line.
        ax.text(6, 92, scene.caption, fontfamily="EB Garamond", fontsize=20,
                color=TEXT, va="top", wrap=True, linespacing=1.35)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor=BG)
    plt.close(fig)
    return out_path


# ─── Video assembly ──────────────────────────────────────────────────────────


def _scene_clip(scene: object, frame_path: Path):
    duration = getattr(scene, "duration", 5.0)
    clip = ImageClip(str(frame_path)).with_duration(duration)
    # Soft fade in/out for the dissolve feel
    fade = min(0.6, duration / 6)
    return clip.with_effects([vfx.FadeIn(fade), vfx.FadeOut(fade)])


def _audio_track(total_duration: float):
    """Mix music + ambience under the visuals, if files are present."""
    pieces = []
    music_files = list(AUDIO_DIR.glob("music.*"))
    ambience_files = list(AUDIO_DIR.glob("ambience.*"))
    for mf in music_files:
        try:
            ac = AudioFileClip(str(mf))
            # Loop or trim to fit total_duration
            ac = ac.with_effects([afx.AudioLoop(duration=total_duration)])
            ac = ac.with_effects([afx.MultiplyVolume(0.55)])
            pieces.append(ac)
        except Exception as e:
            print(f"  ! could not load music {mf.name}: {e}")
    for af in ambience_files:
        try:
            ac = AudioFileClip(str(af))
            ac = ac.with_effects([afx.AudioLoop(duration=total_duration)])
            ac = ac.with_effects([afx.MultiplyVolume(0.30)])
            pieces.append(ac)
        except Exception as e:
            print(f"  ! could not load ambience {af.name}: {e}")
    if not pieces:
        return None
    return CompositeAudioClip(pieces).with_duration(total_duration)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/demo.mp4")
    ap.add_argument("--cache", default="outputs/_video_frames")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clips = []
    print(f"rendering {len(SCENES)} scenes...")
    for i, scene in enumerate(SCENES):
        frame_path = cache / f"scene_{i:02d}.png"
        if isinstance(scene, TitleCard):
            _render_title_card(scene, frame_path)
        elif isinstance(scene, HeroScene):
            _render_hero_scene(scene, frame_path)
        else:
            raise TypeError(f"unknown scene type: {type(scene)}")
        clips.append(_scene_clip(scene, frame_path))

    video = concatenate_videoclips(clips, method="chain")
    total = video.duration
    audio = _audio_track(total)
    if audio is not None:
        video = video.with_audio(audio)
        print(f"audio: {AUDIO_DIR}/* mixed under {total:.1f}s of video")
    else:
        print("audio: silent (drop assets/audio/music.* and/or ambience.* for a music bed)")

    print(f"writing {out_path} ({total:.1f}s, {FPS} fps)...")
    video.write_videofile(str(out_path), fps=FPS, codec="libx264",
                          audio_codec="aac" if audio is not None else None,
                          preset="medium", logger=None)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
