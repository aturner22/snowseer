"""Auto-rendered submission video.

Composes the narrative as a sequence of typographic title cards, principle
diagrams, and per-hero slide pairs. ~3 min, 1080p, 30 fps.

Per-hero structure (the user's requested layout):
  · Slide A — 1x2: snow query  ·  naive direct on snow (red, the failure).
  · Slide B — top row spans both columns with the matches viz (legend
    explained); bottom row is clear-prior+road (green) and cross-season
    overlay (green).

Pacing is information-density-aware: title cards 5-7 s, text-heavy slides
10-12 s, hero slides 8-12 s. Audio: any music.* and ambience.* in
assets/audio/ are looped to the video duration and mixed under captions.

Visual identity (charcoal · cream · rust; EB Garamond + Inter +
JetBrains Mono) follows docs/style/style.md.

Usage:
    uv run python -m src.video --out outputs/demo.mp4
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("IMAGEIO_FFMPEG_NO_PROGRESS", "1")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
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
PAIRS_DIR = ROOT / "data/pairs"
CURATED = ROOT / "data/curated_pairs.json"

BG = "#f6f3ee"
TEXT = "#1c1c1c"
ACCENT = "#b34a25"
MUTE = "#8a8780"
GREEN = "#2e9c56"
RED = "#dc3c32"

W, H = 1920, 1080
FPS = 30

for _f in FONTS_DIR.glob("*.ttf"):
    try:
        fm.fontManager.addfont(str(_f))
    except Exception:
        pass


# ─── Scene data ──────────────────────────────────────────────────────────────


@dataclass
class TitleCard:
    eyebrow: str = ""
    title: str = ""
    body: str = ""
    subnote: str = ""
    duration: float = 6.0


@dataclass
class DiagramCard:
    """A slide whose main content is a procedural diagram drawn by `kind`."""
    kind: str               # 'bridge' | 'pipeline'
    eyebrow: str = ""
    title: str = ""
    caption: str = ""
    duration: float = 10.0


@dataclass
class HeroSlideA:
    """Problem framing for one pair: snow query | naive."""
    pair_id: str
    duration: float = 8.0


@dataclass
class HeroSlideB:
    """Solution for one pair: matches above; clear+mask | cross-season below."""
    pair_id: str
    duration: float = 11.0


# ─── Curated pair display strings ────────────────────────────────────────────


def _curated_entry(pair_id: str) -> dict:
    if not CURATED.exists():
        return {"pair_id": pair_id, "place": "", "condition": "", "snow_captured": "", "clear_captured": ""}
    spec = json.loads(CURATED.read_text())
    for p in spec.get("pairs", []):
        if p.get("pair_id") == pair_id:
            return p
    return {"pair_id": pair_id, "place": "", "condition": "", "snow_captured": "", "clear_captured": ""}


def _hero_strings(pair_id: str) -> tuple[str, str]:
    e = _curated_entry(pair_id)
    place = e.get("place") or ""
    cond = e.get("condition") or ""
    snow_t = e.get("snow_captured") or ""
    clear_t = e.get("clear_captured") or ""
    title = f"{place} — {cond}" if cond else place
    sub = f"{snow_t}  ↔  {clear_t}" if snow_t and clear_t else ""
    return title, sub


# ─── Frame renderers ─────────────────────────────────────────────────────────


def _new_canvas() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_facecolor(BG); ax.set_axis_off()
    return fig, ax


def _eyebrow(ax, x, y, text):
    if not text:
        return
    # Letter-tracking via inserting spaces between characters; matplotlib's
    # Text artist doesn't expose `letterspacing`.
    spaced = "  ".join(list(text)) if len(text) <= 60 else " ".join(list(text))
    ax.text(x, y, spaced, fontfamily="Inter", fontsize=18,
            color=MUTE, fontweight=400, va="top")
    ax.plot([x, x + 6], [y - 1.6, y - 1.6], color=ACCENT, linewidth=2.0)


def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def _render_title_card(card: TitleCard, out: Path) -> Path:
    fig, ax = _new_canvas()
    _eyebrow(ax, 8, 92, card.eyebrow)
    if card.title:
        n_lines = card.title.count("\n") + 1
        size = 70 if n_lines == 1 and len(card.title) < 38 else 50 if n_lines <= 2 else 40
        ax.text(8, 80, card.title, fontfamily="Inter", fontsize=size,
                color=TEXT, fontweight=500, va="top", linespacing=1.2)
    if card.body:
        # Body sized so the slide can be read in 6-10 s.
        ax.text(8, 38, card.body, fontfamily="EB Garamond",
                fontsize=28, color=TEXT, va="top", linespacing=1.4)
    if card.subnote:
        ax.text(8, 7, card.subnote, fontfamily="EB Garamond",
                fontsize=18, color=MUTE, va="bottom", style="italic")
    _save(fig, out)
    return out


def _draw_bridge_diagram(ax):
    """Two regime boxes connected by a 'constant' bridge."""
    # Domain A — data-rich
    a = FancyBboxPatch((4, 22), 28, 38, boxstyle="round,pad=0.3",
                       linewidth=1.2, edgecolor=TEXT, facecolor="white")
    ax.add_patch(a)
    ax.text(18, 55, "Regime A", fontfamily="Inter", fontweight=700, fontsize=28,
            ha="center", color=TEXT)
    ax.text(18, 49, "data-rich", fontfamily="Inter", fontsize=20,
            ha="center", color=MUTE)
    ax.text(18, 38, "summer asphalt\nlane markings\ndaytime",
            fontfamily="EB Garamond", fontsize=22, ha="center", color=TEXT, linespacing=1.45)

    # Domain B — data-poor
    b = FancyBboxPatch((68, 22), 28, 38, boxstyle="round,pad=0.3",
                       linewidth=1.2, edgecolor=TEXT, facecolor="white")
    ax.add_patch(b)
    ax.text(82, 55, "Regime B", fontfamily="Inter", fontweight=700, fontsize=28,
            ha="center", color=TEXT)
    ax.text(82, 49, "data-poor", fontfamily="Inter", fontsize=20,
            ha="center", color=MUTE)
    ax.text(82, 38, "snow-covered road\nlane markings hidden\nthe regime that fails",
            fontfamily="EB Garamond", fontsize=22, ha="center", color=TEXT, linespacing=1.45)

    # Arrow / bridge
    arrow = FancyArrowPatch((33, 42), (67, 42), arrowstyle="-|>", mutation_scale=24,
                            color=ACCENT, linewidth=3.0, zorder=4)
    ax.add_patch(arrow)
    ax.text(50, 48, "the constant", fontfamily="Inter", fontweight=700,
            fontsize=24, ha="center", color=ACCENT)
    ax.text(50, 36.5, "(buildings, signs, road position)", fontfamily="EB Garamond",
            fontstyle="italic", fontsize=18, ha="center", color=MUTE)


def _draw_pipeline_diagram(ax):
    """Six-step horizontal flow, vertically centred in the canvas."""
    steps = [
        ("Snow query", "live frame"),
        ("Clear prior", "Mapillary, same coords"),
        ("Match", "DISK + LightGlue"),
        ("Align", "RANSAC homography"),
        ("Segment", "Mask2Former on prior only"),
        ("Warp", "road mask onto snow"),
    ]
    n = len(steps)
    box_w = 14
    box_h = 26
    spacing = 1.2
    total = n * box_w + (n - 1) * spacing
    x0 = (100 - total) / 2
    y_bottom = 44   # raised so the caption underneath has room without the diagram squatting low
    y_top = y_bottom + box_h
    arrow_y = y_bottom + box_h / 2

    for i, (head, sub) in enumerate(steps):
        x = x0 + i * (box_w + spacing)
        rect = FancyBboxPatch((x, y_bottom), box_w, box_h,
                              boxstyle="round,pad=0.15",
                              linewidth=1.0, edgecolor=TEXT, facecolor="white")
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y_top - 4, head,
                fontfamily="Inter", fontweight=700, fontsize=15,
                ha="center", va="top", color=TEXT)
        ax.text(x + box_w / 2, y_top - 10, sub,
                fontfamily="EB Garamond", fontstyle="italic", fontsize=12,
                ha="center", va="top", color=MUTE, linespacing=1.2)
        ax.text(x + box_w / 2, y_bottom + 4, str(i + 1),
                fontfamily="Inter", fontweight=700, fontsize=22,
                ha="center", color=ACCENT)
        if i < n - 1:
            ax.annotate("", xy=(x + box_w + spacing, arrow_y), xytext=(x + box_w, arrow_y),
                        arrowprops=dict(arrowstyle="-|>", mutation_scale=14,
                                        color=TEXT, linewidth=1.2))


def _render_diagram_card(card: DiagramCard, out: Path) -> Path:
    fig, ax = _new_canvas()
    _eyebrow(ax, 8, 92, card.eyebrow)
    if card.title:
        ax.text(8, 80, card.title, fontfamily="Inter", fontsize=50,
                color=TEXT, fontweight=500, va="top")
    if card.kind == "bridge":
        _draw_bridge_diagram(ax)
    elif card.kind == "pipeline":
        _draw_pipeline_diagram(ax)
    if card.caption:
        # Wrap caption so it doesn't run off the right edge.
        import textwrap
        wrapped = textwrap.fill(card.caption, width=110)
        ax.text(8, 22, wrapped, fontfamily="EB Garamond",
                fontsize=22, color=TEXT, va="top", linespacing=1.4)
    _save(fig, out)
    return out


def _annotate_image(ax, img: np.ndarray, label: str, *, accent: bool = False):
    """Show an image with a short Inter label above and (optional) accent frame."""
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(accent)
        if accent:
            s.set_color(ACCENT); s.set_linewidth(2.4)
    ax.set_title(label, fontfamily="Inter", fontsize=22, color=TEXT,
                 fontweight=500, pad=12, loc="left")


def _render_hero_slide_a(scene: HeroSlideA, out: Path) -> Path:
    pair_id = scene.pair_id
    snow_path = PAIRS_DIR / pair_id / "snow.jpg"
    naive_path = HEROES_DIR / f"{pair_id}__naive_baseline.png"
    if not snow_path.exists() or not naive_path.exists():
        return _render_title_card(
            TitleCard(title="(missing assets)", subnote=str(snow_path), duration=scene.duration), out)

    title, sub = _hero_strings(pair_id)
    fig = plt.figure(figsize=(W/100, H/100), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.18, 1.0], hspace=0.04, wspace=0.04,
                          left=0.04, right=0.96, top=0.97, bottom=0.04)
    head = fig.add_subplot(gs[0, :]); head.set_facecolor(BG); head.set_axis_off()
    head.text(0.0, 0.85, "THE PROBLEM  ·  SNOW QUERY VS NAIVE PREDICTION",
              fontfamily="Inter", fontsize=18, color=MUTE, ha="left", va="top",
              transform=head.transAxes)
    head.plot([0, 0.04], [0.65, 0.65], color=ACCENT, linewidth=2.0,
              transform=head.transAxes)
    head.text(0.0, 0.55, title, fontfamily="Inter", fontsize=34,
              fontweight=500, color=TEXT, ha="left", va="top",
              transform=head.transAxes)
    if sub:
        head.text(0.0, 0.10, sub, fontfamily="EB Garamond", fontsize=20,
                  color=MUTE, style="italic", ha="left", va="top",
                  transform=head.transAxes)

    ax_snow = fig.add_subplot(gs[1, 0])
    ax_naive = fig.add_subplot(gs[1, 1])
    snow = np.array(Image.open(snow_path).convert("RGB"))
    naive = np.array(Image.open(naive_path).convert("RGB"))
    _annotate_image(ax_snow, snow, "Snow query")
    _annotate_image(ax_naive, naive, "Naive direct on snow  (red = predicted road)")
    _save(fig, out)
    return out


def _render_hero_slide_b(scene: HeroSlideB, out: Path) -> Path:
    pair_id = scene.pair_id
    matches_path = HEROES_DIR / f"{pair_id}__matches.png"
    overlay_path = HEROES_DIR / f"{pair_id}__overlay.png"
    clear_path = PAIRS_DIR / pair_id / "clear.jpg"
    if not (matches_path.exists() and overlay_path.exists() and clear_path.exists()):
        return _render_title_card(
            TitleCard(title="(missing assets)", subnote=pair_id, duration=scene.duration), out)

    from .overlay import alpha_blend
    from .pipeline import _load_rgb, _resize_to
    clear = _resize_to(_load_rgb(clear_path))
    road_mask = _segment_for_display(clear)
    clear_with_mask = alpha_blend(clear, road_mask, color=(46, 156, 86), alpha=0.50)

    title, sub = _hero_strings(pair_id)
    fig = plt.figure(figsize=(W/100, H/100), dpi=100, facecolor=BG)
    # Header band gets ~22% of the figure so title + subtitle don't squish.
    gs = fig.add_gridspec(3, 2, height_ratios=[0.22, 0.45, 0.45], hspace=0.20, wspace=0.04,
                          left=0.04, right=0.96, top=0.97, bottom=0.03)
    head = fig.add_subplot(gs[0, :]); head.set_facecolor(BG); head.set_axis_off()
    head.text(0.0, 0.92, "THE SOLUTION  ·  MATCHES, MASK, OVERLAY",
              fontfamily="Inter", fontsize=20, color=MUTE, ha="left", va="top",
              transform=head.transAxes)
    head.plot([0, 0.05], [0.74, 0.74], color=ACCENT, linewidth=2.4,
              transform=head.transAxes)
    head.text(0.0, 0.62, title, fontfamily="Inter", fontsize=36,
              fontweight=500, color=TEXT, ha="left", va="top",
              transform=head.transAxes)
    if sub:
        head.text(0.0, 0.10, sub, fontfamily="EB Garamond", fontsize=22,
                  color=MUTE, style="italic", ha="left", va="top",
                  transform=head.transAxes)

    ax_matches = fig.add_subplot(gs[1, :])
    matches_img = np.array(Image.open(matches_path).convert("RGB"))
    _annotate_image(
        ax_matches, matches_img,
        "Feature correspondences  ·  green = RANSAC inlier  ·  red = rejected",
    )

    ax_clear = fig.add_subplot(gs[2, 0])
    ax_overlay = fig.add_subplot(gs[2, 1])
    overlay = np.array(Image.open(overlay_path).convert("RGB"))
    _annotate_image(ax_clear, clear_with_mask, "Clear prior  ·  road mask in green")
    _annotate_image(ax_overlay, overlay, "Cross-season overlay  ·  same road, transferred", accent=True)
    _save(fig, out)
    return out


_seg_cache = {"obj": None}


def _segment_for_display(rgb: np.ndarray) -> np.ndarray:
    if _seg_cache["obj"] is None:
        from .segmentation import RoadSegmenter
        from .overlay import keep_largest_component
        _seg_cache["obj"] = (RoadSegmenter(), keep_largest_component)
    seg, klc = _seg_cache["obj"]
    return klc(seg.segment_road(rgb))


# ─── Scene script ────────────────────────────────────────────────────────────


def _build_scenes() -> list[object]:
    spec = json.loads(CURATED.read_text()) if CURATED.exists() else {"pairs": []}
    hero_ids = [p["pair_id"] for p in spec.get("pairs", [])][:4]  # first four GREAT-rated
    scenes: list[object] = []

    # Open
    scenes.append(TitleCard(
        eyebrow="SOTA COMMISSION I  ·  MINIMAL-SHOT AUTONOMY",
        title="Constants as the bridge",
        body="A demonstration of cross-season visual prior transfer\nfor autonomous snow ploughs.",
        duration=7.0,
    ))

    # Problem
    scenes.append(TitleCard(
        eyebrow="THE PROBLEM",
        title="A snow plough's job is short:\nkeep the road clear.",
        body="While the plough is working, the road is invisible.\nCurbs are buried. Lane markings are gone. The seam between asphalt and garden is no longer drawn.",
        duration=10.0,
    ))
    scenes.append(TitleCard(
        eyebrow="THE GAP",
        title="Self-driving systems are trained on\ndry roads, deliberately.",
        body="Cityscapes, KITTI, nuScenes, Waymo Open — every canonical training corpus is dominated by clear-weather highways under daylight.\nA stack trained on this data, asked to operate when the road is buried, has been asked the wrong question.",
        duration=12.0,
    ))

    # Why not more data
    scenes.append(TitleCard(
        eyebrow="THE SCALING ARGUMENT",
        title="We are not going to label our way out.",
        body="27 million miles of road. The long tail of conditions any of them can be in is longer than the road itself.\nAnnotating snowy roads, dust storms, fog, washouts — none of these scale.",
        duration=11.0,
    ))

    # Principle (with diagram)
    scenes.append(DiagramCard(
        kind="bridge",
        eyebrow="THE PRINCIPLE",
        title="Find the constant.",
        caption="For every regime where autonomy fails for lack of data, there is an adjacent regime where data exists. Identify what stays the same — and transfer through the constant.",
        duration=12.0,
    ))

    scenes.append(TitleCard(
        eyebrow="THE EXAMPLE",
        title="The plough's road is the same road\nit was last July.",
        body="The curb hasn't moved. The hydrant hasn't moved.\nThe road's appearance has changed completely. Its position in space has not.",
        duration=10.0,
    ))

    # Architecture (with diagram)
    scenes.append(DiagramCard(
        kind="pipeline",
        eyebrow="THE ARCHITECTURE  ·  SIX STEPS",
        title="Snow query → cross-season overlay.",
        caption="Frozen pretrained components throughout. The segmenter is applied to the clear prior only — never to the snow frame. Snow appears at inference time as the runtime input.",
        duration=14.0,
    ))

    # Heroes — two slides each
    for pid in hero_ids:
        scenes.append(HeroSlideA(pair_id=pid, duration=8.0))
        scenes.append(HeroSlideB(pair_id=pid, duration=12.0))

    # Generalising
    scenes.append(TitleCard(
        eyebrow="THE STRUCTURE GENERALISES",
        title="Where the data is missing,\nfind the regime where it isn't.",
        body="Low-light medical imaging without low-light training data.\nPolar earth observation without polar training data.\nA manipulator on Mars without Mars training data.\nEach admits the same structure.",
        duration=12.0,
    ))

    # Close
    scenes.append(TitleCard(
        eyebrow="REPRODUCIBLE  ·  uv run make demo",
        title="Snow-Underlay",
        subnote="Submission to SoTA Commission I — Minimal-Shot Autonomy. May 2026.",
        duration=6.5,
    ))
    return scenes


# ─── Video assembly ──────────────────────────────────────────────────────────


def _scene_clip(scene: object, frame_path: Path):
    duration = getattr(scene, "duration", 6.0)
    clip = ImageClip(str(frame_path)).with_duration(duration)
    fade = min(0.7, duration / 7)
    return clip.with_effects([vfx.FadeIn(fade), vfx.FadeOut(fade)])


def _audio_track(total_duration: float):
    pieces = []
    for mf in AUDIO_DIR.glob("music.*"):
        try:
            ac = AudioFileClip(str(mf))
            ac = ac.with_effects([afx.AudioLoop(duration=total_duration)])
            # Bumped to 0.85 (was 0.55) — earlier render was too quiet for some players.
            ac = ac.with_effects([afx.MultiplyVolume(0.85)])
            pieces.append(ac)
        except Exception as e:
            print(f"  ! music {mf.name}: {e}")
    for af in AUDIO_DIR.glob("ambience.*"):
        try:
            ac = AudioFileClip(str(af))
            ac = ac.with_effects([afx.AudioLoop(duration=total_duration)])
            ac = ac.with_effects([afx.MultiplyVolume(0.45)])
            pieces.append(ac)
        except Exception as e:
            print(f"  ! ambience {af.name}: {e}")
    if not pieces:
        return None
    return CompositeAudioClip(pieces).with_duration(total_duration)


def _render_scene(scene: object, frame_path: Path) -> Path:
    if isinstance(scene, TitleCard):
        return _render_title_card(scene, frame_path)
    if isinstance(scene, DiagramCard):
        return _render_diagram_card(scene, frame_path)
    if isinstance(scene, HeroSlideA):
        return _render_hero_slide_a(scene, frame_path)
    if isinstance(scene, HeroSlideB):
        return _render_hero_slide_b(scene, frame_path)
    raise TypeError(f"unknown scene: {type(scene)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/demo.mp4")
    ap.add_argument("--cache", default="outputs/_video_frames")
    args = ap.parse_args()

    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    scenes = _build_scenes()
    print(f"rendering {len(scenes)} scenes...")
    clips = []
    for i, scene in enumerate(scenes):
        frame_path = cache / f"scene_{i:02d}.png"
        _render_scene(scene, frame_path)
        clips.append(_scene_clip(scene, frame_path))

    video = concatenate_videoclips(clips, method="chain")
    total = video.duration
    audio = _audio_track(total)
    if audio is not None:
        video = video.with_audio(audio)
        print(f"audio: mixed under {total:.1f}s of video")
    else:
        print("audio: silent (drop assets/audio/music.* or ambience.* for a music bed)")

    print(f"writing {out_path} ({total:.1f}s)...")
    video.write_videofile(str(out_path), fps=FPS, codec="libx264",
                          audio_codec="aac" if audio is not None else None,
                          preset="medium", logger=None)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
