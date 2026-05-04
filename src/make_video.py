"""Stitch hero outputs into a navigation traversal video.

Each frame is a snowy query | clear prior + road | snowy + warped overlay panel,
captioned with metadata. The result is what the brief calls a 'simulation
environment' for the agent: each frame is one timestep of the plough's traversal.

Usage:
    uv run python -m src.make_video --fps 1 --out outputs/demo.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("outputs/heroes")
PAIRS_DIR = Path("data/pairs")


def _caption(img: np.ndarray, text: str, *, where: str = "top") -> np.ndarray:
    h, w = img.shape[:2]
    bar_h = 56
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    cv2.putText(
        bar, text, (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA,
    )
    if where == "top":
        return np.vstack([bar, img])
    return np.vstack([img, bar])


def _annotate_panel(panel: np.ndarray, pair_id: str, meta: dict | None) -> np.ndarray:
    bits: list[str] = [pair_id]
    if meta:
        bits.append(f"region={meta.get('region')}")
        bits.append(f"dist={meta.get('distance_m')}m")
        bits.append(f"Δheading={meta.get('heading_delta_deg')}°")
    title = "  |  ".join(bits)
    return _caption(panel, title, where="top")


def _read_panel(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def build(out_path: Path, fps: float, target_height: int) -> Path:
    summary_path = OUT_DIR / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Run the pipeline first; {summary_path} not found.")
    summary = json.loads(summary_path.read_text())

    frames: list[np.ndarray] = []
    for entry in summary:
        pair_id = entry["pair_id"]
        panel_path = OUT_DIR / f"{pair_id}__panel.png"
        panel = _read_panel(panel_path)
        if panel is None:
            continue
        meta_path = PAIRS_DIR / pair_id / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        captioned = _annotate_panel(panel, pair_id, meta)
        # Resize to a consistent target height
        ch, cw = captioned.shape[:2]
        scale = target_height / ch
        new_w = int(round(cw * scale))
        resized = cv2.resize(captioned, (new_w, target_height), interpolation=cv2.INTER_AREA)
        frames.append(resized)

    if not frames:
        raise SystemExit("No panel images found to stitch into a video.")

    # Pad all frames to the maximum width so the writer accepts them.
    max_w = max(f.shape[1] for f in frames)
    padded: list[np.ndarray] = []
    for f in frames:
        h, w = f.shape[:2]
        if w == max_w:
            padded.append(f)
            continue
        pad = np.zeros((h, max_w - w, 3), dtype=np.uint8)
        padded.append(np.hstack([f, pad]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = padded[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"VideoWriter failed to open {out_path}")
    for f in padded:
        writer.write(f)
    writer.release()
    print(f"wrote {len(padded)} frames -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/demo.mp4")
    parser.add_argument("--fps", type=float, default=1.0, help="frames per second")
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    build(Path(args.out), fps=args.fps, target_height=args.height)


if __name__ == "__main__":
    main()
