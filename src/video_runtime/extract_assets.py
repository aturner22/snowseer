"""Extract still frames from rendered overlay clips for use in the
submission video plan + slides + GitHub Pages.

Run after the renders are produced (overlay.mp4, sidebyside.mp4, etc.).
Writes JPEGs at preset timestamps to outputs/toronto_video/<track>/stills/.

The extracted stills are the *visual asset list* referenced by
docs/slides.md (the video script / storyboard).

Usage:
    uv run python -m src.video_runtime.extract_assets --track <track_id>
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/toronto_video"

# Timestamps (seconds) at which to extract a still from each rendered mp4.
# These fall on visually distinctive moments in the canonical 15 s clip:
#   1.0  — early frame, vehicle still in transitional area
#   5.0  — mid-clip, residential road with overlay clearly draped
#   10.0 — past midpoint, road curve visible
#   14.0 — late, near end of window
DEFAULT_TIMESTAMPS = (1.0, 5.0, 10.0, 14.0)


def _extract(mp4: Path, t: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{t}", "-i", str(mp4),
        "-frames:v", "1",
        "-q:v", "2",  # high-quality JPEG (1=highest, 31=lowest)
        str(out_path),
    ], check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--clip-dir", default=None,
                   help="directory containing rendered mp4s. Default: outputs/toronto_video/<track>/")
    p.add_argument("--timestamps", nargs="+", type=float, default=list(DEFAULT_TIMESTAMPS))
    args = p.parse_args()

    clip_dir = Path(args.clip_dir) if args.clip_dir else OUT / args.track
    if not clip_dir.exists():
        raise SystemExit(f"clip dir not found: {clip_dir}")

    stills_dir = clip_dir / "stills"
    if stills_dir.exists():
        shutil.rmtree(stills_dir)
    stills_dir.mkdir(parents=True, exist_ok=True)

    # Find all top-level .mp4s in the clip dir; skip ones inside _v1_canonical_K3/_ablation/_compose_work.
    mp4s = sorted([f for f in clip_dir.iterdir()
                   if f.is_file() and f.suffix == ".mp4"])
    if not mp4s:
        raise SystemExit(f"no mp4s found in {clip_dir}; render first via "
                         f"`make track TRACK={args.track}`")

    print(f"[extract] writing {len(mp4s) * len(args.timestamps)} stills to {stills_dir}")
    for mp4 in mp4s:
        for t in args.timestamps:
            stem = mp4.stem
            ts_label = f"{t:05.1f}".replace(".", "p")
            out = stills_dir / f"{stem}__t{ts_label}.jpg"
            try:
                _extract(mp4, t, out)
                print(f"  {out.name}")
            except subprocess.CalledProcessError as e:
                print(f"  [SKIP] {mp4.name} @ t={t}: {e}")

    print(f"\n[extract] {len(list(stills_dir.iterdir()))} stills under {stills_dir}")


if __name__ == "__main__":
    main()
