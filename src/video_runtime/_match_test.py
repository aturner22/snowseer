"""Empirical match test for the snow↔summer Boreas pair.

Picks a few snow frames from the middle of the window, finds their
GPS-aligned summer counterparts, runs the existing Matcher + RANSAC,
and prints the inlier count per pair plus saves a side-by-side viz.

Pass condition: at least one pair returns >= 20 inliers and the
correspondence viz looks sensible. If not, the lens-water/feature-density
issue is fatal for this track and we switch sources.

Usage:
    uv run python -m src.video_runtime._match_test
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.matching import Matcher, draw_matches
from src.homography import estimate

ROOT = Path(__file__).resolve().parents[2]
TRACK = ROOT / "data/video/tracks/boreas_2021_01_26"
OUT = ROOT / "outputs/video/_match_test"


def _load_camera_poses(p: Path) -> np.ndarray:
    return np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding="utf-8")


def _resize(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_dim / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    snow_window = json.loads((TRACK / "snow/window.json").read_text())
    summer_window = json.loads((TRACK / "summer/window.json").read_text())
    snow_poses = _load_camera_poses(TRACK / "snow/camera_poses.csv")
    summer_poses = _load_camera_poses(TRACK / "summer/camera_poses.csv")

    snow_indices = list(range(snow_window["indices"][0], snow_window["indices"][1]))
    summer_indices = list(range(summer_window["indices"][0], summer_window["indices"][1]))

    # Build summer (x,y) array for nearest-neighbor lookup.
    summer_xy = np.stack([
        summer_poses["easting"][summer_indices],
        summer_poses["northing"][summer_indices],
    ], axis=1)

    # Sample 4 snow frames spread through the window: 50, 150, 250, 330.
    sample = [50, 150, 250, 330]

    matcher = Matcher()

    print(f"\n{'pair_id':<30s}  {'snow_t':>7s}  {'snow_pos':>16s}  {'sum_t':>7s}  "
          f"{'sum_pos':>16s}  {'dist_m':>7s}  {'matches':>8s}  {'inliers':>7s}")
    print("-" * 110)

    for k, snow_off in enumerate(sample):
        if snow_off >= len(snow_indices):
            continue
        snow_idx = snow_indices[snow_off]
        snow_x = float(snow_poses["easting"][snow_idx])
        snow_y = float(snow_poses["northing"][snow_idx])

        # Nearest summer frame in (x, y).
        d2 = np.sum((summer_xy - np.array([snow_x, snow_y])) ** 2, axis=1)
        sum_off = int(np.argmin(d2))
        dist = float(np.sqrt(d2[sum_off]))
        sum_idx = summer_indices[sum_off]
        sum_x = float(summer_poses["easting"][sum_idx])
        sum_y = float(summer_poses["northing"][sum_idx])

        snow_ts = int(snow_poses["GPSTime"][snow_idx])
        summer_ts = int(summer_poses["GPSTime"][sum_idx])
        snow_path = TRACK / "snow/frames" / f"{snow_ts}.png"
        summer_path = TRACK / "summer/frames" / f"{summer_ts}.png"
        if not (snow_path.exists() and summer_path.exists()):
            print(f"  {k}: missing files ({snow_path.exists()=}, {summer_path.exists()=})")
            continue

        # Load + resize.
        snow_img = cv2.cvtColor(cv2.imread(str(snow_path)), cv2.COLOR_BGR2RGB)
        summer_img = cv2.cvtColor(cv2.imread(str(summer_path)), cv2.COLOR_BGR2RGB)
        snow_img = _resize(snow_img, 1024)
        summer_img = _resize(summer_img, 1024)

        # Match.
        result = matcher.match(snow_img, summer_img)
        homo = estimate(result, snow_img.shape[:2], summer_img.shape[:2])
        n_matches = len(result.kpts0)
        n_inliers = int(np.sum(homo.inlier_mask)) if homo.inlier_mask is not None else 0

        print(f"  test_{k:02d}                       "
              f"{snow_off:>7d}  ({snow_x:+7.1f},{snow_y:+7.1f})  "
              f"{sum_off:>7d}  ({sum_x:+7.1f},{sum_y:+7.1f})  "
              f"{dist:>7.2f}  {n_matches:>8d}  {n_inliers:>7d}")

        # Save viz.
        viz_path = OUT / f"test_{k:02d}__snow{snow_off:04d}__summer{sum_off:04d}__matches.png"
        draw_matches(snow_img, summer_img, result, inlier_mask=homo.inlier_mask, out_path=viz_path)


if __name__ == "__main__":
    main()
