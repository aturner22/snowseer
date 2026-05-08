"""Track loader for Phase K — yields aligned (snow_frame, gps_pose) tuples.

A 'track' is a directory of `data/video/tracks/<track_id>/{snow,summer}/`
that's already been populated by `src.video_runtime.fetch_track`. The Track
class indexes the on-disk frames + camera_poses.csv so the per-frame pipeline
can iterate over the snow frames in capture order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRACKS_DIR = ROOT / "data/video/tracks"


@dataclass
class FrameMeta:
    idx: int                  # local index in the snow window (0..N)
    seq_idx: int              # absolute index into the full sequence's camera_poses.csv
    gpstime: int              # microsecond timestamp (matches the frame filename)
    easting: float
    northing: float
    heading: float
    path: Path                # absolute path to the PNG


def _load_camera_poses(p: Path) -> np.ndarray:
    return np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding="utf-8")


class Track:
    """Indexed view of one track's snow frames + summer prior pool.

    Attributes:
        track_id: e.g. 'boreas_2021_01_26'
        track_dir: <root>/data/video/tracks/<track_id>
        snow_meta: list[FrameMeta] for each snow frame in window order
        summer_meta: list[FrameMeta] for each summer frame in window order
    """

    def __init__(self, track_id: str):
        self.track_id = track_id
        self.track_dir = TRACKS_DIR / track_id
        if not self.track_dir.exists():
            raise FileNotFoundError(
                f"Track {track_id} not found at {self.track_dir}. "
                f"Run `make video-fetch TRACK={track_id}` first."
            )
        self.track_meta = json.loads((self.track_dir / "track.json").read_text())
        self.snow_meta = self._build_meta(self.track_dir / "snow")
        self.summer_meta = self._build_meta(self.track_dir / "summer")

    def _build_meta(self, half_dir: Path) -> list[FrameMeta]:
        window = json.loads((half_dir / "window.json").read_text())
        poses = _load_camera_poses(half_dir / "camera_poses.csv")
        seq_indices = list(range(window["indices"][0], window["indices"][1]))
        out: list[FrameMeta] = []
        for local_idx, seq_idx in enumerate(seq_indices):
            ts = int(poses["GPSTime"][seq_idx])
            path = half_dir / "frames" / f"{ts}.png"
            if not path.exists():
                # Frame may be missing if the download was capped. Skip it.
                continue
            out.append(FrameMeta(
                idx=local_idx,
                seq_idx=seq_idx,
                gpstime=ts,
                easting=float(poses["easting"][seq_idx]),
                northing=float(poses["northing"][seq_idx]),
                heading=float(poses["heading"][seq_idx]),
                path=path,
            ))
        return out

    def snow_frame_count(self) -> int:
        return len(self.snow_meta)

    def load_frame(self, meta: FrameMeta, max_dim: int | None = None) -> np.ndarray:
        """Load a frame, optionally resizing the long edge to `max_dim`."""
        img = cv2.imread(str(meta.path))
        if img is None:
            raise FileNotFoundError(f"Frame missing: {meta.path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if max_dim is not None:
            h, w = img.shape[:2]
            s = max_dim / max(h, w)
            if s < 1.0:
                img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                                 interpolation=cv2.INTER_AREA)
        return img

    def iter_snow(self, start: int = 0, end: int | None = None,
                  stride: int = 1, max_dim: int | None = None):
        """Yield (FrameMeta, np.ndarray) tuples for the snow stream."""
        end = end or len(self.snow_meta)
        for i in range(start, end, stride):
            m = self.snow_meta[i]
            yield m, self.load_frame(m, max_dim=max_dim)
