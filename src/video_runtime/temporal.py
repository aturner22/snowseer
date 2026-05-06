"""Temporal smoothing strategies for the per-frame video pipeline.

Three approaches, each implemented as a stateful object with one method
`smooth(raw_mask) -> mask` that returns a binary uint8 mask in snow-image
space, given the per-frame raw fusion output.

K.4.a — `FlowSmoother`        : Farneback flow propagation between keyframes
K.4.b — `EMASmoother`         : exponential moving average on the soft mask
K.4.c — `HomographySmoother`  : low-pass on the homography matrix elements
                                 (TODO if EMA + flow aren't enough)

All three share a common `Smoother` interface so `pipeline_v.run_track`
can swap strategies via a flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np


class Smoother(Protocol):
    """A temporal smoother takes the *raw* per-frame mask plus the snow
    image (for flow), and returns a smoothed binary mask in snow-image space."""

    def smooth(self, raw_mask: np.ndarray | None, snow_image: np.ndarray) -> np.ndarray | None:
        ...

    def reset(self) -> None:
        ...


# ─── EMA on the soft mask ───────────────────────────────────────────────────


@dataclass
class EMASmoother:
    """Exponential moving average on the binary mask, treated as float [0, 1].

    smooth_t = alpha * raw_t + (1 - alpha) * smooth_{t-1}

    `alpha=0.5` weights past and present equally; `alpha=0.7` favours present
    (less sticky, less smoothing); `alpha=0.3` favours past (very sticky,
    can lag the camera). Default 0.5.

    On a missing raw_mask (matcher failure for one frame), we return the
    last smoothed mask unchanged — the road doesn't disappear when one
    frame fails.

    Threshold at 0.5 to get the binary output.
    """
    alpha: float = 0.5
    _state: np.ndarray | None = field(default=None, init=False)

    def reset(self) -> None:
        self._state = None

    def smooth(self, raw_mask: np.ndarray | None, snow_image: np.ndarray) -> np.ndarray | None:
        if raw_mask is None:
            # Frame failed — propagate the last smoothed mask.
            if self._state is None:
                return None
            return (self._state >= 0.5).astype(np.uint8)
        raw_f = (raw_mask > 0).astype(np.float32)
        if self._state is None or self._state.shape != raw_f.shape:
            self._state = raw_f.copy()
        else:
            self._state = self.alpha * raw_f + (1.0 - self.alpha) * self._state
        return (self._state >= 0.5).astype(np.uint8)


# ─── Optical flow propagation ───────────────────────────────────────────────


@dataclass
class FlowSmoother:
    """Farneback optical flow propagates the previous mask forward, then
    blends with the per-frame raw mask.

    On each call:
      1. If we have a previous frame and previous mask:
         - compute Farneback flow from prev_gray → current_gray
         - remap the previous mask through the flow
         - blend remapped + raw via `flow_weight`
      2. Else: use raw_mask as-is.
      3. Cache current frame + smoothed mask for next call.

    `flow_weight` in [0, 1]: higher = trust the flow-warped past mask more.
    Default 0.5.
    """
    flow_weight: float = 0.5
    _prev_gray: np.ndarray | None = field(default=None, init=False)
    _prev_mask: np.ndarray | None = field(default=None, init=False)

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_mask = None

    def smooth(self, raw_mask: np.ndarray | None, snow_image: np.ndarray) -> np.ndarray | None:
        cur_gray = cv2.cvtColor(snow_image, cv2.COLOR_RGB2GRAY)

        if self._prev_gray is None or self._prev_mask is None:
            # First frame — nothing to propagate. Trust raw.
            self._prev_gray = cur_gray
            self._prev_mask = (raw_mask if raw_mask is not None
                               else np.zeros(cur_gray.shape, dtype=np.uint8))
            return raw_mask

        # Flow from prev → current. Each pixel (x, y) gets a (dx, dy) vector.
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, cur_gray, None,
            pyr_scale=0.5, levels=3, winsize=21,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
        )
        h, w = cur_gray.shape
        # remap: for each pixel in CURRENT frame, look up the corresponding
        # location in PREVIOUS frame: prev_xy = (x, y) - flow(x, y).
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = xs - flow[..., 0]
        map_y = ys - flow[..., 1]
        propagated = cv2.remap(
            self._prev_mask, map_x, map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        if raw_mask is None:
            blended = propagated
        else:
            # Blend in [0, 1] space, threshold at 0.5.
            blended_f = (self.flow_weight * propagated.astype(np.float32)
                         + (1.0 - self.flow_weight) * raw_mask.astype(np.float32))
            blended = (blended_f >= 0.5).astype(np.uint8)

        self._prev_gray = cur_gray
        self._prev_mask = blended
        return blended


def make_smoother(name: str | None, **kwargs) -> Smoother | None:
    """Factory for the CLI: `--temporal {none,ema,flow}`."""
    if name in (None, "none", "off"):
        return None
    if name == "ema":
        return EMASmoother(alpha=kwargs.get("alpha", 0.5))
    if name == "flow":
        return FlowSmoother(flow_weight=kwargs.get("flow_weight", 0.5))
    raise ValueError(f"Unknown temporal smoother: {name!r}. Try 'none', 'ema', 'flow'.")
