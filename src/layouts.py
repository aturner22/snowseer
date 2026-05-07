"""Per-pair still-image layout permutations.

Given the raw artefacts produced for a single (snow, clear) pair —
snow query, clear prior, road mask on the clear prior, snow with naive
red mask, snow with cross-season green overlay, the matches-viz canvas —
this module composes a fan of 2-up / 3-up combinations and singles
suitable for slide-deck arrangement.

Outputs land in `out_dir` with stable `<pair_id>__<layout>.png` names.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.overlay import alpha_blend

GREEN = (46, 156, 86)
GAP_PX = 12
BG = (246, 243, 238)


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    return cv2.resize(img, (int(round(w * scale)), target_h), interpolation=cv2.INTER_AREA)


def _hstack(panels: list[np.ndarray], gap_px: int = GAP_PX) -> np.ndarray:
    target_h = max(p.shape[0] for p in panels)
    panels = [_resize_to_height(p, target_h) for p in panels]
    if gap_px <= 0:
        return np.hstack(panels)
    gap = np.full((target_h, gap_px, 3), BG, dtype=np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.hstack([out, gap, p])
    return out


def _save(canvas: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def save_extra_layouts(
    *,
    snow: np.ndarray,
    clear: np.ndarray,
    road_mask_clear: np.ndarray,
    snow_naive: np.ndarray,
    snow_overlay: np.ndarray,
    matches_canvas: np.ndarray | None,
    out_dir: Path,
    pair_id: str,
) -> None:
    clear_with_mask = alpha_blend(clear, road_mask_clear, color=GREEN, alpha=0.50)

    # Singles (raw / lightly-annotated).
    _save(snow, out_dir / f"{pair_id}__snow.png")
    _save(clear, out_dir / f"{pair_id}__prior.png")
    _save(clear_with_mask, out_dir / f"{pair_id}__prior_with_mask.png")

    # 2-up paired comparisons.
    _save(_hstack([snow, clear]),                     out_dir / f"{pair_id}__pair_snow_prior.png")
    _save(_hstack([snow, snow_naive]),                out_dir / f"{pair_id}__pair_snow_naive.png")
    _save(_hstack([snow, snow_overlay]),              out_dir / f"{pair_id}__pair_snow_overlay.png")
    _save(_hstack([snow_naive, snow_overlay]),        out_dir / f"{pair_id}__pair_naive_overlay.png")
    _save(_hstack([clear_with_mask, snow_overlay]),   out_dir / f"{pair_id}__pair_prior_overlay.png")

    # 3-up storytelling triptychs.
    _save(_hstack([snow, snow_naive, snow_overlay]),
          out_dir / f"{pair_id}__triptych_failure_fix.png")
    if matches_canvas is not None:
        _save(_hstack([snow, matches_canvas, snow_overlay]),
              out_dir / f"{pair_id}__triptych_recovery.png")
        _save(_hstack([clear_with_mask, matches_canvas, snow_overlay]),
              out_dir / f"{pair_id}__triptych_provenance.png")
