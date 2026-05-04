"""4-panel contact-sheet generator for pair-by-pair visual audit.

Each row of the contact sheet shows, left-to-right:
    (1) snowy query frame
    (2) clear prior + Cityscapes road mask overlaid
    (3) snow + warped road overlay (the cross-season output)
    (4) snow + naive Cityscapes segmenter applied directly (the failure
        condition that motivates the cross-season approach)

The sheet is one tall PNG; scrollable in any image viewer. Each row carries
a label with pair_id, inliers, snow-quality composite, and accept-status.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

HEROES = Path("outputs/heroes")
PAIRS = Path("data/pairs")
OUT = Path("outputs/audit")


def _read(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _label_strip(width: int, text: str, *, height: int = 44) -> np.ndarray:
    bar = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(bar, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
    return bar


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_h / h
    return cv2.resize(img, (int(round(w * scale)), target_h), interpolation=cv2.INTER_AREA)


def _make_row(
    snow: np.ndarray,
    clear_with_mask: np.ndarray | None,
    overlay: np.ndarray | None,
    naive: np.ndarray | None,
    target_h: int = 360,
) -> np.ndarray:
    """One row of the contact sheet: 4 thumbnails side by side."""
    placeholder = np.full((target_h, target_h * 16 // 9, 3), 64, dtype=np.uint8)
    cv2.putText(placeholder, "(no overlay)", (placeholder.shape[1] // 4, target_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)

    snow_r = _resize_to_height(snow, target_h)
    parts = [snow_r]
    for img in (clear_with_mask, overlay, naive):
        if img is None:
            parts.append(_resize_to_height(placeholder.copy(), target_h))
        else:
            parts.append(_resize_to_height(img, target_h))
    # Pad to a uniform per-cell width so the columns line up across rows.
    cell_w = max(p.shape[1] for p in parts)
    aligned: list[np.ndarray] = []
    for p in parts:
        if p.shape[1] == cell_w:
            aligned.append(p)
        else:
            padded = np.zeros((p.shape[0], cell_w, 3), dtype=np.uint8)
            padded[:, : p.shape[1]] = p
            aligned.append(padded)
    return np.hstack(aligned)


def _composite_clear_with_mask(pair_dir: Path, mask_path: Path | None = None) -> np.ndarray | None:
    """Reconstruct the 'clear + road mask' middle panel from cached pieces.

    Falls back to reading the existing 3-panel and slicing its middle third.
    """
    panel = _read(HEROES / f"{pair_dir.name}__panel.png")
    if panel is None:
        return None
    h, w = panel.shape[:2]
    # Three columns laid out by matplotlib are roughly thirds; trim a bit of margin.
    third = w // 3
    middle = panel[:, third : 2 * third, :]
    return middle


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = HEROES / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []

    # Optional: pull manual curation labels.
    manual_path = Path("data/manual_snow_curation.json")
    manual = json.loads(manual_path.read_text()) if manual_path.exists() else {}

    rows: list[np.ndarray] = []
    target_w: int | None = None
    target_h = 360

    for entry in summary:
        pair_id = entry["pair_id"]
        snow_path = PAIRS / pair_id / "snow.jpg"
        snow = _read(snow_path)
        if snow is None:
            continue
        overlay = _read(HEROES / f"{pair_id}__overlay.png")
        naive = _read(HEROES / f"{pair_id}__naive_baseline.png")
        clear_with_mask = _composite_clear_with_mask(PAIRS / pair_id)

        row = _make_row(snow, clear_with_mask, overlay, naive, target_h=target_h)
        if target_w is None:
            target_w = row.shape[1]
        elif row.shape[1] != target_w:
            row = cv2.resize(row, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Quality and curation metadata
        sq_path = PAIRS / pair_id / "snow_quality.json"
        sq = json.loads(sq_path.read_text()) if sq_path.exists() else {}
        meta_path = PAIRS / pair_id / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        manual_verdict = (manual.get(pair_id) or {}).get("verdict", "—")

        label = (
            f"{pair_id}    "
            f"region={meta.get('region','?')}    "
            f"inliers={entry.get('n_inliers','?')}    "
            f"refined={entry.get('refined', False)}    "
            f"auto={'accept' if entry.get('accept') else 'reject'}    "
            f"manual={manual_verdict}    "
            f"snow_q={(sq.get('composite') or 0):.2f}"
        )
        bar = _label_strip(target_w, label)
        rows.append(np.vstack([bar, row]))

    if not rows:
        raise SystemExit("no panels to stitch")

    sheet = np.vstack(rows)
    sheet_path = OUT / "contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"wrote {sheet_path}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
