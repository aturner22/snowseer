"""Contact-sheet generator for pair-by-pair visual audit.

Reads the existing panel PNGs and stitches them into a single vertically-stacked
sheet so a human can scan all pairs at once and tag them ✅ / ⚠ / ❌.
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


def _label(img: np.ndarray, text: str) -> np.ndarray:
    h, w = img.shape[:2]
    bar = np.full((44, w, 3), 32, dtype=np.uint8)
    cv2.putText(bar, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = HEROES / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []

    rows: list[np.ndarray] = []
    target_w: int | None = None
    for entry in summary:
        pair_id = entry["pair_id"]
        meta_path = PAIRS / pair_id / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        panel = _read(HEROES / f"{pair_id}__panel.png")
        if panel is None:
            # graceful failure case — show the matches viz instead
            panel = _read(HEROES / f"{pair_id}__matches.png")
            note = "[FAILURE — no overlay produced]"
        else:
            note = ""
        if panel is None:
            continue
        # Normalise widths so vstack works.
        if target_w is None:
            target_w = panel.shape[1]
        elif panel.shape[1] != target_w:
            scale = target_w / panel.shape[1]
            panel = cv2.resize(panel, (target_w, int(panel.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        label = (
            f"{pair_id}    inliers={entry.get('n_inliers','?')}    "
            f"matches={entry.get('n_matches','?')}    "
            f"d={meta.get('distance_m','?')}m    Δh={meta.get('heading_delta_deg','?')}°    {note}"
        )
        rows.append(_label(panel, label))

    if not rows:
        raise SystemExit("no panels to stitch")
    sheet = np.vstack(rows)
    sheet_path = OUT / "contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"wrote {sheet_path}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
