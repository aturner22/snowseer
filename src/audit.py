"""Multi-prior contact sheet generator.

Per pair, two rows:
  row 1: snow | naive | overlay_union | overlay_weighted | overlay_majority
  row 2: per-prior overlay strip (K thumbnails)

One PNG, scrollable in any image viewer. Use it to A/B the three fusion
strategies and decide which is the right primary for the demo.
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


def _placeholder(w: int, h: int, label: str) -> np.ndarray:
    out = np.full((h, w, 3), 64, dtype=np.uint8)
    cv2.putText(out, label, (w // 6, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    return out


def _hstack_resized(imgs: list[np.ndarray], target_h: int = 320) -> np.ndarray:
    sized = [_resize_to_height(im, target_h) for im in imgs]
    max_w = max(im.shape[1] for im in sized)
    aligned = []
    for im in sized:
        if im.shape[1] == max_w:
            aligned.append(im)
        else:
            padded = np.zeros((im.shape[0], max_w, 3), dtype=np.uint8)
            padded[:, : im.shape[1]] = im
            aligned.append(padded)
    return np.hstack(aligned)


def _build_row(pair_id: str, summary_entry: dict, manual: dict, ratings: dict) -> np.ndarray | None:
    snow_path = PAIRS / pair_id / "snow.jpg"
    snow = _read(snow_path)
    if snow is None:
        return None

    naive = _read(HEROES / f"{pair_id}__naive_baseline.png")
    over_union = _read(HEROES / f"{pair_id}__overlay_union.png")
    over_weighted = _read(HEROES / f"{pair_id}__overlay_weighted.png")
    over_majority = _read(HEROES / f"{pair_id}__overlay_majority.png")
    priors_strip = _read(HEROES / f"{pair_id}__priors.png")

    h = 320
    placeholder = _placeholder(snow.shape[1], snow.shape[0], "(missing)")

    # Row 1: snow / naive / union / weighted / majority
    row1_cells = [
        snow,
        naive if naive is not None else placeholder,
        over_union if over_union is not None else placeholder,
        over_weighted if over_weighted is not None else placeholder,
        over_majority if over_majority is not None else placeholder,
    ]
    row1 = _hstack_resized(row1_cells, target_h=h)

    # Row 2: priors strip (already wide)
    if priors_strip is not None:
        row2 = _resize_to_height(priors_strip, h)
        # pad/crop to match row1 width
        if row2.shape[1] < row1.shape[1]:
            pad = np.zeros((h, row1.shape[1] - row2.shape[1], 3), dtype=np.uint8)
            row2 = np.hstack([row2, pad])
        elif row2.shape[1] > row1.shape[1]:
            row2 = row2[:, : row1.shape[1]]
    else:
        row2 = np.full((h, row1.shape[1], 3), 250, dtype=np.uint8)

    # Per-cell labels above row 1 (grey strip)
    cell_w = row1.shape[1] // 5
    label_h = 36
    labels_row = np.full((label_h, row1.shape[1], 3), 240, dtype=np.uint8)
    for i, name in enumerate(["snow query", "naive (red)", "union+erosion", "weighted (primary)", "majority"]):
        cv2.putText(labels_row, name, (i * cell_w + 12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (28, 28, 28), 1, cv2.LINE_AA)

    # Pair-level header strip
    rating = (ratings.get(pair_id) or {}).get("rating", "—")
    manual_verdict = (manual.get(pair_id) or {}).get("verdict", "—")
    inliers = summary_entry.get("n_inliers", "?")
    refined = summary_entry.get("refined", False)
    header = _label_strip(
        row1.shape[1],
        f"{pair_id}    inliers={inliers}    refined={refined}    "
        f"manual_snow={manual_verdict}    rating={rating.upper()}",
    )

    return np.vstack([header, labels_row, row1, row2])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = HEROES / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []

    manual_path = Path("data/manual_snow_curation.json")
    manual = json.loads(manual_path.read_text()) if manual_path.exists() else {}
    ratings_path = Path("data/manual_result_curation.json")
    ratings = json.loads(ratings_path.read_text()) if ratings_path.exists() else {}

    rows: list[np.ndarray] = []
    target_w: int | None = None
    for entry in summary:
        row = _build_row(entry["pair_id"], entry, manual, ratings)
        if row is None:
            continue
        if target_w is None:
            target_w = row.shape[1]
        elif row.shape[1] != target_w:
            row = cv2.resize(row, (target_w, row.shape[0]), interpolation=cv2.INTER_AREA)
        rows.append(row)

    if not rows:
        raise SystemExit("no rows to stitch")
    sheet = np.vstack(rows)
    out_path = OUT / "contact_sheet.png"
    cv2.imwrite(str(out_path), sheet)
    print(f"wrote {out_path}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
