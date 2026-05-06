"""Preview a Mapillary sequence by pulling thumbnails of N evenly-spaced
images. Outputs a grid montage so we can eyeball aesthetics before
committing to a full pull.

Usage:
    uv run python -m data.preview_sequence --image-id <id> --n 6
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/video/recon/_thumbs"
GRAPH = "https://graph.mapillary.com"


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "snow-underlay-recon/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def _fetch_json(url: str) -> dict:
    return json.loads(_fetch(url).decode())


def _list_sequence_image_ids(token: str, sequence_id: str) -> list[str]:
    qs = urllib.parse.urlencode({
        "access_token": token,
        "sequence_id": sequence_id,
    })
    j = _fetch_json(f"{GRAPH}/image_ids?{qs}")
    return [str(x.get("id")) for x in j.get("data", [])]


def _image_thumb_url(token: str, image_id: str) -> str:
    qs = urllib.parse.urlencode({
        "access_token": token,
        "fields": "thumb_1024_url",
    })
    j = _fetch_json(f"{GRAPH}/{image_id}?{qs}")
    return j.get("thumb_1024_url", "")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence-id", required=False)
    p.add_argument("--image-id", required=False, help="single-image preview")
    p.add_argument("--n", type=int, default=6, help="frames to sample")
    p.add_argument("--label", default="preview")
    args = p.parse_args()

    from data.fetch_mapillary import _load_dotenv
    _load_dotenv()
    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        raise SystemExit("MAPILLARY_TOKEN not set.")

    OUT.mkdir(parents=True, exist_ok=True)

    if args.sequence_id:
        ids = _list_sequence_image_ids(token, args.sequence_id)
        if not ids:
            raise SystemExit(f"No images for sequence {args.sequence_id}")
        # Sample N evenly spaced.
        n = min(args.n, len(ids))
        step = max(1, len(ids) // n)
        sample = ids[::step][:n]
        target = OUT / f"{args.label}_seq_{args.sequence_id[:10]}.jpg"
    elif args.image_id:
        sample = [args.image_id]
        target = OUT / f"{args.label}_img_{args.image_id}.jpg"
    else:
        raise SystemExit("Pass --sequence-id or --image-id")

    thumbs: list[np.ndarray] = []
    for i, iid in enumerate(sample):
        try:
            url = _image_thumb_url(token, iid)
            data = _fetch(url)
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            # Resize for grid.
            h, w = img.shape[:2]
            tgt_w = 400
            scale = tgt_w / w
            img = cv2.resize(img, (tgt_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)
            thumbs.append(img)
            print(f"  [{i+1}/{len(sample)}] {iid}  {img.shape[1]}x{img.shape[0]}")
        except Exception as exc:
            print(f"  [{i+1}/{len(sample)}] {iid}: {exc}")

    if not thumbs:
        raise SystemExit("no thumbnails fetched")

    # Pad widths
    max_w = max(t.shape[1] for t in thumbs)
    max_h = max(t.shape[0] for t in thumbs)
    thumbs = [_pad(t, max_w, max_h) for t in thumbs]
    n_cols = min(3, len(thumbs))
    n_rows = (len(thumbs) + n_cols - 1) // n_cols
    canvas = np.full((max_h * n_rows, max_w * n_cols, 3), 30, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, n_cols)
        canvas[r * max_h:(r + 1) * max_h, c * max_w:(c + 1) * max_w] = t
    cv2.imwrite(str(target), canvas)
    print(f"\nwrote {target}")


def _pad(img: np.ndarray, w: int, h: int) -> np.ndarray:
    out = np.full((h, w, 3), 30, dtype=np.uint8)
    out[:img.shape[0], :img.shape[1]] = img
    return out


if __name__ == "__main__":
    main()
