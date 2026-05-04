"""Print and persist a curated subset of pairs.

This is the *content-level* sanity check that complements Mapillary's
metadata-level pairing. Tight GPS + heading still produces some pairs that
are visually different scenes (opposing carriageways of a divided road,
different streets at the same lat/lng, etc.). The pipeline exposes an
`accept` flag per pair driven by `ACCEPT_INLIER_MIN`; this script aggregates
those flags into a curated index that downstream consumers (Streamlit,
notebook, video) read.

Usage:
    uv run python -m data.curate_pairs                # report only
    uv run python -m data.curate_pairs --write-index  # write data/pairs/curated.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAIRS_DIR = Path("data/pairs")
SUMMARY = Path("outputs/heroes/summary.json")
CURATED_INDEX = PAIRS_DIR / "curated.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-index", action="store_true", help="write data/pairs/curated.json")
    ap.add_argument("--summary", default=str(SUMMARY))
    args = ap.parse_args()

    if not Path(args.summary).exists():
        raise SystemExit(f"{args.summary} not found. Run `uv run python -m src.pipeline` first.")

    summary = json.loads(Path(args.summary).read_text())
    accepted = [s for s in summary if s.get("accept")]
    rejected = [s for s in summary if not s.get("accept")]

    print(f"Total pairs:    {len(summary)}")
    print(f"Accepted:       {len(accepted)}  ({len(accepted)/max(1,len(summary)):.0%})")
    print(f"Rejected:       {len(rejected)}  ({len(rejected)/max(1,len(summary)):.0%})")
    print()
    print("Rejected (likely content-mismatched or too-noisy):")
    for s in sorted(rejected, key=lambda x: x.get("n_inliers", 0)):
        print(f"  - {s['pair_id']:60s}  inliers={s.get('n_inliers',0):3d}  matches={s.get('n_matches',0):4d}")
    print()
    print("Accepted (top by inliers):")
    for s in sorted(accepted, key=lambda x: -x.get("n_inliers", 0))[:10]:
        print(f"  + {s['pair_id']:60s}  inliers={s.get('n_inliers',0):3d}  matches={s.get('n_matches',0):4d}")

    if args.write_index:
        CURATED_INDEX.parent.mkdir(parents=True, exist_ok=True)
        CURATED_INDEX.write_text(
            json.dumps(
                {
                    "accepted_pair_ids": [s["pair_id"] for s in accepted],
                    "rejected_pair_ids": [s["pair_id"] for s in rejected],
                    "threshold_inliers_min": 15,
                },
                indent=2,
            )
        )
        print(f"\nwrote {CURATED_INDEX}")


if __name__ == "__main__":
    main()
