"""Build a matches sidecar cache for a track.

For each cached snow frame, re-run the matcher on (snow_image, best_prior)
where best_prior is the per-prior record with the highest inlier count.
Save keypoints + inlier mask + prior image so the matches video / stills
can be rendered without re-running matching.

Sidecar path: outputs/toronto_video/<track>/_matches_<cache_tag>.pkl

Schema:
    {
      "track": <track_id>,
      "cache_tag": <tag>,
      "frames": [
        {
          "snow_idx": int,
          "prior_idx": int,
          "prior_image": np.ndarray (HxWx3 uint8),
          "kpts0": np.ndarray (N,2 float32),
          "kpts1": np.ndarray (N,2 float32),
          "inlier_mask": np.ndarray (N,) bool,
        },
        ...
      ],
    }
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from src.homography import estimate
from src.matching import Matcher
from src.video_runtime.prior_pool import PriorPool
from src.video_runtime.track import Track

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--cache-tag", required=True)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--max-dim", type=int, default=1024)
    args = p.parse_args()

    cache_path = ROOT / f"outputs/toronto_video/{args.track}/_cache_{args.cache_tag}.pkl"
    sidecar_path = ROOT / f"outputs/toronto_video/{args.track}/_matches_{args.cache_tag}.pkl"

    if not cache_path.exists():
        raise SystemExit(f"missing cache {cache_path}")

    with open(cache_path, "rb") as fh:
        payload = pickle.load(fh)
    results = payload["results"]

    track = Track(args.track)
    pool = PriorPool(track, K=args.K, max_dim=args.max_dim)
    matcher = Matcher()

    print(f"[matches] {args.track}: replaying matcher on {len(results)} frames "
          f"(best prior per frame)")

    t0 = time.time()
    out_frames = []
    for i, r in enumerate(results):
        ok = [(j, pp) for j, pp in enumerate(r.per_prior) if pp["ok"]]
        if not ok:
            out_frames.append(None)
            print(f"  [{i + 1:>3d}/{len(results)}] no successful prior — skip", flush=True)
            continue
        best_j, best_pp = max(ok, key=lambda tup: tup[1]["n_inliers"])
        priors = pool.select(r.snow_meta)
        prior = priors[best_j]
        match = matcher.match(r.snow_image, prior.image)
        homo = estimate(match, r.snow_image.shape[:2], prior.image.shape[:2])
        if homo.inlier_mask is None:
            out_frames.append(None)
            print(f"  [{i + 1:>3d}/{len(results)}] homography failed on best prior — skip", flush=True)
            continue
        out_frames.append({
            "snow_idx": int(r.snow_meta.idx),
            "prior_idx": int(prior.meta.idx) if prior.meta is not None else -1,
            "prior_image": prior.image,
            "kpts0": match.kpts0.astype(np.float32),
            "kpts1": match.kpts1.astype(np.float32),
            "inlier_mask": homo.inlier_mask.astype(bool),
        })
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(results) - i - 1)
        n_inliers = int(homo.inlier_mask.sum())
        print(f"  [{i + 1:>3d}/{len(results)}] matched (n_inliers={n_inliers}); "
              f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "wb") as fh:
        pickle.dump({
            "track": args.track,
            "cache_tag": args.cache_tag,
            "frames": out_frames,
        }, fh)
    print(f"\n[matches] wrote {sidecar_path}")


if __name__ == "__main__":
    main()
