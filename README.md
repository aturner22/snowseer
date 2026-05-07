# Snow-Underlay

> *Constants as the bridge.* Minimal-shot autonomy by extending a model trained on one regime into the regime where we lack data — by anchoring on what stays the same between them.

A submission to [SoTA Commission I — Minimal-Shot Autonomy](https://sotaletters.substack.com/p/sota-commission-i-minimal-shot-autonomy).

This branch (`video`) is the moving demonstration: the same cross-season principle, now operating frame by frame on a snow plough's live video. The earlier static per-image demo (the precursor) is preserved at `git checkout main` (single-prior, the v1 narrative) and `git checkout v1.2-multi-prior-experiment` (the multi-prior fusion ablation), and is also runnable here via `make stills` (single-prior, default) or `make stills-multi`.

---

## What this project is really about

Self-driving systems are trained on dry roads, deliberately. Every canonical autonomy dataset — Cityscapes, KITTI, nuScenes, Waymo Open — is dominated by clear-weather European or American highways under daylight. A system trained on this corpus and asked to operate when the road is buried in snow has not been asked a research question. It has been asked the wrong question.

The familiar response is *we just need more data*. Annotate snowy roads. Annotate dust storms. Annotate fog, lava, oil spills, washouts. There are 27 million miles of road in the world, and the long tail of conditions that road can be in is longer than the road itself. We are not going to label our way out of it.

There is a different move available. **For almost every operating regime where autonomy fails for lack of data, there is an adjacent regime — temporally, seasonally, geographically — where we have plenty of data, and where the *parts that matter* are the same.** A snow plough's road is the same road it was last July. The curb hasn't moved. The hydrant hasn't moved. The road's *appearance* has changed completely; its *position in space* has not.

If we can identify what stays constant between the data-rich regime and the data-poor one, we can extend our existing models into the new regime without learning a single thing about it. We use the constants as a bridge.

This project is one concrete instantiation of that idea, applied to autonomous snow ploughs. The principle is general; the snow plough is the vehicle.

## The video extension — and why it matters

The earlier version of this work (v1.x, see the side branches and `make stills`) demonstrated the principle on individual snow / clear-season image pairs. It worked, on fourteen hand-curated locations. The thesis was already there.

But a snow plough does not see one frame; it drives. The interesting question is whether the principle survives motion. The video pipeline here answers that:

- A 15-second snow drive (Boreas dataset, Toronto, January 2021, road buried, lane markings invisible).
- The cross-season pipeline runs **per frame**: pull K=3 nearest summer Boreas captures of the same physical road, match snow→summer, warp the segmenter's road mask back, fuse, smooth.
- The resulting road overlay tracks the buried road continuously through the clip.

The composition is the same as the static demo. The new components are minimal:

- **`Track` + `PriorPool`** (`src/video_runtime/`): index a snow stream and a paired summer stream, look up the K nearest summer priors per snow frame by UTM distance.
- **EMA temporal smoothing** (`src/video_runtime/temporal.py`): exponentially-weighted moving average on the binary mask, α = 0.4. Reduces frame-to-frame jitter without introducing the drift we saw with optical flow propagation.
- **Cache layer** (`pipeline_v.run_track`): the matching step is dominant cost; pickled FrameResults let us re-render different smoothers / layouts in seconds.

Everything else — the matcher (DISK + LightGlue), the segmenter (Mask2Former-Cityscapes), the homography (USAC-MAGSAC), the warp — is unchanged from the static path.

## Architecture

| Component | Role | Pretrained on | Reference |
| --- | --- | --- | --- |
| **DISK** | Local feature detector + descriptor | MegaDepth (outdoor scenes, no snow) | Tyszkiewicz et al., *DISK: Learning local features with policy gradient*, NeurIPS 2020 |
| **LightGlue** | Sparse feature matcher | MegaDepth | Lindenberger et al., *LightGlue: Local Feature Matching at Light Speed*, ICCV 2023 |
| **USAC-MAGSAC** | Robust homography fit | — | Barath et al., *MAGSAC++, a fast, reliable and accurate robust estimator*, CVPR 2020 |
| **Mask2Former** | Semantic road segmentation, on the summer prior only | Cityscapes (clear-weather European driving, no snow) | Cheng et al., *Masked-attention Mask Transformer for Universal Image Segmentation*, CVPR 2022 |
| **Boreas** | Open snow-driving dataset (snow + paired summer captures, cm-accurate Applanix poses) | — | Burnett et al., *Boreas: A Multi-Season Autonomous Driving Dataset*, IJRR 2023, CC BY 4.0 |
| **Mapillary API v4** | Open imagery substrate (used for the static-stills precursor and for additional priors) | — | mapillary.com/developer/api-documentation |

Every learned component is **frozen**. Nothing is trained, nothing is fine-tuned. Snowy imagery enters the system only at inference time.

## How it works (the per-frame pipeline)

`src/video_runtime/pipeline_v.run_track` iterates over the snow stream. Each frame:

1. **Look up K=3 summer priors** by UTM distance (`PriorPool.select`, KD-tree on summer poses). The same summer Boreas trajectory loops the same Glen Shields route as the snow trajectory; nearest-neighbour in `(easting, northing)` finds the priors that physically passed the closest to the current snow pose.
2. **Match snow → each prior** via DISK + LightGlue (`src/matching.py`). For each, fit a homography with `cv2.findHomography(USAC_MAGSAC)` biased to ground-plane keypoints (`src/homography.py`).
3. **Warp the prior's road mask** (Mask2Former on the summer prior, cached the first time) back to snow image space via `H⁻¹`. The prior segmentation is reduced to its single largest connected component first.
4. **Fuse the K warped masks** (`src/fuse.weighted_soft_average`, weighted by per-prior inlier count). Edge-erode each prior's coverage region by 8 px to suppress warp aliasing on frame boundaries.
5. **Foreground-crop** at `y ≥ 0.30 H` (`src/fuse.crop_foreground`). A roof-mounted forward camera's road region sits in the lower 70 % of the frame; the upper 30 % is sky / horizon and we don't claim those pixels as drivable.
6. **EMA-smooth** across frames (`src/video_runtime/temporal.EMASmoother`, α = 0.4). On a frame whose matcher fails entirely, the previous smoothed mask is held — graceful degradation rather than empty-frame flicker.

The result is alpha-blended (green, `#2e9c56`) onto the snow frame and emitted to mp4.

`src/video_runtime/render_all_layouts.py` then produces five visual variants from the same matching cache — overlay, side-by-side, two 3-panel orderings, and a 2×2 quad — for whichever framing best fits a given audience.

## Minimal-shot integrity

| Claim | Status |
| --- | --- |
| Zero snowy frames touch any model weights | ✓ |
| Zero snowy frames touch any annotation pipeline | ✓ |
| Snow appears only as runtime input | ✓ |
| Reproducible from a clean clone with one command | ✓ |
| Pretrained matcher · pretrained segmenter · classical RANSAC | ✓ |

## Reproducing the canonical clip

```bash
uv sync --python 3.12
export MAPILLARY_TOKEN=<token from https://www.mapillary.com/dashboard/developers>
make reproduce
```

That pulls the Boreas snow + summer windows for `boreas_2021_01_26` (~1.4 GB), builds the matching cache (~50 min on Mac CPU), and renders `outputs/video/boreas_2021_01_26/overlay.mp4` — a 15-second clip of the cross-season road overlay.

`MAPILLARY_TOKEN` is only required for the static-stills precursor (`make stills`); the canonical video pipeline runs entirely off Boreas, which is on the AWS Open Data registry and needs no signup.

| Want to … | Run |
| --- | --- |
| Reproduce the canonical clip | `make reproduce` |
| Render all 5 layouts (overlay / sidebyside / 3-panel × 2 / quad) | `make reproduce-all-layouts TRACK=boreas_2021_01_26` |
| Run the pipeline on a different track | `make reproduce-track TRACK=boreas_2024_12_23` |
| Run the static-stills precursor (single-prior, v1 narrative) | `make stills` |
| Run the static-stills multi-prior fusion ablation (Phase J) | `make stills-multi` |
| Open the Streamlit viewer over cached static stills | `make stream` |
| Render the writeup + slide PDFs (local only — gitignored) | `make pdfs` |
| Open the GitHub Pages site locally | open `docs/index.html` |
| List all `make` targets | `make help` |

## Repo layout

```
snow-underlay/
├── Makefile                          # all reproduce / stills / docs commands
├── README.md
├── pyproject.toml · uv.lock
│
├── src/                              # static-prior precursor (used by `make stills`)
│   ├── matching.py                   # DISK + LightGlue
│   ├── homography.py                 # RANSAC, ground-plane biased, iterative refinement
│   ├── segmentation.py               # Mask2Former
│   ├── overlay.py                    # warp + 4-panel figure
│   ├── fuse.py                       # multi-prior fusion + foreground crop
│   ├── snow_quality.py · audit.py    # snow pre-filter + contact sheet
│   ├── pipeline.py                   # static-pair pipeline
│   └── video_runtime/                # per-frame video pipeline (the canonical path)
│       ├── track.py                  # snow stream + summer stream loaders
│       ├── prior_pool.py             # K-NN prior selection by UTM
│       ├── pipeline_v.py             # run_track entry point + cache + checkpoint resume
│       ├── temporal.py               # EMA / flow smoothers (EMA wins)
│       ├── overlay_render.py         # render_overlay / sidebyside / 3-panel / quad
│       ├── augment.py                # naive baseline + summer panel cache
│       ├── extract_assets.py         # extract preset stills (1.0/5.0/10.0/14.0 s) from mp4s
│       ├── fetch_track.py            # Boreas S3 fetcher with retry
│       ├── render.py                 # CLI entry
│       └── render_all_layouts.py     # batch renderer (5 layouts)
│
├── data/
│   ├── curated_pairs.json            # 27 reviewed Mapillary pairs (14 GREAT+OKAY headline + 13 review-pool)
│   ├── manual_*_curation.json        # Streamlit curator state
│   ├── fetch_mapillary.py            # Mapillary v4 fetcher
│   ├── find_snow_sequences.py        # winter-sequence reconnaissance
│   ├── preview_sequence.py           # thumbnail montage tool
│   ├── pairs/                        # static-stills pair downloads (gitignored)
│   └── video/                        # (gitignored — regenerate via `make video-fetch`)
│       ├── tracks/<track_id>/        # per-track snow + summer windows
│       │   ├── snow/{frames/, camera_poses.csv, calib/, window.json}
│       │   └── summer/{frames/, camera_poses.csv, calib/, window.json}
│       └── recon/                    # Mapillary scanner outputs
│
├── demo/
│   ├── streamlit_app.py              # cached-output viewer
│   └── curate_snow.py                # Streamlit big-image accept/reject
│
├── docs/
│   ├── style/                        # visual identity (charcoal · cream · rust) + Marp theme
│   ├── _assets/                      # Pages-deployed JPEGs (poster placeholders for mp4s)
│   ├── writeup.md                    # ≤ 2-page essay (PDF gitignored)
│   ├── slides.md                     # Marp deck + submission-video storyboard appendix (PDF gitignored)
│   └── index.html                    # GitHub Pages site
│
├── outputs/
│   ├── heroes/                       # static-stills panels (gitignored)
│   ├── audit/                        # static-stills contact sheet (gitignored)
│   └── video/<track_id>/             # video renders + matching cache (gitignored)
│
└── assets/
    └── fonts/                        # EB Garamond · Inter · JetBrains Mono (OFL)
```

`_archive/` (gitignored) holds legacy code kept on local disk for reference but not part of the canonical repo: the auto-rendered submission video composer (`compose_final.py`, audio, music — user composes externally now), multi-prior schema migrations from Phase J, the v1 walkthrough notebook, the Phase A audit notes, the deferred 02_video_walkthrough.ipynb (the writeup essay + Pages site replaced its narrative role).

## Honest limits

**The contribution is bounded.** Snow-Underlay is one channel of a fuller autonomy stack — a 2D road-position prior — not a complete perception system. The output answers *where the road should be*, not *where to drive*. A snow-covered car parked on the road would still sit inside the green overlay; the system has no notion of obstacles, drivable surface, or 3D geometry. This pipeline is designed to feed into a stack alongside lidar, depth estimation, and obstacle detection — not to replace any of them. The contribution we are demonstrating is the *move* — transferring knowledge across regimes through a learned-invariant constant — not a turnkey snowplough perception system.

Within that scope:

- A single homography assumes a near-planar scene. When matches concentrate on building façades rather than the road plane, the warp can drift. Ground-plane bias and iterative refinement mitigate but don't eliminate.
- Heavy snow on the lens (water droplets, lens-occlusion) reduces match quality on affected frames. The system fails *gracefully* — low inlier counts → EMA holds the previous good mask.
- Per-sequence Boreas UTM frames don't share a global origin, so summer pairings across years require careful trajectory alignment. The `fetch_track` script handles this for the canonical pairing; non-default summer pairings are a manual operation.
- **Inlier count is not a reliable predictor of overlay quality.** A pair with hundreds of inliers can warp the mask onto the wrong region if the inliers concentrate on building façades. Empirical, from the static-stills work; survives into video.

## Future direction

1. **Real-time** rather than pseudo-realtime. The current matcher is the cost; replacing it with a learned cross-season-robust correspondence model trained on synthetic snow + clear pairs would close the gap. The principle is unchanged.
2. **Local 2D bird's-eye-view map** maintained over the snow drive: aggregate per-frame masks into a GPS-keyed map, reproject into camera frame on demand. Trades per-frame cost for global consistency.
3. **Operator's own prior captures** instead of Mapillary / public datasets. Same pipeline, much tighter geometric prior, no contributor-coverage gaps.

The principle is unchanged through all three: the plough never has to learn what snow looks like; it only has to find what is constant under the snow.

## Closing

The contribution is the move, not the parts. Where data is missing, find a regime where it isn't; identify what is constant between the two; transfer through the constant. Snow on a road is one example. Polar imaging without polar training data, low-light medical imaging without low-light training data, a manipulator on Mars without Mars training data — all admit the same structure.

Constants as the bridge.

---

*Acknowledgements.* Boreas dataset (Burnett et al. 2023, UTIAS-ASRL) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Mapillary imagery under the open-data licence. Visual identity inspired by [SOTA Letters](https://sotaletters.substack.com/). Bensound *Slow Motion* (free with attribution) used in optional title-card composition. Models pretrained by their respective authors and used frozen. Repository licensed under MIT.
