# Snow-Underlay

> A snow plough's job is short: keep the road clear. While it's doing it, the road is invisible — buried, lane markings gone, curb-line erased. A self-driving stack trained on Cityscapes will report, with calibrated confidence, that the entire scene is sky.

> The contribution this repository ships is *not* a snow plough. It is a primitive — **the constants-bridge** — a composition that takes a model trained on regime A, an inference target in regime B, and a known invariant between A and B, and uses the invariant to transfer the model into B without retraining. The snow plough is one consumer of this primitive, demonstrated in motion on a 15-second snow-buried Toronto street. *Generalisation, not memorisation.*

A submission to [SoTA Commission I — Minimal-Shot Autonomy](https://sotaletters.substack.com/p/sota-commission-i-minimal-shot-autonomy). Headline artefact: `outputs/video/boreas_2021_01_26/overlay.mp4`. Reproduce with `make reproduce`.

**Documents**: [`docs/writeup.md`](docs/writeup.md) (the essay) · [`docs/analysis.ipynb`](docs/analysis.ipynb) (the work, shown with the work) · [`docs/audit_log.md`](docs/audit_log.md) (every approach + dataset we tried and rejected, with reasons) · [`docs/submission_video_plan.md`](docs/submission_video_plan.md) (90–120s shot list) · [`docs/external_datasets.md`](docs/external_datasets.md) (acquisition guide for non-Boreas snow datasets) · [`docs/index.html`](docs/index.html) (Pages site).

**Quick start:**

```bash
git clone https://github.com/aturner22/snowseer; cd snowseer
git checkout video
uv sync --python 3.12
make reproduce        # canonical 15 s clip; ~50 min cache + ~1 min render
make demo SNOW=<jpg> PRIOR=<jpg>   # live: any (snow, prior) pair → 15 panels
open docs/index.html               # static Pages site, no server needed
```

**For judges**: `make help` lists every reproducible target. The canonical matching cache builds in ~50 min on Mac CPU; subsequent renders (6 visual layouts + 4 timestamps × 4 stills) are under 30 min total. The static-stills precursor (`make stills`) needs a free [Mapillary token](https://www.mapillary.com/dashboard/developers) and adds ~10 min. The interactive demo (`make demo SNOW=... PRIOR=...`) runs the full pipeline on any user-provided pair and emits 15 layout outputs in `outputs/demo/`. CI runs smoke tests on every push.

---

## What this project is really about

Minimal-shot autonomy is the question of how a perception system survives in regimes it has not been heavily trained on. The default answer is *collect more data and retrain*. That answer assumes labelling can keep pace with reality. It cannot — not for snow, dust, ash, washouts, regional construction practices, agricultural off-road, or any of the conditions a vehicle, robot, or drone meets when it leaves the regime its training set was sampled from. A perception system that depends on having been trained on each new condition will always lag every condition it has not yet been trained on.

There is a different move available, and it doesn't require new training data. **For almost every operating regime where autonomy fails for lack of data, there is an adjacent regime — temporally, seasonally, geographically — where data exists, and where the *parts that matter* are the same.** A snow plough's road is the same road it was last July. The curb hasn't moved. The hydrant hasn't moved. The road's *appearance* has changed completely; its *position in space* has not.

If we can identify what stays constant between the data-rich regime and the data-poor one, we can extend our existing models into the new regime without learning a single thing about it. We use the constant as a bridge. This is generalisation, not memorisation.

This project is one concrete instantiation of that idea, applied to autonomous snow ploughs. The principle is general; the snow plough is the demonstration.

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
| **Boreas** *(canonical video demo)* | Open snow-driving dataset (snow + paired summer captures, cm-accurate Applanix poses) | — | Burnett et al., *Boreas: A Multi-Season Autonomous Driving Dataset*, IJRR 2023, CC BY 4.0 |
| **Mapillary API v4** *(static-stills demo)* | One example of an open imagery substrate. Any geo-tagged clear-weather imagery would work — Google Street View, Bing Streetside, the operator's own captures. The pipeline is substrate-agnostic. | — | mapillary.com/developer/api-documentation |

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

### Worked example: one frame, end-to-end

A representative frame from late in the canonical clip — Glen Shields, residential, mid-morning, heavy snow on the pavement — traced through the pipeline.

| Step | What happens |
| --- | --- |
| **1. Prior selection** | KD-tree returns the K=3 nearest summer captures by UTM distance. The canonical loop's summer trajectory is sampled densely; on this frame all three nearest summer captures are within a couple of metres of the snow pose. |
| **2. Match** | DISK runs on the snow frame and on each summer prior; LightGlue produces matched correspondences; USAC-MAGSAC fits H per pair, restricted to keypoints in the lower portion of the image to bias the geometry to the ground plane. The matcher anchors on what the season has not changed: gate posts, fence wires, distant roof edges, masonry corners. |
| **3. Segment the prior** | Mask2Former (frozen, Cityscapes) runs on each summer prior. Each mask is reduced to its single largest connected component — one drivable surface per prior. |
| **4. Warp** | Each prior's road mask is warped via `H⁻¹` into snow image space; the warped extent of each prior's full image is tracked separately for edge-erosion. |
| **5. Fuse + crop** | Inlier-weighted soft-average over the three warped masks, then foreground-cropped (the upper region is sky). What survives is a single road region in the lower image — the road, in the right place, on a frame where the road is invisible. |
| **6. Smooth** | EMA blends with the previous smoothed mask. On a frame whose matcher fails, the previous mask is held — graceful degradation rather than a flicker to nothing. |

The composition is invariant. The same six steps run on every frame; what changes is which features the matcher anchored on, which prior won, how much road is visible. Cached `FrameResult`s make downstream renders that change only the smoother or the layout near-instant — the matching pass runs once per (track, window).

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
| Render all 5 layouts of the canonical (overlay / sidebyside / 3-panel × 2 / quad) | `make reproduce-all-layouts TRACK=boreas_2021_01_26` |
| Pre-flight an alt track (verify priors-exist + summer-segmentation) | `make oracle TRACK=<id>` |
| Run the pipeline on a registered alt track | `make reproduce-track TRACK=<id>` (e.g. `boreas_2025_02_15`) |
| Bundle stills for the GitHub Pages site | `make pages-assets` |
| Run the static-stills precursor (single-prior, v1 narrative) | `make stills` |
| Run the static-stills multi-prior fusion ablation (Phase J) | `make stills-multi` |
| Render the writeup + slide PDFs (local only — gitignored) | `make pdfs` |
| Open the GitHub Pages site locally | `open docs/index.html` |
| Smoke-test the import graph + CLI shape | `make test` |
| Tidy local clutter (logs, `__pycache__`, `.DS_Store`) | `make tidy` |
| List all `make` targets | `make help` |

**Registered tracks** (Boreas snow + summer pairings, Glen Shields loop, Toronto):

| Track ID | Snow capture | Role |
| --- | --- | --- |
| `boreas_2021_01_26` | Heavy snow, mid-morning | Canonical 15 s clip (`make reproduce`) |
| `boreas_2025_02_15` | Active snowfall, late afternoon | Robustness clip — same intersection, different day (`make reproduce-track TRACK=boreas_2025_02_15`) |

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
│       ├── window_oracle.py          # pre-flight: priors-exist + summer-segmentation check
│       ├── temporal.py               # EMA / flow smoothers (EMA wins)
│       ├── overlay_render.py         # render_overlay / sidebyside / 3-panel / quad
│       ├── augment.py                # naive baseline + summer panel cache
│       ├── extract_assets.py         # extract preset stills (1.0/5.0/10.0/14.0 s) from mp4s
│       ├── fetch_track.py            # Boreas S3 fetcher with retry + --snow-start-s/--snow-end-s overrides
│       ├── render.py                 # CLI entry
│       └── render_all_layouts.py     # batch renderer (5 layouts)
│
├── data/
│   ├── demo_pairs.json               # demo manifest: 27 Mapillary snow + clear pairs the fetcher pulls
│   ├── fetch_mapillary.py            # Mapillary v4 fetcher (uses demo_pairs.json with --curated-only)
│   ├── pairs/                        # fetched pair downloads (gitignored)
│   └── video/                        # (gitignored — regenerate via `make video-fetch`)
│       └── tracks/<track_id>/        # per-track snow + summer windows
│           ├── snow/{frames/, camera_poses.csv, calib/, window.json}
│           └── summer/{frames/, camera_poses.csv, calib/, window.json}
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

## The contribution

The contribution is not a snow plough. It is a primitive — **the constants-bridge**: a composition that takes a model trained on regime A, an inference target in regime B, and a known invariant between A and B, and uses the invariant to transfer the model into B without retraining. The invariant in this work is geometric (the road sits where it sat last July) but the shape is general: anatomy across imaging modalities, terrain across illumination, scene geometry across weather, orbital schedule across polar darkness.

The snow plough is one consumer of the primitive. The road-overlay channel is what the constants-bridge looks like when consumed for buried-road perception; it is not a replacement for the perception stack. The plough's other channels (lidar, depth, obstacle detection) keep doing their work; this primitive frees them from also having to solve the road-position problem on a buried road.

## Honest limits

The output answers *where the road should be*, not *where to drive*. A snow-covered car parked on the road would still sit inside the green overlay; the system has no notion of obstacles, drivable surface, or 3D geometry. That is the scope, not a bug.

Within that scope:

- A single homography assumes a near-planar scene. When matches concentrate on building façades rather than the road plane, the warp can drift. Ground-plane bias and iterative refinement mitigate but don't eliminate.
- Heavy snow on the lens (water droplets, lens-occlusion) reduces match quality on affected frames. The system fails *gracefully* — when the matcher returns few correspondences, EMA holds the previous good mask.
- Per-sequence Boreas UTM frames don't share a global origin, so summer pairings across years require careful trajectory alignment. The `fetch_track` script handles this for the canonical pairing; non-default summer pairings are a manual operation.
- **Inlier count is not a reliable predictor of overlay quality.** A pair with many correspondences can warp the mask onto the wrong region if those correspondences concentrate on building façades. Empirical, from the static-stills work; survives into video.

## Future direction

1. **Real-time** rather than pseudo-realtime. The current matcher is the cost; replacing it with a learned cross-season-robust correspondence model trained on synthetic snow + clear pairs would close the gap. The principle is unchanged.
2. **Local 2D bird's-eye-view map** maintained over the snow drive: aggregate per-frame masks into a GPS-keyed map, reproject into camera frame on demand. Trades per-frame cost for global consistency.
3. **Operator's own prior captures** instead of Mapillary / public datasets. Same pipeline, much tighter geometric prior, no contributor-coverage gaps.

The principle is unchanged through all three: the plough never has to learn what snow looks like; it only has to find what is constant under the snow.

## Generalising

The constants-bridge is the contribution; snow on a road is one instantiation. The shape repeats across the long tail of underdata regimes that minimal-shot autonomy faces.

- **Polar Earth observation.** The invariant is the orbital geometry: known satellite passes, known coordinates. A model trained on temperate land cover or feature detection transfers into the polar regime through the orbital correspondence, without polar labels.
- **Low-light medical imaging.** The invariant is the patient's anatomy across imaging conditions. The same vessel runs in the same place; a previous well-lit acquisition or anatomical atlas supplies the constant. The well-lit-trained classifier reaches the low-light regime through registration.
- **Off-Earth manipulation** (or any environment with no in-distribution training data — deep ocean, decommissioning site, wildfire zone). The invariant is the rigid-body geometry of task and tools; a known robot pose constrains where the wrench has to be. Earth-trained perception transfers into the foreign regime through the geometry the robot already has.

In each case there is a regime where data is rich, a regime where data is sparse (sometimes structurally so), and *something* connecting the two without needing data from both sides. **Find what stays the same. Walk across.**

Constants as the bridge.

---

*Acknowledgements.* Boreas dataset (Burnett et al. 2023, UTIAS-ASRL) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Mapillary imagery under the open-data licence. Visual identity inspired by [SOTA Letters](https://sotaletters.substack.com/). Bensound *Slow Motion* (free with attribution) used in optional title-card composition. Models pretrained by their respective authors and used frozen.
