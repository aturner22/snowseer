# Snowseer

> Achieving minimal-shot autonomy by recognising constants across environments.

Snowseer infers the position of the road buried under the snow in a snow plough's camera by transferring a road segmentation from a clear-season prior of the same place. Nothing in the pipeline has been trained or fine-tuned on snow. The matcher anchors on features that survive the seasonal change (gateposts, fence wires, masonry corners, distant roof edges) and the homography fitted to those features carries the prior's road mask through to the snow frame.

Submitted to [SoTA Commission I: Minimal-Shot Autonomy](https://sotaletters.substack.com/p/sota-commission-i-minimal-shot-autonomy), May 2026. Companion site at [`docs/index.html`](docs/index.html). Interactive walkthrough at [`docs/analysis.ipynb`](docs/analysis.ipynb).

## Quick start

```bash
git clone https://github.com/aturner22/snowseer
cd snowseer
uv sync --python 3.12
export MAPILLARY_TOKEN=<token from https://www.mapillary.com/dashboard/developers>
make reproduce
```

`make reproduce` runs three steps sequentially (~3 hours on Mac CPU): the January 2021 canonical clip, the February 2025 robustness clip, and the 18-pair static-stills precursor. `make help` lists every other target.

## Minimal-shot autonomy

Minimal-shot autonomy is the question of how a perception system survives in regimes it has not been heavily trained on. The default answer is to collect more data and retrain. That answer assumes labelling can keep pace with reality. It cannot, not for snow, dust, smoke, washouts, or any of the long-tail conditions a deployed vehicle, robot, or drone meets in the real world.

For almost every operating environment where autonomy fails for lack of data, an adjacent regime exists, temporally or seasonally or geographically, where data is plentiful and rich, and whose key components remain the same across environments. The road that needs to be ploughed this winter is the same road it was in the summer. Its appearance has changed, but its position in space has not.

## The constants-bridge

The contribution is not a snow plough. It is a primitive. A composition that takes a model trained on regime A, an inference target in regime B, and a known invariant linking A and B, and uses the invariant to transfer the model into regime B without retraining. The invariant in this work is geometric (the road sits where it sat last summer), but the shape is general: anatomy across medical imaging environments, terrain across illumination states, scene structure across weather conditions.

The constituent parts are not new. Geometric scene analysis, classical RANSAC, and pretrained feature matchers and segmenters have been combined in many ways across the computer-vision literature. The contribution is to identify the composition of the environment itself as a key feature to use, and to give an end-to-end working demonstration. The feature matcher is not generalising its recognition of "snow". It has not been trained on snow. The focus is generalisation via what stays the same.

## The Snowseer system

A pre-trained feature matcher establishes correspondences between the live snow frame and a clear-season prior of the same coordinates. A homography (projection transformation) is fitted to those correspondences and geometrically connects the clear-season prior to the snow frame. A pre-trained segmenter produces a road mask on the *clear* prior, and that mask is warped through the homography onto the winter image, producing an overlay of where the road is underneath the snow.

Per snow frame, six steps:

1. Pull the live snowy frame from the plough's camera.
2. Pull a clear-season prior of approximately the same coordinates.
3. Examine the two with DISK + LightGlue feature-matching models.
4. Estimate a homography via a USAC-MAGSAC RANSAC model.
5. Run a Mask2Former segmenter on the *clear* prior only, to obtain the true position of the road in clear conditions.
6. Warp the road mask into the snow frame via the homography and overlay onto the plough's visuals.

The video processor wraps this per-still-pair pipeline in three layers: a track loader indexing snow and summer streams by GPS pose, a prior pool returning the K = 3 nearest summer captures by distance for each snow frame, and an exponential moving average (α = 0.4) on the binary road mask to produce a smoother continuous render of the transferred road position.

## Components

| Component | Role | Model | Dataset |
| --- | --- | --- | --- |
| Feature detector | Locate keypoints in each image | **DISK** *(NeurIPS '20)* | MegaDepth |
| Feature matcher | Pair keypoints across the snow / summer images | **LightGlue** *(ICCV '23)* | MegaDepth |
| Homography fit | Robust geometric registration of the pair | **USAC-MAGSAC** *(CVPR '20)* | n/a |
| Road segmenter | Produce a road mask on the summer image | **Mask2Former** *(CVPR '22)* | Cityscapes |
| Driving captures | Paired snow + summer Toronto traversals | n/a | **Boreas** *(IJRR '23, CC BY 4.0)* |

Every learned component is frozen. Snowy imagery enters the system only at inference time.

## Make targets

| Target | Action |
| --- | --- |
| `make reproduce` | January 2021 clip + February 2025 clip + 18-pair stills, sequentially |
| `make track TRACK=<id>` | Full pipeline on one registered track |
| `make stills` | Static-stills demo on 18 Mapillary pairs (`MAPILLARY_TOKEN` required) |
| `make notebook` | Re-execute `docs/analysis.ipynb` in place |
| `make test` | Smoke tests, no compute |
| `make clean` | Wipe regenerable outputs |
| `make help` | List all targets |

Registered tracks (Boreas snow + summer pairings, Toronto):

| Track ID | Snow capture | Role |
| --- | --- | --- |
| `boreas_2021_01_26` | Heavy snow, mid-morning | January 2021 canonical clip (14 s) |
| `boreas_2025_02_15` | Active snowfall, late afternoon | February 2025 robustness clip (34 s) |

## Repo layout

```
snowseer/
├── Makefile
├── README.md
├── pyproject.toml · uv.lock
│
├── src/
│   ├── matching.py                    # DISK + LightGlue
│   ├── homography.py                  # USAC-MAGSAC, ground-plane biased
│   ├── segmentation.py                # Mask2Former
│   ├── overlay.py                     # warp + binary mask primitives
│   ├── pipeline.py                    # static cross-season pair pipeline
│   ├── data/
│   │   ├── demo_pairs.json            # 18 Mapillary snow + clear pair manifest
│   │   └── fetch_mapillary.py         # Mapillary v4 fetcher
│   └── video_runtime/
│       ├── track.py                   # snow + summer stream loaders
│       ├── prior_pool.py              # K-NN prior selection by UTM
│       ├── pipeline_v.py              # per-frame pipeline, EMA smoother, cache + checkpoint resume
│       ├── overlay_render.py          # overlay / sidebyside / matches / quad renderers
│       ├── render.py                  # render CLI
│       ├── render_all_layouts.py      # batch render every layout for a track
│       ├── augment.py                 # naive baseline + summer panel cache
│       ├── matches_pass.py            # match-overlay sidecar
│       ├── extract_assets.py          # extract preset stills from rendered mp4s
│       └── fetch_track.py             # Boreas S3 fetcher
│
├── docs/
│   ├── index.html                     # GitHub Pages site
│   ├── analysis.ipynb                 # interactive walkthrough
│   ├── style/                         # site CSS
│   └── assets/                        # site media (stills, clips, fonts, favicon)
│
├── outputs/                           # regenerable, gitignored
│   ├── nordic_stills/                 # static-stills pipeline outputs
│   └── toronto_video/<track>/         # video pipeline outputs + matching cache
│
└── tests/test_smoke.py
```

## Limitations

Some artefacts in the overlay are inherited from the summer prior. Where the front of the summer capture vehicle is visible, the warped road mask begins a short distance ahead of the snow camera rather than directly under it. Where a parked car or other obstacle sits on the road in the prior, the segmenter routes the road class around the obstacle and the overlay carries the cutout forward. Both are tractable with reasonable engineering, for example by extrapolating the road below the visible mask boundary, or by fusing several priors of the same scene so that any one prior's foreground occlusions are filled in by the others.

The pipeline is not, currently, real-time. The matching pass dominates per-frame compute, taking around 16 s per frame on Mac CPU. Demo clips build end-to-end in roughly an hour. Real-time operation needs a substantially faster matcher and segmenter, which is a deployment-engineering problem rather than a research question.

The system is not, currently, deployable arbitrarily. The current implementation is geared toward the specific Toronto and nordic demo material. Generalising to operate on any road with Google Street View or a comparable source available is feasible (the pipeline is substrate-agnostic in principle), but is a future integration step.

## Next steps

1. **Real-time matcher.** Bring per-frame matching from around 16 s on Mac CPU to under 1 s on a deployment-class device. Required for live operation.
2. **Visual place recognition front-end.** Replace GPS-pose lookup with a learned recognition step so the appliance works in GPS-denied environments and without prior pose.
3. **Multi-source clear-season image bank.** Integrate Mapillary global, Street View, and operator captures so any covered road can be a deployment target.
4. **Hardware prototype.** A battery-powered processing unit running the live appliance with a simple HUD-style output.

## Acknowledgements

Boreas dataset (Burnett et al., *Boreas: A Multi-Season Autonomous Driving Dataset*, IJRR 2023) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Mapillary imagery under the Mapillary open-data licence. Models pretrained by their respective authors and used frozen.
