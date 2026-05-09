# Snowseer

> Achieving minimal-shot autonomy by recognising constants across environments.

Snowseer infers the position of the road buried under the snow in a snow plough's camera by transferring a road segmentation from a clear-season image of the same place. Nothing in the pipeline has been trained or fine-tuned on snow. The matcher anchors on features that survive the seasonal change (gateposts, fence wires, masonry corners, distant roof edges) and the homography fitted to those features carries the prior's road mask through to the snow frame.

Submitted to [SoTA Commission I: Minimal-Shot Autonomy](https://sotaletters.substack.com/p/sota-commission-i-minimal-shot-autonomy), May 2026. Full writeup on the companion site at [aturner22.github.io/snowseer](https://aturner22.github.io/snowseer/). Interactive walkthrough at [`docs/analysis.ipynb`](docs/analysis.ipynb).

## Quick start

```bash
git clone https://github.com/aturner22/snowseer
cd snowseer
uv sync --python 3.12
export MAPILLARY_TOKEN=<token from https://www.mapillary.com/dashboard/developers>
make reproduce
```

`make reproduce` runs three steps sequentially (~3 hours on Mac CPU): the January 2021 canonical clip, the February 2025 robustness clip, and the 18-pair static-stills precursor.

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
| `make reproduce` | Both Toronto clips + 18-pair stills, sequentially |
| `make track TRACK=<id>` | Full pipeline on one registered track |
| `make stills` | Static-stills demo on 18 Mapillary pairs (`MAPILLARY_TOKEN` required) |
| `make test` | Smoke tests, no compute |
| `make clean` | Wipe regenerable outputs |
| `make help` | List all targets |

Registered Toronto tracks:

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

## Limitations and next steps

The full discussion is in the [Limitations](https://aturner22.github.io/snowseer/#limitations) and [Next steps](https://aturner22.github.io/snowseer/#next-steps) sections of the site. In short: artefacts in the summer prior (capture-vehicle bonnet, parked cars) carry forward into the overlay; the matcher takes around 16 s per frame on Mac CPU, so live operation needs a faster matcher; the current implementation is geared toward the specific demo material and generalising to any road with Street View or operator captures is a natural next step.

## Acknowledgements

Boreas dataset (Burnett et al., *Boreas: A Multi-Season Autonomous Driving Dataset*, IJRR 2023) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Mapillary imagery under the Mapillary open-data licence. Models pretrained by their respective authors and used frozen.
