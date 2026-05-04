# Snow-Underlay

> *Constants as the bridge.* A demonstration of minimal-shot autonomy: extending a model trained on one regime into the regime where we lack data, by anchoring on what stays the same between them.

A submission to [SoTA Commission I — Minimal-Shot Autonomy](https://sotaletters.substack.com/p/sota-commission-i-minimal-shot-autonomy).

---

## What this project is really about

Self-driving systems are trained on dry roads, deliberately. Every canonical autonomy dataset — Cityscapes, KITTI, nuScenes, Waymo Open — is dominated by clear-weather European or American highways under daylight. A system trained on this corpus and asked to operate when the road is buried in snow has not been asked a research question. It has been asked the wrong question.

The familiar response is *we just need more data*. Annotate snowy roads. Annotate dust storms. Annotate fog, lava, oil spills, washouts. There are 27 million miles of road in the world, and the long tail of conditions that road can be in is longer than the road itself. We are not going to label our way out of it.

There is a different move available. **For almost every operating regime where autonomy fails for lack of data, there is an adjacent regime — temporally, seasonally, geographically — where we have plenty of data, and where the *parts that matter* are the same.** A snow plough's road is the same road it was last July. The curb hasn't moved. The hydrant hasn't moved. The road's *appearance* has changed completely; its *position in space* has not.

If we can identify what stays constant between the data-rich regime and the data-poor one, we can extend our existing models into the new regime without learning a single thing about it. We use the constants as a bridge.

This project is one concrete instantiation of that idea, applied to autonomous snow ploughs. The principle is general; the snow plough is the vehicle.

## The example

A snow plough operates in a regime that self-driving deliberately excludes. Curbs, lane markings, hydrants, and the seam between asphalt and garden are all hidden. Mistakes there damage infrastructure — a wrong overlay near a fire hydrant is not a research finding, it is a bill.

But for almost every road in the developed world there is a *clear-season image of it*. Street View, Mapillary, the operator's own prior captures — the road's location is *known* in the prior. The plough's missing information is not what a road looks like under snow. It is *where this road sits in the camera frame right now*. That is a registration problem. We solved registration twenty years ago.

So:

1. Pull the live snowy frame from the plough's camera.
2. Pull a clear-season prior of the same coordinates from any open imagery substrate (we use Mapillary).
3. Match the two images using a generic, frozen feature matcher. The matcher anchors on what stays constant between the two: buildings, signs, poles, rooflines, distant horizons.
4. Estimate a homography from the matches, biased toward the ground plane.
5. Run a generic Cityscapes-trained road segmenter on the **clear** prior — never on the snow frame.
6. Warp the road mask through the homography onto the snowy frame.

The plough now knows where the road is, and where it isn't, even though it cannot see the road, even though no model in the pipeline has ever been trained on a snowy frame.

![Hero panel — Gällivare snow-banked road with parking-restriction sign; cross-season overlay tracks the cleared lane between the snow piles.](outputs/heroes/gallivare_se__1113124103239974__202392698419785__panel.png)

## Architecture

| Component | Role | Pretrained on | Reference |
| --- | --- | --- | --- |
| **DISK** | Local feature detector + descriptor | MegaDepth (outdoor scenes, no snow) | Tyszkiewicz et al., *DISK: Learning local features with policy gradient*, NeurIPS 2020 |
| **LightGlue** | Sparse feature matcher | MegaDepth | Lindenberger et al., *LightGlue: Local Feature Matching at Light Speed*, ICCV 2023 |
| **USAC-MAGSAC** | Robust homography fit | — | Barath et al., *MAGSAC++, a fast, reliable and accurate robust estimator*, CVPR 2020 |
| **Mask2Former** | Semantic road segmentation on the clear prior only | Cityscapes (clear-weather European driving, no snow) | Cheng et al., *Masked-attention Mask Transformer for Universal Image Segmentation*, CVPR 2022 |
| **Mapillary API v4** | Open imagery substrate (snow + clear queries) | — | mapillary.com/developer/api-documentation |

Every learned component is frozen. Nothing is trained, nothing is fine-tuned. Snowy imagery enters the system only at inference time.

## Minimal-shot integrity

| Claim | Status |
| --- | --- |
| Zero snowy frames touch any model weights | ✓ |
| Zero snowy frames touch any annotation pipeline | ✓ |
| Snow appears only as runtime input | ✓ |
| Reproducible from a clean clone with one command | ✓ |

## Running it

```bash
uv sync --python 3.12
export MAPILLARY_TOKEN=<token from https://www.mapillary.com/dashboard/developers>
make demo
```

That fetches the 14 curated demo pairs, runs the cross-season pipeline, builds the contact sheet, and renders the demo video. The full set of `make` targets is documented in the [`Makefile`](Makefile); `make help` lists them.

| Want to … | Run |
| --- | --- |
| Reproduce the demo from scratch | `make demo` |
| Browse the result in a clickable viewer | `make stream` |
| Re-execute the walkthrough notebook | `make notebook` |
| Render the writeup + slide PDFs | `make pdfs` |

## Repo layout

```
snow-underlay/
├── Makefile                       # one-command reproduction
├── data/
│   ├── curated_pairs.json         # the 14 pairs that constitute the v1.0 demo
│   ├── fetch_mapillary.py         # --curated-only mode + exploration mode
│   └── pairs/<id>/{snow.jpg, clear.jpg, meta.json, snow_quality.json}
├── src/
│   ├── matching.py                # DISK + LightGlue
│   ├── homography.py              # RANSAC, ground-plane biased, iterative refinement
│   ├── segmentation.py            # Mask2Former
│   ├── overlay.py                 # warp + 4-panel figure
│   ├── snow_quality.py            # auto pre-filter (sharpness · brightness · edges)
│   ├── audit.py                   # 4-column contact sheet
│   ├── video.py                   # auto-rendered demo video with music
│   └── pipeline.py                # end-to-end orchestration
├── demo/
│   ├── streamlit_app.py           # judges-clickable viewer over cached outputs
│   ├── curate_snow.py             # Streamlit big-image accept/reject for snow frames
│   └── curate_results.py          # Streamlit overlay-quality rater
├── notebooks/
│   └── 01_walkthrough.ipynb       # story up top, technical replication below
├── outputs/
│   ├── heroes/                    # per-pair panel PNGs + summary.json + INDEX.md
│   ├── audit/                     # contact sheet
│   └── demo.mp4
├── docs/
│   ├── style/style.md             # visual identity (charcoal · cream · rust)
│   ├── writeup.{md,pdf}           # ≤ 2-page essay
│   └── slides.{md,pdf}            # Marp deck
└── assets/
    ├── fonts/                     # EB Garamond · Inter · JetBrains Mono (OFL)
    └── audio/                     # ambient piano + outdoor sound for the video
```

## Honest limits

- A single homography assumes a near-planar scene. When matches concentrate on building façades rather than the road plane, the warp can drift. We bias toward the ground plane and apply iterative refinement; this mitigates but does not eliminate.
- Heavy snow (frosted trees, fully white ground, low contrast) starves the matcher of usable structure. The system fails *gracefully* — low inlier counts trigger no overlay rather than a confidently wrong one.
- Mapillary's `compass_angle` is contributor-uploaded and occasionally wrong, producing tight metadata pairs that are visually different scenes. We catch these via a content-level RANSAC inlier threshold and a final manual review.
- **Inlier count is not a reliable predictor of overlay quality.** A pair with 238 inliers can still warp the mask onto the wrong region if the inliers concentrate on building façades. The system therefore needs a human in the loop on both the snow input and the overlay output. We built two Streamlit raters (`demo/curate_snow.py` and `demo/curate_results.py`) for this.

## Future direction

Three extensions, in increasing ambition:

1. **Piecewise-affine warp** estimated from a semantic ground-plane segmentation of both images. Drops the planar-scene assumption; correctly handles slopes and curves.
2. **Temporal smoothing.** Use the plough's odometry to chain consecutive frames; reject homographies that contradict the previous frame's pose update.
3. **Replace Mapillary with the operator's own prior captures.** Same pipeline, much tighter geometric prior, no contributor-coverage gaps.

The principle is unchanged through all three: the plough never has to learn what snow looks like; it only has to find what is constant under the snow.

## Closing

This demo is small: a pretrained matcher, a pretrained segmenter, a homography, fourteen hero pairs from northern Sweden and Finland. We chose it because it is small enough to verify and large enough to demonstrate the move.

The move is the contribution. Where data is missing, find a regime where it isn't; identify what is constant between the two; transfer through the constant. Snow on a road is one example. Polar imaging without polar training data, low-light medical imaging without low-light training data, a manipulator on Mars without Mars training data — all admit the same structure.

Constants as the bridge.

---

*Acknowledgements.* The visual identity draws on [SOTA Letters](https://sotaletters.substack.com/) for tone and minimal-monochrome layout; the rust accent is our own. Imagery from [Mapillary](https://www.mapillary.com/) under the open-data license. Models pretrained by their respective authors and used frozen. Repository licensed under MIT.
