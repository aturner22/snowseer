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

## How it works (the inner workings, step by step)

The system is one Python pipeline (`src/pipeline.py`) that takes a `pair_dir` containing `snow.jpg`, `clear.jpg`, and `meta.json` and produces a 4-panel figure plus a JSON summary. Each step below corresponds to one chunk of `pipeline.run_pair()`.

### 1. Pair sourcing — `data/fetch_mapillary.py`

The fetcher operates in two modes:
- **`--curated-only`** (the canonical reproducibility path, what `make demo` calls): reads `data/curated_pairs.json`, which contains the 14 pair IDs of the v1 demo set. For each pair it issues a single Mapillary Graph API call per image (`GET /<image_id>?fields=id,geometry,captured_at,thumb_2048_url,…`) to get a fresh signed thumbnail URL (Mapillary URLs expire), downloads `snow.jpg` + `clear.jpg`, writes `meta.json`. Idempotent — skips a pair if its files are already on disk.
- **Exploration mode** (no flag): used when curating a *new* demo set. For each region in `REGIONS` it queries by `bbox + start_captured_at + end_captured_at`, splits the results into a winter set (Dec–Mar) and a summer set (May–Sep), and pairs each winter image with its nearest summer neighbour by lat/lng (`BallTree` + haversine, ≤ 5 m) and heading (≤ ±20°).

### 2. Snow-quality filter — `src/snow_quality.py`

Mapillary contributors upload a lot of motion-blurred night drives and windshield-blocked frames that aren't representative of a snow plough's operating regime. Before any user time is spent, three cheap metrics are computed on the lower 70 % of each snow image (the road region):

- **Sharpness** = variance of the Laplacian. Higher = sharper. Drops blurred frames.
- **Brightness** = median of the V channel of HSV. Drops near-black night frames.
- **Edge density** = fraction of pixels lit by Canny on the lower half. Drops featureless / windshield-obscured frames.

Each metric is rank-normalised across the candidate pool to a percentile in `[0, 1]`; the three percentiles are averaged into a `composite` score and persisted to `data/pairs/<id>/snow_quality.json`.

### 3. Manual snow curation — `demo/curate_snow.py`

The user reviews the auto-passed snow frames in a Streamlit app sorted by composite quality (best first). Spatial+heading dedup (50 m / 40°) collapses sequential frames from the same drive so the user sees one representative per cluster. Decisions persist to `data/manual_snow_curation.json`.

### 4. Feature matching — `src/matching.py`

For each accepted pair, both images are resized to a common max-dimension of 1024 px and converted to an RGB tensor. **DISK** (`KF.DISK.from_pretrained("depth")`) extracts up to 2048 keypoints + 128-dim descriptors per image. **LightGlue** (`KF.LightGlueMatcher("disk")`) then matches the two descriptor sets, producing `(idx_a, idx_b)` pairs and per-match distances. The output is paired pixel coordinates `(N, 2)` for both images plus a per-match confidence in `[0, 1]`.

### 5. Homography estimation — `src/homography.py`

The matches are filtered to **ground-plane candidates only**: keypoints whose `y` coordinate is in `[0.5 H, 1.0 H]` of their respective image. This prevents matches on building façades from dominating the homography fit, which would align the buildings rather than the road plane.

`cv2.findHomography(src, dst, cv2.USAC_MAGSAC, ransacReprojThreshold=3.0)` fits the homography by RANSAC. Output: a `3×3` matrix `H` mapping snow → clear, plus an inlier mask. If fewer than 8 ground-plane matches survive, the restriction is dropped and we fit on all matches; if that fails too, the pair is rejected.

**Iterative refinement** (`refine_iteratively`): if initial inliers < 25, the pipeline computes the road mask on the clear prior (next step), warps it into snow image space, and re-fits the homography only on matches whose snow keypoint sits inside the warped road region. This is the user-original-intuition "wiggle features until they line up" loop, with explicit semantic gating instead of pixel-level optimisation.

### 6. Road segmentation — `src/segmentation.py`

`facebook/mask2former-swin-tiny-cityscapes-semantic` is run on the **clear** prior only (never on the snow frame). The Hugging Face `Mask2FormerImageProcessor`'s `post_process_semantic_segmentation` returns the per-pixel argmax over Cityscapes' 19 classes; we keep `class == 0` (road).

The resulting binary mask is reduced to its single largest 8-connected component (`overlay.keep_largest_component`, ≥ 500 px). A snow plough cares about the *one* drivable surface in front of it, not the long tail of disconnected sidewalks, distant road patches, or warp-aliased islands.

### 7. Mask transfer — `src/overlay.py`

The road mask is warped from the clear-prior image into the snow image space via `cv2.warpPerspective(mask, np.linalg.inv(H), snow.shape)`. The result is reduced to its largest component again (warp aliasing can leave a few pixels of road outside the main blob). The result is alpha-blended onto the snow frame in **green** (`#2e9c56`) — the same green used on the prior road mask, signalling *same road, transferred*.

### 8. Naive baseline (the contrast condition)

The same Mask2Former segmenter is run **directly on the snow frame**, producing a road mask in red (`#dc3c32`). The contrast in the 2×2 panel is the demonstration: the segmenter trained on dry asphalt either predicts road on the wrong region (visible red on snow piles, façades, sky) or predicts nothing at all. Either reads as failure.

### 9. Result curation — `demo/curate_results.py`

The user rates each overlay panel on a 4-point scale: GREAT / OKAY / NOT_GOOD / AWFUL. Decisions persist to `data/manual_result_curation.json`. The GREAT+OKAY survivors are baked into `data/curated_pairs.json` and become the canonical demo set. The empirical finding behind this stage: **inlier count is not a reliable predictor of overlay quality** (a pair with 238 inliers landed NOT_GOOD because the inliers concentrated on building façades; a pair with 17 inliers landed GREAT because the few inliers it had hit the road plane). Automated metrics are insufficient; a final human pass is structural.

### 10. Outputs and metrics

For each pair the pipeline persists:
- `outputs/heroes/<id>__panel.png` — the headline 2×2 figure (snow / naive ; clear+mask / overlay)
- `outputs/heroes/<id>__overlay.png` — snow + green overlay only (for slides / video)
- `outputs/heroes/<id>__naive_baseline.png` — snow + red naive prediction
- `outputs/heroes/<id>__matches.png` — feature correspondences (green = inlier, red = rejected)
- A row in `outputs/heroes/summary.json` with `n_matches`, `n_inliers`, `accept`, `iou_overlay_vs_naive`, `iou_overlay_vs_identity`, etc.

The 4-panel contact sheet (`make audit` → `outputs/audit/contact_sheet.png`) stacks every pair's panels into a single tall image for comparison.

The auto-rendered video (`make video` → `outputs/demo.mp4`) sequences title cards, two procedural diagrams (the *bridge* between regimes, and the 6-step pipeline), and a two-slide breakdown per hero (problem framing then solution), bedded with a procedural ambient pad in `assets/audio/music.mp3`. ~3 min, 1080p, 30 fps. The narrative beats live in `SCENES` at the top of `src/video.py`.

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

*Acknowledgements.* The visual identity draws on [SOTA Letters](https://sotaletters.substack.com/) for tone and minimal-monochrome layout; the rust accent is our own. Imagery from [Mapillary](https://www.mapillary.com/) under the open-data license. Demo video music: ["Slow Motion" by Bensound](https://www.bensound.com), free with attribution under their licence. Models pretrained by their respective authors and used frozen. Repository licensed under MIT.
