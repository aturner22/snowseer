# Snow-Underlay: Cross-Season Image Matching for Snow-Plough Autonomy

*Submission to SoTA Commission I — Minimal-Shot Autonomy.*

## Motivation

Autonomous snow-clearing vehicles are a near-textbook minimal-shot autonomy problem. Snow ploughs are deployed on huge scale at short notice, are uneconomical to staff year-round, and operate in a regime that the data which underpins modern self-driving systems deliberately excludes. Cityscapes, KITTI, nuScenes, Waymo Open — the canonical training corpora — are dominated by clear, dry roads. Asking a road-segmentation model trained on dry asphalt to know where the road is when the road is white and lane markings are gone is asking the wrong question.

The hardest single perception problem on a plough is that **the road is invisible**. Curbs, hydrants, and the line where the asphalt ends and a garden begins are all hidden. Mistakes there are not academic; they damage infrastructure. A self-driving stack trained for clear weather has no anchor in this regime.

But for almost every road in the developed world we already have a clear-season image of it — Street View, Mapillary, prior fleet captures. The road location *is known* in the prior. The plough's missing information is not what a road looks like under snow; it is *where this road sits in the camera frame right now*. That is a registration problem, not a learning problem.

## Approach

For each snowy query frame from the plough, the system pulls a clear-season prior of the same coordinates and transfers the road segmentation through a learned image registration. Concretely:

1. **Match.** A pretrained dense feature matcher (DISK + LightGlue) extracts correspondences between the snowy frame and the clear prior. Both networks are trained on MegaDepth — outdoor scenes containing zero snow — and are used frozen. They anchor on the structure that survives snow: buildings, signs, poles, rooflines, distant horizons.
2. **Align.** A homography from query → prior is estimated by USAC-MAGSAC RANSAC, restricted to correspondences in the lower image half. The restriction biases the fit toward the ground plane, which is the surface whose mask we will transfer; matches concentrating on building façades would otherwise pull the homography off the road.
3. **Segment.** A Segformer-B0 fine-tuned on Cityscapes — clear-weather European street imagery, again zero snow in training — produces a binary road mask on the **clear** prior only. The segmenter never sees snow.
4. **Warp.** The road mask is warped from the prior into the query frame via the inverse homography and overlaid.

The plough now knows where the road is, and where it isn't, even though it cannot see the road, and even though no model in the pipeline has ever been trained on a snowy frame.

```
[snow query] ─┐
              ├─> DISK + LightGlue ─> RANSAC homography (ground-plane biased)
[clear prior] ┘                              │
              └─> Segformer (Cityscapes) ─> road mask ─> warpPerspective ─> overlay
```

## Minimal-shot integrity guarantee

This is the load-bearing claim of the submission. **No snowy image was used to train any component.**

| Component | Pretraining corpus | Role of snow at any stage |
| --- | --- | --- |
| DISK | MegaDepth | None — used frozen |
| LightGlue | MegaDepth | None — used frozen |
| Segformer-B0 | Cityscapes | None — applied only to clear priors |

Snow appears only as a runtime input — the query frame whose road we want to recover. The reference frames and the query frames are both pulled from open Mapillary contributor imagery, but they are independent uploads of the same coordinates by different contributors at different times. There is no joint capture, no shared sensor, no training loop that touches snow.

## Demo / simulation environment

The brief asks for a simulation environment for the agent to navigate. We use Mapillary — open, geo-tagged, contributor-uploaded street-level imagery — as the open-world substrate. For each known-snowy region (Östersund, Tromsø, Rovaniemi, Kiruna, Anchorage, Yellowknife) the fetcher pulls all images in a bounding box, splits them by capture month into a winter set and a summer set, and pairs each winter image with its nearest-coordinate-and-heading summer image via a KD-tree. The result is a set of registered snow/clear pairs that the agent then traverses one frame at a time. Random start coordinates within a region give the bonus "randomised scenario generation" without any narrative cost.

A naive baseline — applying the same Cityscapes segmenter directly to the snowy query, no prior — is included for every pair as the contrast condition. It fails as expected: fragmented, shifted, or absent road predictions. This failure is what motivates the cross-season transfer.

## Outcomes

The fetcher pulled **125 candidate pairs** from open Mapillary contributor imagery across 6 cold-climate regions (Kiruna, Rovaniemi, Östersund-E45, Tromsø, Luleå, Gällivare, Bodø). 14 further candidate cities returned 0 pairs — a mixture of Mapillary HTTP 500s and contributor coverage gaps in winter months for southern-Nordic and North-American urban locations.

The pipeline runs end-to-end on CPU. The full notebook is `notebooks/01_walkthrough.ipynb`, cached panel images per pair are in `outputs/heroes/`, the 4-panel contact sheet is `outputs/audit/contact_sheet.png`, the Streamlit demo is `demo/streamlit_app.py`, and a 20-frame traversal video sampled at 0.5 fps is in `outputs/demo.mp4`.

### Five-stage human-in-the-loop curation funnel

| Stage | Count | What it does |
| --- | ---:| --- |
| 0. Mapillary candidates | 125 | Open-API winter+summer pairs at matched coordinates |
| 1. Auto snow-quality filter | 95 (76 %) | Laplacian sharpness + brightness + lower-half edge density |
| 2. Spatial+heading dedup (50 m / 40°) | 63 | Drops sequential frames from the same drive |
| 3. Manual snow curation (Streamlit) | 27 | User accepts only plough-realistic snow frames |
| 4. Pipeline + auto post-curation | 19 | RANSAC inliers ≥ 15, iterative refinement, largest-component cleanup |
| 5. Manual result rating | 14 demo | User rates each overlay GREAT / OKAY / NOT_GOOD / AWFUL; demo = GREAT + OKAY |

The two manual stages are not optional — the system needs a human in the loop both at the snow-input gate (Stage 3) and at the final overlay-quality gate (Stage 5). Why: even a clean snow frame and a high RANSAC inlier count can still produce a visually wrong overlay if the inliers concentrate on building façades. The headline finding from the user-rating pass is **inlier count alone is not a reliable predictor of overlay quality**: a pair with 238 inliers was rated NOT_GOOD (homography aligned the buildings, not the road plane), while a pair with 17 inliers was rated GREAT (few but well-distributed road-plane correspondences).

### Demo heroes (14, GREAT + OKAY)

10 GREAT + 4 OKAY, spread across Gällivare, Kiruna, Luleå, and Rovaniemi. Inlier counts in the GREAT bucket span **17–128** (median 38) — confirming inlier count is a weak proxy for visual correctness. Sample of the GREAT set:

- `gallivare_se__1113124` (128 inliers) — snow-banked road with parking-restriction sign; overlay precisely follows the cleared lane.
- `lulea_se__1235981` (101 inliers) — clean residential street alignment.
- `gallivare_se__724743` (83 inliers) — direct front-of-camera road buried in snow; overlay traces it accurately.
- `kiruna_se__173943` (47 inliers) — red Falun-style houses; the original v0.1 poster, still in the demo set.
- `kiruna_se__245577` (17 inliers) — low inlier count but the inliers happened to land on the road plane; clean overlay.

The 4-panel contact sheet (`outputs/audit/contact_sheet.png`) lays out **snow / clear+mask / overlay / naive** for every pair (including the rejects) so the contrast condition and the failure modes are inspectable at a glance.

### Honest limit exhibits (kept for the writeup, not the demo)

- **Drift case (`rovaniemi_fi__1263019`, 6 inliers, rated NOT_GOOD)**: the visually-compelling Revontuli tunnel-entrance — Feb 2026 snow ↔ Jul 2020 clear, 0.45 m apart. Iterative segmentation-guided refinement does not help (the snow side has no usable road-surface features), so the overlay drifts ~10 % laterally.
- **Content mismatch (`rovaniemi_fi__1457451`, 12 inliers, rated AWFUL)**: opposing carriageways of a divided highway with the same lat/lng but inverted view direction. Mapillary's `compass_angle` was wrong; the matcher found false-positive lane-marking correspondences.
- **High-inlier visual failure (`kiruna_se__191430`, 238 inliers, rated NOT_GOOD)**: the homography aligns building façades rather than the road plane, so the warped overlay sits on the snow but in the wrong place. The clearest evidence that automated curation is insufficient.

### Naive-baseline contrast condition

For every accepted pair the pipeline also runs the same Mask2Former segmenter **directly on the snow frame** (no cross-season prior, no matching, no warp) and persists `*__naive_baseline.png`. The naive output is fragmented or absent on snow-covered roads, providing the contrast that motivates the cross-season approach. The pipeline computes both `IoU(overlay, naive)` and `IoU(overlay, identity-warp)` per pair (the latter being the "trust the prior with no alignment" condition); both are persisted in `summary.json` and surfaced as live metrics in the Streamlit viewer.

## Future direction

Three obvious extensions, in order of impact and complexity:

1. **Replace single-homography with a piecewise-affine warp** estimated from semantic ground-plane segmentation of both images. Removes the planar-scene assumption; correctly handles slopes and curves.
2. **Lift to the temporal domain.** Smooth overlays across consecutive frames using the plough's odometry; reject homographies that contradict the previous frame's pose update.
3. **Replace Mapillary with the plough operator's own prior captures.** Same pipeline, much tighter geometric prior, no contributor-coverage gaps.

The spirit of the contribution is unchanged in all three: the plough never has to learn what snow looks like; it only has to find what is constant under the snow.
