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

The fetcher pulled **39 paired snowy/clear frames** from open Mapillary contributor imagery across Kiruna (Sweden), Rovaniemi (Finland), Östersund-E45 (Sweden — the brief's reference location), and Tromsø (Norway). Five further candidate regions (Östersund city centre, Sundsvall, Anchorage, Yellowknife, Sundsvall) had no usable winter imagery within a 5 m / ±20° heading window despite live Mapillary coverage in summer — a real-world contributor-coverage gap, not a code defect. The pipeline runs end-to-end on CPU; the full notebook is `notebooks/01_walkthrough.ipynb`, cached panel images per pair are in `outputs/heroes/`, and the Streamlit demo is `demo/streamlit_app.py`. A 17-frame traversal video sampled at 0.5 fps is in `outputs/demo.mp4`.

A **content-level curation step** (`data/curate_pairs.py`) — RANSAC inlier count under the ground-plane-biased homography, threshold 15 — accepts **15 / 39** pairs (38 %). The remainder are largely Mapillary heading-metadata false positives (opposing carriageways of a divided road, different streets at the same lat/lng) plus a long-tail of motion-blurred night captures and snow-at-night graceful failures. The audit (`docs/audit.md`) documents this honestly.

The accepted pairs split into:

- **Clean wins (12 pairs, 26–238 inliers):** Kiruna wide-curved-intersection (238 inliers, road fully buried, overlay tracks the rightward sweep — the new poster); Kiruna red-Falun-houses (47 inliers, road fully buried); Kiruna brick residential (74 inliers); Kiruna snow-banked residential (43 inliers); residential street with parked car (83 inliers, overlay threads past the obstacle); three Rovaniemi tunnel interiors (55, 98, 101 inliers); Rovaniemi night-vs-day intersection (73 inliers, dramatic lighting change); Rovaniemi highway-under-bridge (47 inliers).
- **Borderline mid-quality (3 pairs, 19–33 inliers):** content-borderline pairs where the inlier count clears the threshold but the visual is from a slightly different lane / approach — included to be honest about the limit of automated curation.

Two pairs were retained as **deliberate honest-limit exhibits**:

- **Drift case (Rovaniemi Revontuli tunnel-entrance, 6 inliers):** Feb 2026 snow ↔ Jul 2020 clear, 0.45 m apart — the visually most-compelling pair, but only 6 lower-half matches survived RANSAC; iterative segmentation-guided refinement does not help (the snow side has no usable road-surface features), so the overlay drifts ~10 % laterally. Honest about the planar-scene homography limit.
- **Graceful failure (Kiruna snow-at-night, 0 inliers):** matcher finds 4 candidate matches, RANSAC rejects all, **no overlay produced**. Safety-positive failure mode for a plough — no overlay beats a wrong overlay near a hydrant.

The expected naive baseline failure — running the same Cityscapes Segformer directly on the snowy frame — produces fragmented, sky-mistaken-for-road, or absent road predictions. The pipeline computes both `IoU(overlay, naive)` and `IoU(overlay, identity-warp)` (the "what if we trusted the prior with no alignment" condition) and persists them per pair; the Streamlit viewer surfaces both as live metrics.

## Future direction

Three obvious extensions, in order of impact and complexity:

1. **Replace single-homography with a piecewise-affine warp** estimated from semantic ground-plane segmentation of both images. Removes the planar-scene assumption; correctly handles slopes and curves.
2. **Lift to the temporal domain.** Smooth overlays across consecutive frames using the plough's odometry; reject homographies that contradict the previous frame's pose update.
3. **Replace Mapillary with the plough operator's own prior captures.** Same pipeline, much tighter geometric prior, no contributor-coverage gaps.

The spirit of the contribution is unchanged in all three: the plough never has to learn what snow looks like; it only has to find what is constant under the snow.
