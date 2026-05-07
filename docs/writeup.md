---
title: "Constants as the bridge"
subtitle: "Minimal-shot autonomy, in motion"
author: "Snow-Underlay · SoTA Commission I · 2026-05-10"
geometry: margin=2cm
mainfont: "EB Garamond"
sansfont: "Inter"
monofont: "JetBrains Mono"
fontsize: 11pt
linestretch: 1.35
colorlinks: true
linkcolor: "gray"
---

A snow plough's job is short: keep the road clear. The catch is that, while the plough is doing it, the road is invisible. Curbs are buried, lane markings are gone, the seam between asphalt and garden is no longer drawn. A self-driving stack trained on Cityscapes will report, with calibrated confidence, that the entire scene is sky.

Minimal-shot autonomy is the question of how a perception system survives in regimes it has not been heavily trained on. The default answer is *collect more data and retrain*. That answer assumes labelling can keep pace with reality. It cannot — not for snow, dust, ash, washouts, regional construction practices, agricultural off-road, novel industrial sites, or any of the countless conditions a vehicle, robot, or drone meets when it leaves the regime its training set was sampled from. **A perception system that depends on having been trained on each new condition will lag every condition it has not yet been trained on.**

There is a different move available, and it doesn't require new training data. **For almost every operating regime where autonomy fails for lack of data, there is an adjacent regime — temporally, seasonally, geographically — where data exists, and where the parts that matter are the same.** A snow plough's road is the same road it was last July. The curb hasn't moved. The hydrant hasn't moved. The road's *appearance* has changed completely; its *position in space* has not.

If we can identify what stays constant between the data-rich regime and the data-poor one, we can extend our existing models into the new regime without learning a single thing about it. **We use the constant as a bridge.** This is generalisation, not memorisation: the system reaches a regime its components have never been trained on by anchoring on what does not change between the two.

This essay is one concrete instantiation of that idea, applied to autonomous snow ploughs. The principle is general; the snow plough is the demonstration. The headline artefact is a video: a continuous overlay of road position on a snow-buried street, frame by frame, produced by a pipeline whose only learned components have never seen snow — and yet they generalise into snow because the geometric correspondence between snow and clear-season imagery does the bridging for them.

## The example

Self-driving systems are trained on dry roads, deliberately. Cityscapes, KITTI, nuScenes, Waymo Open — the canonical training corpora — are dominated by clear-weather European or American highways under daylight. A snow plough operates in the regime that data deliberately excludes. Mistakes there damage infrastructure. The familiar move — collect more snowy data, annotate it, retrain — is uneconomic at the scale of a 27-million-mile road network.

But the road's *location* hasn't moved. For almost every road in the developed world there is a clear-season image of it: Street View, Mapillary, the operator's own prior captures, the dataset's own summer traversal. The plough's missing information is not what a road looks like under snow. It is *where this road sits in the camera frame right now*. That is a registration problem, not a learning problem. We solved registration twenty years ago.

So the system, in six steps:

1. Pull the live snowy frame from the plough's camera.
2. Pull a clear-season prior of the same coordinates. The substrate is interchangeable — Mapillary, Google Street View, the operator's own clear-weather drives, the host dataset's summer subset. We use Boreas's matched summer captures for the canonical video demo (cm-accurate Applanix poses) and Mapillary for the static-stills precursor (open contributor imagery), but the principle is substrate-agnostic.
3. Match the two images using a generic, frozen feature matcher. The matcher anchors on what stays constant between the two: buildings, signs, poles, rooflines, distant horizons.
4. Estimate a homography from the matches, biased toward the ground plane.
5. Run a generic Cityscapes-trained road segmenter on the **clear** prior — never on the snow frame.
6. Warp the road mask through the homography onto the snowy frame.

The plough now knows where the road is, and where it isn't, even though it cannot see the road, and even though no model in the pipeline has ever been trained on a snowy frame.

## Architecture

DISK (Tyszkiewicz et al., NeurIPS 2020) extracts local features. LightGlue (Lindenberger et al., ICCV 2023) matches them. USAC-MAGSAC (Barath et al., CVPR 2020) fits a homography by RANSAC, restricted to lower-image matches to bias toward the ground plane. Mask2Former (Cheng et al., CVPR 2022), pretrained on Cityscapes (Cordts et al., CVPR 2016), produces the road mask on the clear prior. The mask (and its warp into snow image space) is reduced to its single largest connected component, because a plough cares about the *one* drivable surface in front of it.

The dataflow is:

```
   ┌──────────────┐                    ┌──────────────────────┐
   │  Snow frame  │                    │  Clear-prior frame    │  any geo-tagged
   │   (live)     │                    │  (Boreas summer,      │  clear-weather
   │              │                    │   Mapillary, GSV…)    │  imagery substrate
   └──────┬───────┘                    └──────────┬────────────┘
          │                                       │
          │                                       ▼
          │                            ┌──────────────────────┐
          │                            │   Mask2Former         │  frozen
          │                            │  (Cityscapes road)    │  Cityscapes
          │                            └──────────┬────────────┘
          │                                       │ road mask in prior space
          │                                       │
          └──────────►  DISK + LightGlue  ◄───────┘             frozen
                              │                                  MegaDepth
                              ▼ correspondences
                    USAC-MAGSAC homography                       classical
                    (ground-plane biased)
                              │
                              ▼ H
                  warp prior mask → snow space
                              │
                              ▼
              fuse over K=3 priors  +  EMA over time
                              │
                              ▼
                  ┌──────────────────────┐
                  │  Road overlay on the │
                  │  snow frame (green)  │
                  └──────────────────────┘
```

## How it works — a worked example

The diagram describes the recipe in the abstract; one frame shows the recipe in operation. We pick a representative frame from late in the canonical clip — Glen Shields, residential, mid-morning, heavy snow on the pavement — and trace it through.

1. **Prior selection.** The summer prior pool's KD-tree on summer poses returns the three nearest summer captures by UTM distance. The canonical loop's summer trajectory is sampled densely; on this frame all three nearest summer captures are within a couple of metres of the snow pose. "Nearest summer capture" is essentially the same place, on a different day, in a different season.

2. **Match.** DISK runs on the snow frame and on each of the three summer priors. LightGlue produces descriptor-matched correspondence pairs; USAC-MAGSAC fits a homography per (snow, prior) pair, restricted to keypoints in the lower portion of the image so the geometry is biased to the ground plane rather than to building façades. The matcher anchors on what the season has not changed: gate posts, fence wires, distant roof edges, masonry corners.

3. **Segment the prior.** Mask2Former, frozen on Cityscapes, produces a road mask on each of the three summer priors. Mask2Former has never been shown a snowy frame; it doesn't have to be, because it never runs on one — it only ever sees the clear prior. Each prior's road mask is reduced to its single largest connected component (a plough cares about the one drivable surface in front of it, not the long tail of parking-lot fragments).

4. **Warp the masks back.** Each prior's road mask is warped via `H⁻¹` into the snow image's pixel space. The pipeline also tracks the warped extent of each prior's full image so the next step can edge-erode where each prior actually covered, rather than dragging mask boundaries across the visible-region edge.

5. **Fuse and crop.** The three warped masks combine via inlier-weighted soft-average; the result is foreground-cropped (the upper portion of a roof-mounted forward camera is sky, and we don't claim those pixels as drivable). What survives is a single road region in the lower image — the road, in the right place, on a frame where there is no road visible.

6. **Smooth over time.** An EMA on the binary mask blends this frame's raw output into the previous smoothed mask. On a frame whose matcher fails entirely, the smoothed mask is held — graceful degradation rather than a flicker to nothing.

The resulting road mask is alpha-blended onto the snow frame in green and emitted to the output stream. The cached `FrameResult` makes downstream renders that change only the smoother or the layout near-instant — the matching pass runs once per track, then is reused across visualisations.

The composition is invariant. The same six steps run on every frame; what changes is which features the matcher anchored on, which prior won, how much of the road is visible. The dataflow does not.

The video extension wraps that static pipeline in three thin layers:

- A **track loader** indexing a snow stream and a paired summer stream by GPS pose.
- A **prior pool** that, for each snow frame, picks the K = 3 nearest summer captures by UTM distance and caches their Mask2Former mask once.
- An **EMA temporal smoother** ($\alpha = 0.4$) running over the binary mask, suppressing per-frame jitter without introducing the drift we saw with optical-flow propagation.

A pickled cache layer makes the matching pass reusable: subsequent renders that change only the smoother (`temporal=ema|flow|none`) skip the ~50-minute matching pass entirely.

Snow imagery in the canonical video demo comes from Boreas (Burnett et al., IJRR 2023), CC BY 4.0 on the AWS Open Data registry — same FLIR Blackfly S camera, same Glen Shields loop, with paired summer traversals on the same UTM coordinates. The static-stills precursor uses Mapillary as an example of an open contributor-imagery substrate; in production the choice is open — Google Street View, Bing Streetside, an operator's own clear-weather captures, or any geo-tagged source. The principle is the geometric correspondence, not the substrate.

Every learned component is frozen. Snow appears only at inference, as the runtime input.

## What we showed

A 15-second clip from a snow-buried Toronto residential street (Boreas `boreas-2021-01-26-11-22`, January 2021) with a continuous green road overlay tracking the buried road frame by frame. Side by side with the naive baseline — the same Cityscapes segmenter applied directly to the snow frame, painting a confident red mask over the entire scene — the contrast is the demonstration. The naive method's red drifts. The cross-season pipeline's green stays on the road.

We render the same clip in five layouts (single overlay, snow-vs-overlay sidebyside, two 3-panel orderings with the naive baseline, and a 2 × 2 quad with the summer prior visible) so the audience can see the same evidence at different depths. The single overlay is the headline; the quad is the receipts.

The static-stills precursor is preserved in the same repository (`make stills`) and produces twenty-seven snow + clear-season hero pairs across Northern Sweden and Finland — the v1.x demo's foundation, twenty-seven different physical locations on which the video extension was built.

The most interesting honest finding survives from the static work and into the moving demo: **inlier count is not a reliable predictor of overlay quality**. A pair with many correspondences can warp the mask onto the wrong region if those correspondences concentrate on building façades. A pair with very few correspondences can be perfect if those few happen to land on the road. The system therefore needs a human in the loop on the input and the output, even after the matcher succeeds.

## The contribution

The contribution is not a snow plough. It is a primitive: **the constants-bridge.**

A constants-bridge is a composition that takes a model trained on regime A, an inference target in regime B, and a known invariant between A and B, and uses the invariant to transfer the model into B without retraining. The invariant in this work is geometric — the road sits where it sat last July — but the shape is more general. Anything that stays constant between two regimes is a candidate bridge: anatomy across imaging modalities, terrain across illumination conditions, scene geometry across weather. The primitive is substrate-agnostic, regime-agnostic, modality-agnostic.

The snow plough is one consumer of this primitive. The road-overlay channel we built is what the constants-bridge *looks like* when consumed for snow-buried road perception; it is not the whole perception stack. The output answers *where the road should be*, not *where to drive*. The most obvious criticism — a snow-covered car parked on the road still sits inside the green overlay — would be fair if the claim were "this replaces a perception stack". It isn't. The plough's perception stack remains a multi-channel system; the constants-bridge contributes one channel and frees the other channels (lidar, depth, obstacle detection) from having to also solve the road-position problem on a buried road. The contribution we are demonstrating is the **primitive**, not its first application.

## What we tried that didn't work

The full audit of approaches and datasets we surveyed and rejected, with the rule each rejection codified, lives in `docs/audit_log.md`. Two illustrative rejections worth surfacing here for what they reveal:

We tried using **previous snow frames as additional priors** for the current frame ("synthetic priors"). Snow→snow matching has identical lighting / lens / viewpoint conditions between consecutive frames, so DISK + LightGlue returned roughly three times more correspondences per pair than snow→summer. The single-frame masks were visibly broader and more confident. In motion the failure was a positive feedback loop: each frame's slightly-too-large mask seeded the next frame's synthetic prior, which produced a slightly-larger mask, and the road overlay drifted outward into bushes and treelines over five to ten seconds. We rejected synthetic priors. **Optical-flow propagation** between matched keyframes failed the same way for a different reason — vanishing-point flow stretches the mask outward — and was rejected for the same kind of evidence: looks plausible per-frame, fails in motion.

We kept **EMA on the binary mask**, α = 0.4. It is the simplest possible smoother. It does the least damage: drops jitter without drifting, and on a frame where matching fails entirely it holds the previous good mask rather than flickering empty. Counterintuitively, the simpler tool is the right tool here, *because* it does not recursively feed back into the matcher.

The pattern — counterintuitive failures of "obvious improvements" that look better in stills — is itself an artefact of the demo. Static frames hide motion artefacts. The discipline that emerged is *never declare a video result a win from sampled stills.*

A separate lesson came from the alt-track work. We initially picked a 350-frame window from a different snow drive of the same Toronto loop (`boreas_2024_12_23`) without verifying the snow trajectory's GPS coordinates against the summer trajectory's. Roughly 75% of the windowed frames lay outside the summer's coverage — the matcher could not have succeeded on those frames regardless of how good the matching was, because there were no priors to match against. We burned roughly 90 minutes of compute on a window that was structurally broken. We added a pre-flight oracle (`src/video_runtime/window_oracle.py`) that verifies prior availability + summer-segmentation quality before any cache build is committed. **The rule: never start a cache build without `make oracle TRACK=<id>` passing first.**

The same rule caught a more subtle near-miss. The Mapillary recon agent surfaced a 400-frame Tromsø winter sequence; the oracle gave it a green light. On closer inspection of the actual frames, the snow imagery turned out to be shot *out the side of a moving bus.* The numerical checks (inlier counts, segmentation quality) all passed because the matcher's job was satisfied per-frame; the *use case* was wrong. The pipeline cannot answer "where is the road in front of a forward-facing camera" from a sideways-facing bus camera. We archived the candidate. **The amended rule: never commit cache compute without auditing the actual frames.** Numbers passing oracle checks is necessary but not sufficient.

The full list of these rejections — including the algorithmic experiments above, the alt-track windowing failure, the Tromsø bus, the broader Mapillary recon's structural null result, and the access-friction rejections of ACDC and MUSES — sits in `docs/audit_log.md`.

## What we did not do

The minimal-shot guarantee is only honest if it is concretely audited. Here is the explicit list:

- We did **not train** any component. DISK, LightGlue, and Mask2Former are pretrained on MegaDepth and Cityscapes respectively; we used them frozen.
- We did **not fine-tune** any component on snow imagery, on a snow corpus, on synthetic snow, or on adverse-weather augmentation.
- We did **not collect** a snow corpus. The snowy Boreas frames enter the pipeline only as runtime inputs.
- We did **not write** a single line of snow-aware logic — no snow detector, no lighting normaliser, no temperature-conditional branching. Snow is a regime the system has never been told about.
- We did **not** quote percentages or IoU scores. The project has no labelled ground truth (Boreas has cm-accurate poses but no pixel-level road labels on snowy frames). Quoted percentages without ground truth are cherry-picked by definition. The qualitative claim — *the road overlay tracks the buried road continuously* — is the honest claim.
- We did **not** beat a benchmark. The brief explicitly rejects leaderboard metrics; the contribution is structural, not numerical.
- We did **not** engage the WOD-E2E dataset the brief mentions. The brief lists it as one possible deliverable, not a requirement; we judged our 3-day time better spent on architectural framing + the analysis notebook + the live demo than on running on a third dataset.
- We did **not** chase ACDC or MUSES through their benchmark-portal access flow — the access friction is incompatible with the deadline. Both are documented in `docs/external_datasets.md` as candidates for a longer cycle.

The contribution is the **composition**, not any single component. DISK, LightGlue, USAC-MAGSAC, Mask2Former are all off-the-shelf; the move is using them together to bridge a regime where one of them would otherwise fail. The video extension adds GPS-keyed prior selection, EMA temporal smoothing, a cache. None of those is individually novel. The composition is.

## Generalising

The structure of the constants-bridge is: a model trained on regime A; an inference target in regime B; a known invariant between the two; transfer through the invariant. Snow on a road is one instance. The shape repeats across the long tail of underdata regimes that minimal-shot autonomy has to face.

**Polar Earth observation.** Climate scientists, ice-sheet researchers, and shipping-route planners want to apply the rich library of land-cover classifiers and feature detectors trained on temperate satellite imagery to the polar regions, where labelled imagery is sparse and seasonally extreme. The invariant is the orbital geometry: the same satellite passes over the same coordinates on a known cadence; for any polar pixel today, an analyst can look up where on Earth it is and what an unconfounded model would expect to find there if the lighting and surface conditions matched a regime where labels exist. Annotating the long tail of polar conditions exhaustively is uneconomic for the same reason annotating snowy roads is — but if the orbital correspondence carries the rich-regime model into the polar regime, the unlabelled gap closes without new labels.

**Low-light medical imaging.** Endoscopists, ophthalmologists, and any modality where a patient's anatomy is the same body across imaging conditions face the same problem in miniature: their well-trained classifiers fail when the imaging itself drifts (low light, novel scope, unusual contrast agent). The invariant is the patient's anatomy across imaging conditions — the same vessel runs in the same place; the same landmark is the same landmark — and a previous well-lit acquisition (or even just an anatomical atlas) can supply the constant. Labelling each new low-light condition is uneconomic given the long tail of scopes, sensors, and conditions; the constants-bridge moves the model across the imaging gap by registering to the known anatomy.

**Off-Earth manipulation.** A manipulator on Mars (or in any operating environment with no in-distribution training data — the deep ocean, a nuclear-decommissioning site, a wildfire zone) cannot be trained on the operating environment because there is no operating environment data to train on. The invariant is the rigid-body geometry of the task and tools: a wrench is a wrench is a wrench; a known robot pose constrains where the wrench has to be. Labelling Mars is impossible. The constants-bridge transfers Earth-trained perception across the geometry the robot already knows it has.

The pattern is the same in each. There is a regime where data is rich. There is a regime where data is sparse, sometimes structurally so. And there is *something* — a geometric correspondence, an orbital schedule, an anatomical atlas, a robot kinematic chain — that connects the two without needing data from both sides. **Find what stays the same. Walk across.**

---

*Code, video clips, and the static-stills precursor: see the [project repository](../README.md). Reproducible from a clean clone via `make reproduce` (canonical 15-second clip), `make oracle TRACK=<id>` (pre-flight before any new track), `make reproduce-track TRACK=<id>` (oracle-verified alt window), `make stills` (static-prior 27-pair Mapillary precursor), or `make demo SNOW=<jpg> PRIOR=<jpg>` (live interactive entry on any user-provided pair).*

*Companion documents in this directory: `analysis.ipynb` (the work, with the work shown — qualitative walkthrough, code, rejected experiments), `audit_log.md` (every approach + dataset surveyed + rejected with reasons), `submission_video_plan.md` (90–120s shot list with narration + asset paths), `external_datasets.md` (acquisition guide for non-Boreas snow datasets). Boreas dataset (UTIAS-ASRL) under CC BY 4.0. Submitted to SoTA Commission I — Minimal-Shot Autonomy, May 2026.*
