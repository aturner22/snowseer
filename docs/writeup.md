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

Reading the diagram in the abstract is not the same as seeing it operate on one frame, with numbers. We pick frame 137 of the canonical 15 s clip (Boreas snow stream index 237, video time 13.7 s) and trace the pipeline.

The snow frame is 1024 × 857 px, taken at UTM coordinates (-195.55, 126.41) in Boreas's per-sequence local frame. Glen Shields, residential, mid-morning, heavy snow on the pavement.

1. **Prior selection.** The summer prior pool's KD-tree on summer poses returns the three nearest by UTM distance: 0.49 m, 0.68 m, 1.50 m. All three are within two metres of the snow pose — the canonical loop's summer trajectory is sampled densely enough that "nearest summer capture" is sub-metre accurate at this frame.

2. **Match.** DISK runs on the snow frame and on each of the three summer priors, extracting up to 2048 keypoints per image. LightGlue matches them; USAC-MAGSAC fits a homography per (snow, prior) pair, restricted to keypoints in the lower 70 % of the image so the geometry is biased to the ground plane rather than to building façades. After RANSAC the inlier counts are 43, 38, 41 — strong correspondence on this frame.

3. **Segment the prior.** Mask2Former, frozen on Cityscapes, produces a road mask on each of the three summer priors. Mask2Former has never been shown a snowy frame; it doesn't have to be, because it never sees one — it only ever sees the clear prior. Each prior's road mask is reduced to its single largest connected component (a plough cares about the one drivable surface, not parking-lot fragments).

4. **Warp the masks back.** Each prior's road mask is warped via `H⁻¹` into the snow image's pixel space. The pipeline also tracks the warped extent of each prior's full image so the next step can edge-erode where each prior actually covered, rather than dragging mask boundaries across the visible-region edge.

5. **Fuse and crop.** The three warped masks combine via inlier-weighted soft-average; the result is foreground-cropped at y = 0.30 H (the upper 30 % of a roof-mounted forward camera is sky, and we don't claim those pixels as drivable). On this frame the fused mask covers 24.5 % of the foreground — the road occupies roughly a quarter of the lower 70 % of the image.

6. **Smooth over time.** EMA with α = 0.4 blends this frame's raw mask into the previous smoothed mask. On a frame whose matcher fails entirely, the smoothed mask is held — graceful degradation rather than a flicker to nothing.

The resulting road mask is alpha-blended onto the snow frame in green and emitted to the output stream. Frame 137 took 18 s of CPU time; matching dominates. The cached `FrameResult` makes downstream renders that change only the smoother or the layout instant — the matching pass is run once per (track, window).

That is the whole pipeline, on one frame, with real numbers. The dataflow doesn't change between this frame and any other on this track; the inlier counts shift, the prior distances shift, the fused coverage shifts. The composition is invariant.

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

The most interesting honest finding survives from the static work and into the moving demo: **inlier count is not a reliable predictor of overlay quality**. A pair with hundreds of inliers can warp the mask onto the wrong region if the inliers concentrate on building façades. A pair with seventeen inliers can be perfect if those seventeen happen to land on the road. The system therefore needs a human in the loop on the input and the output, even after the matcher succeeds.

## Scope of the contribution

The system answers *where the road should be*. It does not answer *where to drive*. That distinction matters, because the most obvious criticism of a road-overlay pipeline is the obstacle case: a snow-covered car parked on the road would still sit inside the green overlay; the pipeline knows nothing about obstacles, drivable surface, or 3D geometry. That criticism would be fair if the claim were "this replaces a perception stack". It is not the claim. The claim is that this is *one channel* — a 2D road-position prior — to feed alongside lidar, depth estimation, and obstacle detection in a fuller stack. The contribution is the **move**: how to extend a model from a data-rich regime into an adjacent data-poor one, by anchoring on what stays constant between the two. Snow on a road is the instance we built; the structure transfers.

## What we extended (and what we tried that didn't work)

The video pipeline is a thin wrapper around the static pipeline, but two extensions deserve note for what they reveal.

We tried using **previous snow frames as additional priors** for the current frame ("synthetic priors"). Snow→snow matching is much higher confidence than snow→summer because lighting, lens, and viewpoint conditions are identical between consecutive snow frames; the matcher returned three to seven times more inliers per pair. In single-frame stills, the resulting mask was visibly broader and more confident. In motion, the mask drifted outward over time: each frame's slightly-too-large mask seeded the next frame's synthetic prior, which produced a slightly-larger mask, and the road mask grew into bushes and treelines over a few seconds. The failure was a positive feedback loop the static stills hid. We rejected synthetic priors.

We tried **optical-flow propagation** between matched keyframes. Same outcome, different mechanism: forward-driving cameras have vanishing-point flow that stretches the previous mask outward at every step. We rejected it.

We kept **EMA on the binary mask**, $\alpha = 0.4$. It is the simplest possible smoother and it does the least damage: it drops jitter without drifting, and on a frame where matching fails entirely, it holds the previous good mask rather than flickering empty.

This pattern — counterintuitive failures of "obvious" improvements that look better in stills — is itself an artefact of the demo. Static frames hide motion artefacts. We learned to verify motion before claiming wins.

A second lesson came out of the alt-track work. We initially picked a different snow drive of the same Toronto loop (Boreas `boreas_2025_02_15`, active snowfall, late afternoon) and tried to demo it on the same window-selection logic as the canonical. Most of the early frames returned no usable priors — not because the matcher was bad, but because the snow trajectory's UTM coordinates lay outside the summer trajectory's coverage on those frames. The pipeline can only succeed where (a) a prior exists *and* (b) the prior's road segmentation is non-degenerate. We added a small pre-flight check (`window_oracle.py`) that verifies both conditions before any cache build is committed. **The discipline that emerged is: never demonstrate the pipeline on data that has no chance of matching, and never trust a "candidate window" before its priors have been segmented and inspected.**

## What we didn't

We didn't train anything. We didn't fine-tune. We didn't collect a snow corpus. We didn't write a single line of snow-aware logic. The only handle we offered the model on the snow regime was the clear prior of the same place, plus the generic robustness of pretrained matchers and segmenters that have never seen snow.

We didn't claim the system replaces lidar or 3D depth estimation. The output is a 2D road *prior*, not a drivable-surface estimate. We didn't claim the homography is exact — it isn't, the world isn't planar — only that it transfers the road mask approximately and the approximation is enough.

We didn't claim real-time. The current matcher runs at ~5 s per snow→summer pair on Mac CPU; the canonical 15-second clip's matching cache builds in ~50 minutes. The cache layer makes that cost amortise across renders, but real-time would require a substantially faster correspondence model — that would be a learned component, and trained on synthetic snow to preserve the no-snow-in-training guarantee.

We didn't claim novelty in any single component. The novelty, such as it is, is in the *composition*: the matcher, the segmenter, and the homography are off-the-shelf; the move is using them together to bridge a regime where one of them would otherwise fail. The video extension adds GPS-keyed prior selection, EMA temporal smoothing, and a cache. None individually novel. The composition is.

## Generalising

The structure of the move is: a model trained on regime A; an inference target in regime B; a known correspondence between the two; transfer through the correspondence. Snow on a road is one instance. Others admit the same structure: low-light medical imaging without low-light training data, polar earth observation without polar training data, a manipulator on Mars without Mars training data. In each case there is a regime in which we have plenty of data, an adjacent regime in which we don't, and a constant between the two — temporal, geometric, anatomical — that lets us bridge.

We are not going to label our way out of every long-tail regime. But for many of them, we don't have to. We just have to find what stays the same and walk across.

---

*Code, video clips, and the static-stills precursor: see the [project repository](README.md). Reproducible from a clean clone via `make reproduce` (canonical 15-second clip), `make oracle TRACK=<id>` (pre-flight before any new track), `make reproduce-track TRACK=<id>` (oracle-verified alt window), or `make stills` (static-prior 14-pair Mapillary precursor). Boreas dataset (UTIAS-ASRL) under CC BY 4.0. Submission video composed externally; storyboard at `docs/slides.md`. Submitted to SoTA Commission I — Minimal-Shot Autonomy, May 2026.*
