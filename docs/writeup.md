---
title: "Constants as the bridge"
subtitle: "Minimal-shot autonomy via geometric correspondence"
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

## The concept

An autonomous system trained on regime A often has to operate in a
regime B for which little or no labelled data exists. The default
response is to collect more data and retrain. This is reasonable when
B is small or stable; it scales poorly when B is one of an open-ended
set of conditions a deployed system encounters in the field — weather,
season, lighting, terrain, sensor degradation, novel sites.

There is a second option that does not require new labels.
If some property *X* is invariant between A and B and is observable
in both, then a model trained on A can be applied to data acquired
in A and the result transferred to B through *X*. We refer to this
composition as a *constants-bridge*: a model on A, an inference
target in B, and an invariant linking the two.

The components are not new. Geometric registration, classical CV
matching, and frozen pretrained models have been used in many
combinations. The contribution here is to identify the composition
itself as a primitive — useful whenever the cost of labelling B is
high and some invariant connects B to a regime where labels are
abundant — and to give a worked instantiation that runs end-to-end.

## A worked instance: snow-buried roads

Most automotive perception models are trained on dry roads.
A snow plough operates outside that distribution: lane markings,
curbs, and the road–verge boundary are buried, and an
off-the-shelf segmenter applied directly to the snow image labels
the entire scene with low confidence or with confident error.
Collecting and annotating a snow-corpus that covers the long tail
of conditions is expensive and chronically incomplete.

The road, however, has not moved. For most roads in mapped areas,
clear-weather imagery exists at the same coordinates: prior dataset
captures, contributor uploads, the operator's own previous traversals.
The plough's missing information is not what a road looks like in
snow, but where the road is in its current camera frame. That is a
registration problem. The invariant *X* is the road's location, and
it is observable in both the snowy frame (live) and the clear-season
frame (archival).

## The pipeline

A frozen feature matcher establishes correspondences between the
live snow frame and a clear-season prior of the same coordinates.
A homography fitted to those correspondences, biased toward the
ground plane, registers the prior to the snow frame. A frozen
Cityscapes-trained segmenter produces a road mask on the *clear*
prior — never on the snow frame — and the mask is warped through
the homography into the snow image. The inputs are the live frame
and a geo-tagged clear-season image; the output is a road mask in
the live frame's coordinates.

```
   ┌──────────────┐                    ┌──────────────────────┐
   │  Snow frame  │                    │  Clear-prior frame   │  any geo-tagged
   │   (live)     │                    │  (Boreas summer,     │  clear-weather
   │              │                    │   Mapillary, GSV…)   │  imagery substrate
   └──────┬───────┘                    └──────────┬───────────┘
          │                                       │
          │                                       ▼
          │                            ┌──────────────────────┐
          │                            │   Mask2Former        │  frozen
          │                            │  (Cityscapes road)   │  Cityscapes
          │                            └──────────┬───────────┘
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

Components used unmodified:
DISK (Tyszkiewicz et al., NeurIPS 2020) for keypoint extraction,
LightGlue (Lindenberger et al., ICCV 2023) for descriptor matching,
USAC-MAGSAC (Barath et al., CVPR 2020) for robust homography, and
Mask2Former (Cheng et al., CVPR 2022) trained on Cityscapes
(Cordts et al., CVPR 2016) for road segmentation. None is fine-tuned
on snow.

The video extension adds three thin layers around this static core:
a track loader indexing snow and summer streams by GPS pose; a prior
pool that returns the K = 3 nearest summer captures by UTM distance,
caching their road mask; an EMA on the binary mask (α = 0.4) for
temporal smoothing. A pickled cache stores per-frame matching
results so subsequent renders that change only the smoother or
layout do not re-run the matching pass.

## Results

The headline artefact is a 15-second video clip from a snow-covered
residential street in Toronto (Boreas `boreas-2021-01-26-11-22`,
January 2021). The pipeline produces a continuous road-region
overlay tracking the buried road frame by frame. A side-by-side
naive baseline — the same Cityscapes segmenter applied directly to
the snow frame — is included for contrast: the naive overlay is
spatially incoherent across frames; the cross-season pipeline's
overlay tracks the road.

The same clip is rendered in five layouts (single overlay,
snow-vs-overlay side-by-side, two three-panel orderings with the
naive baseline, and a 2 × 2 quad with the summer prior visible) so
the same evidence can be inspected at different depths. A
twenty-seven pair static-prior precursor (`make stills`) covers
distinct snow scenes across northern Sweden and Finland and
predates the video extension.

Without pixel-level snowy-road ground truth, IoU and coverage
percentages would be cherry-picked. We therefore report
qualitatively: the road overlay tracks the buried road continuously
through the canonical clip on a pipeline whose learned components
have never seen snow.

## What did not work

Two seemingly-natural extensions failed in motion in ways static
inspection did not predict.

*Synthetic priors from previous snow frames.* Snow-to-snow matching
returns roughly three times as many correspondences as snow-to-summer.
In single-frame stills the resulting masks were broader and more
confident. In motion the slightly-too-large mask seeded the next
frame's synthetic prior, producing a slightly-larger mask, and the
overlay drifted outward into vegetation over five to ten seconds —
a positive feedback loop in which higher matcher confidence reflected
matching to an increasingly wrong reference. Rejected.

*Optical-flow propagation* between sparsely-matched keyframes
exhibited the same shape: vanishing-point flow on a forward-driving
camera pushes mask boundaries outward at every step, and the road
region grew until it engulfed sidewalks. Rejected.

We retain a binary-mask EMA at α = 0.4. It is the simplest available
smoother and does not feed its output back into the matcher; on
frames where matching fails entirely it holds the previous mask
rather than flickering empty. The general lesson — *do not call a
video result a win from sampled stills* — applies whenever the
quality measure is computed per-frame.

A second finding from the static work persists into the moving demo:
the per-frame inlier count is not a reliable predictor of overlay
quality. A pair with many correspondences can warp the road mask
onto the wrong region if those correspondences cluster on building
façades; a pair with few correspondences can produce a clean fit if
the few happen to lie on the road. The pipeline therefore benefits
from human review on input and on output even after the matcher
succeeds.

## Boundaries

The contribution is the composition, not any single component. DISK,
LightGlue, USAC-MAGSAC, and Mask2Former are off-the-shelf and used
frozen. The video extension's prior pool, EMA smoothing, and cache
are not individually novel. No component is trained or fine-tuned
on snow.

The output answers *where the road should be*, not *where to drive*.
A snow-covered car parked on the road still falls inside the green
overlay; the overlay is a road-position channel, not an obstacle map.
A complete plough perception stack would combine this channel with
lidar, depth, and obstacle detection; the contribution here removes
the road-position problem from the other channels' workload, not the
other channels.

The substrate of the clear-season prior is interchangeable. We use
Boreas's matched summer captures for the canonical demo (cm-accurate
Applanix poses, same FLIR Blackfly S camera) and Mapillary for the
static-stills precursor (open contributor imagery). Google Street
View, Bing Streetside, and operator-owned clear-weather captures all
satisfy the same interface. The principle is the geometric
correspondence; the substrate is a deployment choice.

## Generalising

The constants-bridge structure repeats wherever (i) data exists for
some regime A and is scarce in a related regime B, and (ii) some
observable invariant connects A to B without requiring data from
both sides.

*Polar Earth observation.* Land-cover classifiers and feature
detectors are well-trained on temperate satellite imagery; labelled
polar coverage is sparse and seasonally extreme. The orbital
geometry is the invariant: the same satellite passes over the same
coordinates on a known cadence, so an analyst can register today's
polar pixel against the well-labelled regime where conditions
matched.

*Low-light or novel-modality medical imaging.* Classifiers trained
on standard-acquisition imagery degrade when the acquisition itself
drifts. The patient's anatomy is the invariant: the same vessel and
landmark structure is present across imaging conditions, and a
prior well-lit acquisition or anatomical atlas registers a new
acquisition into the regime where the model was trained.

The shape is consistent across instances: a rich-data regime, a
data-sparse regime, and an invariant — geometric, anatomical,
orbital, kinematic — that connects them. The cost of labelling
the long tail is replaced with the cost of identifying and
registering against the appropriate constant.

---

*Code, video clips, and the static-stills precursor: see the
[project repository](../README.md). Reproducible from a clean clone
via `make reproduce` (canonical 15-second clip),
`make reproduce-track TRACK=<id>` (any registered track), or
`make stills` (27-pair static-prior precursor).
Companion notebook at `analysis.ipynb` walks through the principle,
the worked example, and the rejected experiments in code. Boreas
dataset (UTIAS-ASRL) under CC BY 4.0. Submitted to SoTA Commission I
— Minimal-Shot Autonomy, May 2026.*
