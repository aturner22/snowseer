---
title: "snowseer"
subtitle: "Constants as the bridge — minimal-shot autonomy"
author: "SoTA Commission I — May 2026"
geometry: margin=2cm
mainfont: "EB Garamond"
sansfont: "Inter"
monofont: "JetBrains Mono"
fontsize: 11pt
linestretch: 1.35
colorlinks: true
linkcolor: "gray"
---

## Minimal-Shot Autonomy

Minimal-shot autonomy is the question of how a perception system survives in regimes it has not been heavily trained on. The default response is to collect more data and retrain. That answer scales when the new regime is small or stable; it does not scale across the long tail of conditions a deployed vehicle, robot, or drone meets when it leaves the regime its training set was sampled from — snow, dust, smoke, washouts, regional construction practices, agricultural off-road, novel industrial sites. A perception system that depends on having been trained on each new condition will lag every condition it has not yet been trained on.

A snow plough's job is short: keep the road clear. The catch is that, while the plough is doing it, the road is invisible. Curbs are buried, lane markings are gone, the seam between asphalt and verge is no longer drawn. A self-driving stack trained on Cityscapes, applied directly to the plough's camera, will report with calibrated confidence that the entire scene is sky. Annotating a labelled snow corpus dense enough to cover the long tail of road / weather / time-of-day combinations is uneconomic and chronically incomplete.

There is a second move available, and it does not require new training data. For almost every operating regime where autonomy fails for lack of data, an adjacent regime exists — temporally, seasonally, geographically — where data is rich, and where the parts that matter to the task are the same. The plough's road is the same road it was last July. Its appearance has changed completely; its position in space has not. snowseer is one demonstration of how to use that distinction.

## The constants-bridge

We propose a primitive: the *constants-bridge*. A constants-bridge is a composition that takes a model trained on regime A, an inference target in regime B, and a known invariant linking A and B, and uses the invariant to transfer the model into B without retraining. The invariant in this work is geometric — the road sits where it sat last summer — but the shape is more general. Anything that stays observable and unchanged between two regimes is a candidate bridge: anatomy across imaging conditions, terrain across illumination, scene structure across weather, orbital geometry across polar darkness.

The constituent parts are not new. Geometric registration, classical RANSAC, and frozen pretrained feature matchers and segmenters have been combined in many ways across the computer-vision literature. The contribution this submission makes is to identify the composition itself as a primitive worth naming, and to give an end-to-end working instantiation that exhibits the property the brief calls out — *generalisation through visual understanding, not memorisation*. The matcher in our pipeline is not generalising "snow"; it has not been trained on snow. It is generalising what stays the same. The architecture of the contribution is to make that generalisation load-bearing.

## How the system works

A frozen feature matcher establishes correspondences between the live snow frame and a clear-season prior of the same coordinates. A homography fitted to those correspondences, biased toward the ground plane, registers the prior to the snow frame. A frozen Cityscapes-trained segmenter produces a road mask on the *clear* prior — never on the snow frame — and the mask is warped through the homography into the snow image's pixel space.

Per snow frame, six steps:

1. Pull the live snowy frame from the plough's camera.
2. Pull a clear-season prior of the same coordinates (Boreas's matched summer drive in the canonical demo; Mapillary contributor uploads in the static-stills precursor).
3. Match the two with DISK + LightGlue.
4. Estimate a homography by USAC-MAGSAC RANSAC, biased to the ground plane.
5. Run Mask2Former (Cityscapes-trained) on the *clear* prior only.
6. Warp the road mask into the snow frame via the homography.

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

Components used unmodified: DISK (Tyszkiewicz et al., NeurIPS 2020) for keypoint extraction, LightGlue (Lindenberger et al., ICCV 2023) for descriptor matching, USAC-MAGSAC (Barath et al., CVPR 2020) for robust homography, Mask2Former (Cheng et al., CVPR 2022) trained on Cityscapes (Cordts et al., CVPR 2016) for road segmentation. None is fine-tuned on snow.

The video extension wraps the per-pair pipeline in three thin layers: a track loader indexing snow and summer streams by GPS pose; a prior pool returning the K = 3 nearest summer captures by UTM distance for each snow frame and caching their road masks; an exponential moving average ($\alpha = 0.4$) on the binary mask. A pickled cache stores per-frame matching results so subsequent renders that change only the smoother or layout do not re-run the matching pass.

## What we built, and what we found out trying it

The headline artefact is a 15-second video clip from a snow-covered residential street in Toronto (Boreas `boreas-2021-01-26-11-22`, January 2021). The pipeline produces a continuous green road-region overlay tracking the buried road frame by frame. A side-by-side naive baseline — the same Cityscapes segmenter applied directly to the snow frame — is included for contrast: the naive overlay is spatially incoherent across frames; the cross-season pipeline's overlay tracks the road. A second canonical clip (`boreas_2025_02_15`) on the same Toronto loop in different snowfall conditions runs the same pipeline with the same parameters, evidence that the principle is robust to the conditions inside the snow regime, not just to one capture. A 27-pair static-stills precursor (`make stills`) covers distinct snow scenes across northern Sweden and Finland and predates the video extension.

Without pixel-level snowy-road ground truth, IoU and coverage percentages would be cherry-picked. The honest claim is qualitative: the road overlay tracks the buried road continuously through the canonical clip on a pipeline whose learned components have never seen snow, and the same pipeline runs unchanged on a different snowfall on the same loop and on twenty-seven static pairs in different countries.

A minimal-shot integrity audit accompanies the codebase: zero snowy frames touch any model weights; zero snowy frames touch any annotation pipeline; snow appears only as runtime input; pretrained matcher, pretrained segmenter, classical RANSAC; reproducible from a clean clone with `make reproduce`. The only handle the system has on the snow regime is the clear-season prior of the same place.

## Known limitations

*Synthetic priors from previous snow frames* return three to seven times more correspondences per pair than snow-to-summer, because lighting, lens, and viewpoint are identical between consecutive snow frames. In single-frame stills the resulting masks looked broader and more confident. In motion, each frame's slightly-too-large mask seeded the next frame's synthetic prior, producing a slightly-larger mask, and the road overlay drifted outward into vegetation over five to ten seconds — a positive-feedback loop in which higher matcher confidence reflected matching to an increasingly wrong reference. We rejected synthetic priors. *Optical-flow propagation between sparsely-matched keyframes* exhibited the same shape: vanishing-point flow on a forward-driving camera pushed mask boundaries outward at every step, and the road region grew until it engulfed sidewalks. We rejected it. The lesson generalised — failure modes that look better in stills and worse in motion are themselves evidence of mechanism, and the discipline that emerged was *do not call a video result a win from sampled stills*.

*The per-frame inlier count is not a reliable predictor of overlay quality.* A pair with many correspondences can warp the road mask onto the wrong region if those correspondences cluster on building façades; a pair with few correspondences can produce a clean fit if the few happen to lie on the road. The pipeline therefore benefits from human review on input and output even after the matcher succeeds.

*The pipeline is not real-time.* The matching pass dominates per-frame compute; the canonical 15-second clip's cache builds in roughly an hour on Mac CPU. The cache layer makes that cost amortise across renders, but real-time would require a substantially faster matcher — this is a deployment-engineering choice, not a research question, and is the first item in the next-steps section.

## Where this could go

The constants-bridge primitive is the foundation for a general image-banking-and-transfer appliance. Input: a live camera feed and a location signal (GPS, IMU, or a learned visual place recognition step). Bank: a large geo-tagged image database — Mapillary global, Street View, the operator's own captures. Process: register the live frame against nearest bank candidates via geometric correspondence. Output: any pre-computed annotation from the bank, transferred into live-frame coordinates. The snowplough's road-position channel is one consumer of this appliance. The same appliance, with the same recipe, powers fog / dust / smoke / heavy-rain / night driving (the bank is the clear-conditions capture of the same place), heads-up display navigation, seeing around corners (the bank is earlier captures of the upcoming intersection from this drive or from V2V partners), and construction-zone delta detection (bank-vs-live = what is new on site).

Five concrete v3 items the prize money would fund:

1. **Real-time matcher** — port DISK + LightGlue (or an alternative such as RoMa or LoFTR) to GPU; bring per-frame matching from ~16 s on Mac CPU to ~30 ms on a Jetson-class device; required for live operation.
2. **Visual place recognition front-end** — replace the GPS-pose lookup with a learned VPR step so the appliance works in GPS-denied environments and without prior pose.
3. **Multi-source bank** — the substrate-pluggable layer (archived after our v3 substrate experiment in `_archive/v3_substrate_experiments/` did not beat operator-paired drives with the current matcher) revived with a dense correspondence model so amateur Mapillary uploads can stand in for operator captures.
4. **Hardware prototype** — a battery-powered Jetson running the live appliance, with a simple HUD output, demonstrating the snow / fog / night-driving consumer scenarios end-to-end.
5. **Two more constants-bridge instances** for proof of generality beyond snow. Strong candidates: agricultural autonomy (a previous-season drone overflight of the field is the bank, the ground vehicle bridges via terrain geometry) and industrial inspection in obscured environments (mining, post-fire, decommissioning — pre-incident facility scan as the bank, in-suit camera bridges through structural geometry).

The next phase is therefore not "improve the snowplough." It is "build the general appliance, of which the snowplough is one demonstration." Real-world deployment of minimal-shot autonomy looks like a small piece of hardware that turns a clear-conditions image bank into a transfer of perception into whatever obscured regime the operator currently faces.

---

*Code, video clips, and the static-stills precursor: see the [project repository](../README.md). Reproducible from a clean clone via `make reproduce` (canonical 15-second clip), `make track TRACK=<id>` (any registered track), or `make stills` (27-pair static-prior precursor). Companion notebook at `analysis.ipynb` walks through the principle, the worked example, and the dead ends in code. Boreas dataset (Burnett et al., IJRR 2023, UTIAS-ASRL) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Mapillary imagery under the Mapillary open-data licence. Models pretrained by their respective authors and used frozen. Submitted to [SoTA Commission I — Minimal-Shot Autonomy](https://sotaletters.substack.com/p/sota-commission-i-minimal-shot-autonomy), May 2026.*
