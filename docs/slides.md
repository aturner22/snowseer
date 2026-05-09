---
marp: true
theme: snow-underlay
paginate: true
header: "Snowseer  ·  SoTA Commission I  ·  Minimal-Shot Autonomy"
footer: "Constants as the bridge"
---

<!-- _class: title -->
<!-- _paginate: false -->

# Snowseer

## Achieving minimal-shot autonomy by recognising constants across environments.

###### SoTA Commission I  ·  May 2026

---

## Minimal-shot autonomy

Minimal-shot autonomy concerns how a system survives in **unfamiliar environments.**

The commonly accepted response is to collect more data and retrain.

It does not scale across the long tail of conditions a deployed vehicle, robot, or drone meets in the real world. Snow, dust, smoke, washouts and variable human infrastructure are obvious examples.

###### A perception system that depends on having been trained on each new condition will lag every condition it has not yet encountered.

---

<!-- _class: image-right -->

![bg right:55%](assets/media/toronto_2021_snow_naive_overlay_t005.jpg)

## A snow plough's job is simple

**Sweep the road clear.**

The catch is that while the plough is doing it, the road is necessarily invisible. Curbs are buried, lane markings are gone, the boundary between tarmac and verge is no longer defined.

A self-driving stack trained on traditional road conditions, applied directly to the plough's camera, will report with calibrated confidence that the entire scene is road and should be cleared.

---

## A second move

For almost every operating environment where autonomy fails for lack of data, an adjacent regime exists, temporally or seasonally or geographically, where data is plentiful and rich, and whose key components remain the same across environments.

The road that needs to be ploughed this winter is the same road it was in the summer.

Its appearance has changed. Its position in space and relative to local landmarks has not.

###### Snowseer is one demonstration of leveraging structural constants across regimes to achieve minimal-shot autonomy.

---

<!-- _class: principle -->

# The constants-bridge

A composition that takes a model trained on regime A, an inference target in regime B, and a known invariant linking the two, and uses the invariant to transfer the model into regime B **without retraining.**

###### The invariant in this work is geometric. The road sits where it sat last summer, in the same place relative to other landmarks. The shape is general: anatomy across imaging conditions, terrain across illumination, scene structure across weather.

---

## Constants in this work, made visible

![](assets/media/nordic_gallivare_matches.png)

###### Snow query (left) and the paired summer prior (right). Green correspondences mark the features that survive the season: gateposts, fence wires, masonry corners, distant roof edges. None land on the road surface itself. The homography fitted to those carries the prior's road mask into the snow frame.

---

## How the system works (per snow frame)

1. Pull the **live snowy frame** from the plough's camera.
2. Pull a **clear-season prior** of approximately the same coordinates.
3. **Match** the two with DISK + LightGlue.
4. Estimate a **homography** via USAC-MAGSAC RANSAC.
5. Run a Mask2Former segmenter on the **clear prior only.**
6. **Warp** the road mask into the snow frame and overlay.

The plough now knows where the road is. No model in the pipeline has been trained on snow.

---

## Components

| Component | Role | Model | Dataset |
|---|---|---|---|
| Feature detector | Locate keypoints in each image | **DISK** *(NeurIPS '20)* | MegaDepth |
| Feature matcher | Pair keypoints across snow / summer | **LightGlue** *(ICCV '23)* | MegaDepth |
| Homography fit | Robust geometric registration | **USAC-MAGSAC** *(CVPR '20)* | n/a |
| Road segmenter | Road mask on the summer image | **Mask2Former** *(CVPR '22)* | Cityscapes |
| Driving captures | Paired snow + summer Toronto traversals | n/a | **Boreas** *(IJRR '23, CC BY 4.0)* |

Every learned component is **frozen.** Snow appears only at inference, as the runtime input.

---

## The dataflow

```
   Snow frame                Clear-prior frame
        │                            │
        │                            ▼
        │                   Mask2Former  (frozen)
        │                            │ road mask in prior space
        │                            │
        └──►  DISK + LightGlue  ◄────┘   (frozen)
                     │
                     ▼ correspondences
            USAC-MAGSAC homography  (ground-plane biased)
                     │
                     ▼ H
            warp prior mask → snow space
                     │
                     ▼
            fuse over K=3 priors  +  EMA over time
                     │
                     ▼
            Road overlay on the snow frame
```

###### A track loader, a prior pool of K=3 nearest summer captures by UTM, and an EMA on the binary mask (α = 0.4) wrap the per-still pipeline for the video case.

---

## One frame, end-to-end

![](assets/media/toronto_2021_quad_t005.jpg)

###### Top-left: snow input. Top-right: naive Cityscapes baseline applied directly to snow, painting the road class across snow and sky. Bottom-left: paired summer prior with successful road segmentation. Bottom-right: cross-season overlay produced by warping the prior's road mask through the homography into the snow frame.

---

## Demo: Toronto, January 2021

![](assets/media/toronto_2021_quad_t005.jpg)

###### 14 s drive of a snow-buried residential street. After importing this deck, replace the still above with the demo clip via Insert → Video → Drive (`toronto_2021_quad.mp4`).

---

## Demo: Toronto, February 2025

![](assets/media/toronto_2021_quad_t005.jpg)

###### 34 s drive in active snowfall, late afternoon. Same code, different snow, different drive. After importing, replace the still with `toronto_2025_quad.mp4` via Insert → Video → Drive.

---

## Nordic stills precursor

The same pipeline verified across 18 hand-picked nordic snow / summer pairs from Mapillary. Three representative examples (cross-season overlay in green): Gällivare, Kiruna, Luleå.

![](assets/media/nordic_3up.jpg)

###### Distinct snow scenes, road layouts, lighting and environments, all on a pipeline whose components have never seen snow.

---

## Limitations

**Some artefacts in the overlay are inherited from the summer prior.** Where the front of the summer capture vehicle is visible, the warped road mask begins a short distance ahead of the snow camera rather than directly under it. Where a parked car or other obstacle sits on the road in the prior, the segmenter routes the road class around it and the overlay carries the cutout forward.

**The pipeline is not, currently, real-time.** The matching pass dominates per-frame compute, taking around 16 s per frame on Mac CPU. Demo clips build end-to-end in roughly an hour. Real-time operation needs a substantially faster matcher and segmenter.

**The system is not, currently, deployable arbitrarily.** The current code is geared toward the specific Toronto and nordic demo material. Generalising to any road with Google Street View or a comparable source available is feasible (the pipeline is substrate-agnostic in principle), but is a future integration step.

---

## Next steps

1. **Real-time matcher.** Bring per-frame matching from around 16 s on Mac CPU to under 1 s on a deployment-class device. Required for live operation.
2. **Visual place recognition front-end.** Replace GPS-pose lookup with a learned recognition step so the appliance works in GPS-denied environments and without prior pose.
3. **Multi-source clear-season image bank.** Integrate Mapillary global, Street View, and operator captures so any covered road can be a deployment target.
4. **Hardware prototype.** A battery-powered processing unit running the live appliance with a simple HUD-style output.

###### The snow plough's road-position channel is one consumer of this appliance. The same recipe could power fog, dust, smoke, heavy rain and night driving, and many more obscured-regime use cases.

---

## Reproduce

```
git clone https://github.com/aturner22/snowseer
cd snowseer
uv sync --python 3.12
make reproduce
```

`make track TRACK=<id>`: full pipeline on any registered track.
`make stills`: static-stills precursor (needs `MAPILLARY_TOKEN`).
`make slides`: build `docs/slides.pptx` for Google Slides.

Site: [aturner22.github.io/snowseer](https://aturner22.github.io/snowseer/).

---

<!-- _class: title -->
<!-- _paginate: false -->

# Constants as the bridge

## Find what stays the same. Walk across.

###### Snowseer  ·  SoTA Commission I  ·  May 2026
