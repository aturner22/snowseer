---
marp: true
theme: snow-underlay
paginate: true
header: "Snow-Underlay  ·  SoTA Commission I — Minimal-Shot Autonomy"
footer: "Constants as the bridge — in motion"
---

<!-- _class: title -->
<!-- _paginate: false -->

# Constants as the bridge

## Minimal-shot autonomy, in motion

###### Snow-Underlay  ·  SoTA Commission I  ·  May 2026

---

<!-- _class: image-right -->

![bg right:55%](../outputs/video/boreas_2021_01_26/stills/overlay__t001p0.jpg)

## A snow plough's job is short.

Keep the road clear.

The catch:
**while the plough is doing it,
the road is invisible.**

A self-driving stack trained on Cityscapes
will report, with calibrated confidence,
that the entire scene is sky.

---

<!-- _class: image-right -->

![bg right:55%](../outputs/video/boreas_2021_01_26/stills/snow_naive_overlay__t005p0.jpg)

## Off-the-shelf segmentation, applied directly:

a confident red mask
that does not survive a glance.

Or worse — silent failure:
zero road predicted,
no warning issued.

---

<!-- _class: pullquote -->

> Minimal-shot autonomy is the question
> of how a perception system survives
> in regimes it has not been heavily trained on.

###### Labelling cannot keep pace with reality. Every condition the training set missed is a regime where memorisation alone fails.

---

## There is a different move.

For almost every regime where autonomy fails for lack of data,
there is an *adjacent regime* — temporally, seasonally, geographically —
where data exists, and where the parts that matter are the same.

The plough's road is the same road it was last July.

The curb hasn't moved. The hydrant hasn't moved.

The road's *appearance* has changed completely.
The road's **position in space** has not.

---

<!-- _class: principle -->

# Constants as the bridge

If we can identify what stays constant
between the data-rich regime and the data-poor one,
we can extend our existing models into the new regime —
**without learning a single thing about it.**

###### Generalisation, not memorisation.

---

## The recipe (per snow frame)

<div class="recipe">

1. &nbsp;Pull the **live snowy frame** from the plough's camera.
2. &nbsp;Pull a **clear-season prior** of the same coordinates (Boreas paired summer drives).
3. &nbsp;**Match** the two with a frozen feature matcher.
4. &nbsp;Estimate a **homography**, biased toward the ground plane.
5. &nbsp;Run a Cityscapes road segmenter — **on the clear prior only**.
6. &nbsp;**Warp** the road mask onto the snowy frame.

</div>

The plough now knows where the road is.
It has not been trained on snow.

---

## Architecture

| Component | Role | Pretrained on |
|-----------|------|---------------|
| **DISK** &nbsp;*(NeurIPS '20)* | Local features | MegaDepth · no snow |
| **LightGlue** &nbsp;*(ICCV '23)* | Sparse matcher | MegaDepth · no snow |
| **USAC-MAGSAC** &nbsp;*(CVPR '20)* | Robust homography | — |
| **Mask2Former** &nbsp;*(CVPR '22)* | Road segmenter | Cityscapes · no snow |
| **Boreas** &nbsp;*(IJRR '23)* | Snow + paired summer captures | — (CC BY 4.0) |

Every learned component is **frozen**.
Snow appears only at inference, as the runtime input.

---

## The dataflow

```
   Snow frame                Clear-prior frame
        │                            │
        │                            ▼
        │                   Mask2Former (frozen)
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

###### The substrate of the clear prior is interchangeable. The geometric correspondence is what does the work.

---

## A worked example — one frame

| Step | What happens |
|---|---|
| Prior selection | three nearest summer captures, all within a couple of metres |
| Match (DISK + LightGlue + USAC-MAGSAC) | matcher anchors on gate posts, fence wires, masonry corners, distant rooflines |
| Segment the *prior* (Mask2Former) | Cityscapes road class · largest connected component |
| Warp via H⁻¹ | three masks projected back to snow space |
| Fuse + foreground crop | one road region in the lower image — where the road actually is |
| EMA smooth | blend with previous frame's smoothed mask |

###### The composition is invariant across frames. Which features the matcher anchors on shifts; which prior wins shifts; the dataflow does not. Cached `FrameResult` makes downstream renders near-instant.

---

## In motion

A 15-second snow drive on a buried Toronto residential street.
Per snow frame: K=3 nearest summer priors, snow→summer match,
warp the segmenter's road mask back, fuse, EMA-smooth.

The cross-season principle survives motion.

###### outputs/video/boreas_2021_01_26/overlay.mp4 — 15 s, 1024×856, ~7 MB

---

<!-- _class: full-bleed -->

![bg fit](../outputs/video/boreas_2021_01_26/stills/overlay__t005p0.jpg)

---

## What we tried that didn't work

> A pair with **3–7×** more inliers can still produce a worse video.

**Synthetic priors** from past snow frames matched dramatically better in stills.
In motion, each frame's slightly-too-large mask seeded the next frame's prior;
the road overlay drifted outward into bushes within seconds. *Positive feedback loop.*

**Optical-flow propagation** between matched keyframes: vanishing-point flow
stretches the previous mask outward at every step. Same outcome, different mechanism.

###### EMA on the binary mask, α = 0.4, was what survived the motion test. Failure modes are evidence too.

---

## Minimal-shot integrity

| Claim | Status |
|-------|:------:|
| Zero snowy frames touch any model weights | ✓ |
| Zero snowy frames touch any annotation pipeline | ✓ |
| Snow appears only as runtime input | ✓ |
| Pretrained matcher · pretrained segmenter · classical RANSAC | ✓ |
| Reproducible from a clean clone with one command (`make reproduce`) | ✓ |

> The only handle we offered the model on the snow regime
> was the clear prior of the same place.

---

## The contribution is a *primitive*

> The constants-bridge: a composition that takes a model trained on regime A,
> an inference target in regime B, and a known invariant between A and B,
> and uses the invariant to transfer the model into B without retraining.

The snow plough is one **consumer** of this primitive.
The road-overlay channel is what the constants-bridge looks like
when consumed for buried-road perception.

###### The output answers *where the road should be*, not *where to drive*. Other channels (lidar, depth, obstacle detection) keep doing their work; the primitive frees them from solving the road-position problem on a buried road.

---

## Generalising

The shape repeats.

| Regime | Invariant | Why labelling fails |
|---|---|---|
| **Polar Earth observation** | orbital geometry — known passes, known coordinates | polar conditions are seasonally extreme + sparsely sampled |
| **Low-light medical imaging** | patient anatomy across imaging conditions | each new scope/sensor/contrast is its own regime |
| **Off-Earth / hostile manipulation** | rigid-body geometry of task and tools | the operating environment has no in-distribution data |

Snow on a road is one instantiation. The constants-bridge transfers.

> Find what stays the same. Walk across.

---

<!-- _class: title -->
<!-- _paginate: false -->

# Find what stays the same

## and walk across.

###### Constants as the bridge.

---

<!-- _class: footer-card -->
<!-- _paginate: false -->

###### Reproduce

```
git clone <repo>
cd snow-underlay
uv sync --python 3.12
make reproduce
```

###### Read

`README.md` &nbsp;·&nbsp; `docs/writeup.md` (render with `make writeup`) &nbsp;·&nbsp; `docs/index.html` (Pages)

###### Submission

SoTA Commission I — Minimal-Shot Autonomy &nbsp;·&nbsp; May 2026 &nbsp;·&nbsp; Boreas dataset CC BY 4.0

---

<!-- _class: footer-card -->
<!-- _paginate: false -->

###### Appendix

The submission-video plan lives at `docs/submission_video_plan.md` — beat-by-beat shot list, narration, asset paths, music cue points.
