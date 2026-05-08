---
marp: true
theme: snow-underlay
paginate: true
header: "snowseer  ·  SoTA Commission I — Minimal-Shot Autonomy"
footer: "Constants as the bridge"
---

<!-- _class: title -->
<!-- _paginate: false -->

# snowseer

## Constants as the bridge

###### Minimal-shot autonomy  ·  SoTA Commission I  ·  May 2026

---

<!-- _class: image-right -->

![bg right:55%](../outputs/video/boreas_2021_01_26/stills/overlay__t001p0.jpg)

## When labels can't keep up

A snow plough's job is short: **keep the road clear.**

The catch — while the plough is doing it,
**the road is invisible.**

Cityscapes-trained perception, applied directly,
reports the entire scene as sky.

Annotating the long tail of road / weather / time-of-day
combinations is uneconomic.

---

<!-- _class: image-right -->

![bg right:55%](../outputs/video/boreas_2021_01_26/stills/snow_naive_overlay__t005p0.jpg)

## Off-the-shelf segmentation, applied directly

A confident red mask
that does not survive a glance.

Or worse — silent failure:
zero road predicted,
no warning issued.

###### Minimal-shot autonomy: how a perception system survives in regimes it has not been heavily trained on. The default — *collect more data and retrain* — does not scale across the long tail.

---

## A different move

For almost every regime where autonomy fails for lack of data,
an adjacent regime exists — temporally, seasonally, geographically —
where data is rich, and where the parts that matter are the same.

The plough's road is the same road it was last July.

The curb hasn't moved. The hydrant hasn't moved.

The road's *appearance* has changed completely.
The road's **position in space** has not.

---

<!-- _class: principle -->

# The constants-bridge

A composition that takes a model trained on regime A,
an inference target in regime B,
and a known invariant linking the two,
and uses the invariant to transfer the model into B
**without retraining.**

###### *Generalisation, not memorisation.*

---

## The recipe (per snow frame)

<div class="recipe">

1. &nbsp;Pull the **live snowy frame** from the plough's camera.
2. &nbsp;Pull a **clear-season prior** of the same coordinates.
3. &nbsp;**Match** the two with a frozen feature matcher.
4. &nbsp;Estimate a **homography**, biased toward the ground plane.
5. &nbsp;Run a Cityscapes road segmenter — **on the clear prior only.**
6. &nbsp;**Warp** the road mask onto the snowy frame.

</div>

The plough now knows where the road is.
No model in the pipeline has been trained on snow.

---

## Architecture

| Component | Role | Pretrained on |
|-----------|------|---------------|
| **DISK** &nbsp;*(NeurIPS '20)* | Local features | MegaDepth · no snow |
| **LightGlue** &nbsp;*(ICCV '23)* | Sparse matcher | MegaDepth · no snow |
| **USAC-MAGSAC** &nbsp;*(CVPR '20)* | Robust homography | — |
| **Mask2Former** &nbsp;*(CVPR '22)* | Road segmenter | Cityscapes · no snow |
| **Boreas** &nbsp;*(IJRR '23)* | Snow + paired summer captures | — (CC BY 4.0) |

Every learned component is **frozen.**
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

<!-- _class: full-bleed -->

![bg fit](../outputs/video/boreas_2021_01_26/stills/overlay__t005p0.jpg)

---

## What we built

A 15-second snow drive on a buried Toronto residential street.
Per snow frame: K=3 nearest summer priors, snow→summer match,
warp the segmenter's road mask back, fuse, EMA-smooth.

The cross-season principle survives motion.

A second snow drive of the same loop, different snowfall,
runs unchanged. A 27-pair static-stills precursor across
northern Sweden and Finland predates the video extension.

###### `outputs/video/boreas_2021_01_26/overlay.mp4` — reproduce with `make reproduce`.

---

## Known limitations

**Synthetic priors from past frames** — 3-7× more inliers than snow→summer, but in motion each frame's slightly-too-large mask seeded the next frame's prior. The road overlay drifted outward into vegetation. *Positive feedback loop.* Rejected.

**Optical-flow propagation** — vanishing-point flow on a forward-driving camera stretches the previous mask outward at every step. Same shape, different mechanism. Rejected.

**Inlier count is not a reliable predictor of overlay quality.** Matches concentrated on building façades can warp the road mask onto buildings. The system needs a human in the loop on input and output.

**Not real-time.** Matcher-bottlenecked at ~16 s per frame on Mac CPU. A deployment-engineering choice; first item in next-steps.

###### Failure modes that look better in stills and worse in motion are themselves evidence of mechanism.

---

## Minimal-shot integrity

| Claim | Status |
|-------|:------:|
| Zero snowy frames touch any model weights | ✓ |
| Zero snowy frames touch any annotation pipeline | ✓ |
| Snow appears only as runtime input | ✓ |
| Pretrained matcher · pretrained segmenter · classical RANSAC | ✓ |
| Reproducible from a clean clone with `make reproduce` | ✓ |

> The only handle the system has on the snow regime
> is the clear-season prior of the same place.

---

## Generalising

| Regime | Invariant | Why labelling fails |
|---|---|---|
| **Polar Earth observation** | orbital geometry — same satellite, same coordinates, known cadence | polar conditions are seasonally extreme + sparsely sampled |
| **Low-light medical imaging** | patient anatomy across imaging conditions | each new scope/sensor/contrast is its own regime |
| **Agricultural off-road** | field geometry from a previous-season drone overflight | every season + crop + region is a new long-tail entry |

Snow on a road is one instantiation. The constants-bridge transfers.

---

## Where this could go

The constants-bridge primitive is the foundation for
**a general image-banking-and-transfer appliance:**

> Live frame + location → register against a banked clear-conditions image → transfer any pre-computed annotation into the live frame.

**Consumers** — snow / fog / dust / smoke / heavy rain / night driving · seeing round corners · heads-up display navigation · construction-zone delta detection · industrial inspection in obscured environments · agricultural autonomy.

**Five v3 items the prize money funds**

1. **Real-time matcher** — DISK + LightGlue (or RoMa / LoFTR) on GPU; ~16 s → ~30 ms.
2. **Visual place recognition** front-end for GPS-denied environments.
3. **Multi-source bank** — substrate-pluggable layer revived with dense correspondence.
4. **Hardware prototype** — Jetson-class device with HUD output; live snow / fog / night.
5. **Two more constants-bridge instances** for proof of generality.

The next phase is the appliance, of which the snowplough is one consumer.

---

## Reproduce

```
git clone https://github.com/aturner22/snowseer
cd snowseer
uv sync --python 3.12
make reproduce
```

`make track TRACK=<id>` — full pipeline on any registered track.
`make stills` — static-stills precursor (needs `MAPILLARY_TOKEN`).
`make oracle TRACK=<id>` — pre-flight gate before a new cache build.
`make notebook` — re-execute `docs/analysis.ipynb` in place.

---

<!-- _class: title -->
<!-- _paginate: false -->

# Constants as the bridge

## A primitive — and an appliance

###### snowseer · SoTA Commission I — Minimal-Shot Autonomy · May 2026

---

<!-- _class: footer-card -->
<!-- _paginate: false -->

###### Read

`README.md` &nbsp;·&nbsp; `docs/writeup.md` &nbsp;·&nbsp; `docs/analysis.ipynb` &nbsp;·&nbsp; `docs/index.html`

###### Submission

[SoTA Commission I — Minimal-Shot Autonomy](https://sotaletters.substack.com/p/sota-commission-i-minimal-shot-autonomy)
&nbsp;·&nbsp; May 2026 &nbsp;·&nbsp; Boreas dataset CC BY 4.0
