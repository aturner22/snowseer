---
marp: true
theme: snow-underlay
paginate: true
header: "Snow-Underlay  ·  SoTA Commission I"
footer: "Constants as the bridge"
---

<!-- _class: title -->

# Constants as the bridge

## Minimal-shot autonomy, demonstrated on a snow plough

---

## A snow plough's job is short

Keep the road clear.

The catch: while the plough is doing it,
the road is invisible.

Curbs are buried. Lane markings are gone. The seam
between asphalt and garden is no longer drawn.

A self-driving stack trained on Cityscapes will report,
with calibrated confidence, that the entire scene is sky.

---

## We are not going to label our way out

> 27 million miles of road.
> The long tail of conditions any of them can be in
> is longer than the road itself.

The familiar response — annotate snowy roads, dust storms, fog,
washouts, lava — does not scale.

There is a different move.

---

## The move

For almost every regime where autonomy fails for lack of data,
there is an *adjacent regime* — temporally, seasonally, geographically —
where data exists, and where the parts that matter are the same.

The plough's road is the same road it was last July.

The curb hasn't moved. The hydrant hasn't moved.

The road's *appearance* has changed completely.
The road's *position in space* has not.

---

## Constants as the bridge

If we can identify what stays constant between the data-rich regime
and the data-poor one, we can extend our existing models into
the new regime **without learning a single thing about it**.

We use the constants as a bridge.

---

## The example, in six steps

1. Pull the live snowy frame from the plough's camera.
2. Pull a clear-season prior of the same coordinates (Mapillary).
3. Match the two with a generic frozen feature matcher.
4. Estimate a homography, biased toward the ground plane.
5. Run a generic Cityscapes road segmenter on **the clear prior only**.
6. Warp the road mask onto the snowy frame.

The plough now knows where the road is. It has not been trained on snow.

---

![w:100%](../outputs/heroes/gallivare_se__1113124103239974__202392698419785__panel.png)

###### A user-rated GREAT pair. Snow query · clear prior + Cityscapes road · cross-season overlay (rust) · naive direct-on-snow (the failure condition).

---

## Architecture

| Component | Role | Pretrained on |
|-----------|------|---------------|
| **DISK** *(NeurIPS '20)* | Local features | MegaDepth, no snow |
| **LightGlue** *(ICCV '23)* | Sparse matcher | MegaDepth, no snow |
| **USAC-MAGSAC** *(CVPR '20)* | Robust homography | — |
| **Mask2Former** *(CVPR '22)* | Road segmenter | Cityscapes, no snow |

Every learned component is **frozen**. Snow appears only at inference.

---

## Minimal-shot integrity

| Claim | Status |
|-------|:---:|
| Zero snowy frames touch any model weights | ✓ |
| Zero snowy frames touch any annotation pipeline | ✓ |
| Snow appears only as runtime input | ✓ |
| Reproducible from a clean clone, one command | ✓ |

> The only handle we offered the model on the snow regime
> was the clear prior of the same place.

---

## What we showed

**14 user-rated GREAT or OKAY heroes** from a 5-stage curation funnel:
125 candidates → 95 auto pre-filter → 63 deduped → 27 manual snow accept → 19 auto-accepted overlays → 14 manual GREAT or OKAY.

The interesting finding: **inlier count is not a reliable predictor of overlay quality**.

A pair with 238 inliers was rated NOT_GOOD; a pair with 17 inliers was rated GREAT.

The system needs a human in the loop on input *and* output.
We shipped two small Streamlit raters for the work.

---

## What we didn't

We didn't train. We didn't fine-tune. We didn't collect a snow corpus.

We didn't write a single line of snow-aware logic.

The novelty, such as it is, is in the *composition*.

---

## Generalising

The structure of the move:

> A model trained on regime A.
> An inference target in regime B.
> A known correspondence between the two.
> Transfer through the correspondence.

Snow on a road is one instance.
Low-light medical imaging without low-light training data.
Polar earth observation without polar training data.
A manipulator on Mars without Mars training data.

Each admits the same structure.

---

<!-- _class: title -->

# Constants as the bridge.

## We just have to find what stays the same and walk across.

---

###### Reproduce: `make demo` · Repo: github.com/&lt;you&gt;/snow-underlay · Submission: SoTA Commission I, May 2026
