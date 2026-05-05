---
marp: true
theme: snow-underlay
paginate: true
header: "Snow-Underlay  ·  SoTA Commission I — Minimal-Shot Autonomy"
footer: "Constants as the bridge"
---

<!-- _class: title -->
<!-- _paginate: false -->

# Constants as the bridge

## Minimal-shot autonomy, demonstrated on a snow plough

###### Snow-Underlay  ·  SoTA Commission I  ·  May 2026

---

<!-- _class: image-right -->

![bg right:55%](../data/pairs/kiruna_se__173943764513956__2572648156371424/snow.jpg)

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

![bg right:55%](../outputs/heroes/kiruna_se__173943764513956__2572648156371424__naive_baseline.png)

## Off-the-shelf segmentation, applied directly:

a fragmented red mask
that does not survive a glance.

Or worse — silent failure:
zero road predicted,
no warning issued.

---

<!-- _class: pullquote -->

> 27 million miles of road.
> The long tail of conditions
> any of them can be in
> is *longer than the road itself*.

###### We are not going to label our way out of it.

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

---

## The recipe

<div class="recipe">

1. &nbsp;Pull the **live snowy frame** from the plough's camera.
2. &nbsp;Pull a **clear-season prior** of the same coordinates (Mapillary).
3. &nbsp;**Match** the two with a generic frozen feature matcher.
4. &nbsp;Estimate a **homography**, biased toward the ground plane.
5. &nbsp;Run a generic Cityscapes road segmenter — **on the clear prior only**.
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

Every learned component is **frozen**.
Snow appears only at inference, as the runtime input.

---

<!-- _class: full-bleed -->

![bg fit](../outputs/heroes/gallivare_se__1113124103239974__202392698419785__panel.png)

---

###### Gällivare, Sweden  ·  snow-banked road, parking sign  ·  128 RANSAC inliers

## Snow query · Clear prior + Cityscapes road · Cross-season overlay · Naive direct-on-snow

The matcher anchors on what stays constant —
buildings, signs, the parking-sign post — and the homography
carries the road mask across.

---

<!-- _class: full-bleed -->

![bg fit](../outputs/heroes/kiruna_se__173943764513956__2572648156371424__panel.png)

---

###### Kiruna, Sweden  ·  red Falun-style houses, road buried  ·  47 RANSAC inliers

## A scene with no visible road surface.

The Falun-red houses survive the season change.
The matcher uses them. The homography lands on the right plane.
The road mask transfers cleanly onto a road
the model can no longer see.

---

<!-- _class: full-bleed -->

![bg fit](../outputs/heroes/kiruna_se__5529843027088716__1189235864845198__matches.png)

---

###### Feature correspondences  ·  green = RANSAC inliers (kept)  ·  faint red = rejected outliers

## What the matcher actually finds.

Every line is a candidate correspondence.
RANSAC keeps the ones that fit a single homography (green)
and discards the rest (faded red).

The matcher has never seen snow.
It anchors on rooflines, signage, distant horizons —
the constants between the two scenes.

---

## What we did **not** see coming

> A pair with **238 inliers** was rated NOT_GOOD —
> the inliers concentrated on building façades and the homography
> aligned the buildings rather than the road plane.

> A pair with **17 inliers** was rated GREAT —
> the few it had happened to land on the road.

###### Inlier count alone is not a reliable predictor of overlay quality. The system needs a human in the loop on the input *and* the output.

---

## Minimal-shot integrity

| Claim | Status |
|-------|:------:|
| Zero snowy frames touch any model weights | ✓ |
| Zero snowy frames touch any annotation pipeline | ✓ |
| Snow appears only as runtime input | ✓ |
| Pretrained matcher · pretrained segmenter · classical RANSAC | ✓ |
| Reproducible from a clean clone, one command | ✓ |

> The only handle we offered the model on the snow regime
> was the clear prior of the same place.

---

## What we didn't claim

We didn't train. We didn't fine-tune. We didn't collect a snow corpus.
We didn't write a single line of snow-aware logic.

We didn't claim the system replaces lidar or 3D depth.
The output is a 2D road *prior*, not a drivable-surface estimate.

We didn't claim novelty in any single component.
The novelty, such as it is, is in the **composition**.

---

## Generalising

The structure of the move:

> A model trained on regime A.
> An inference target in regime B.
> A known correspondence between the two.
> Transfer through the correspondence.

Snow on a road is one instance.

*Low-light medical imaging without low-light training data.
Polar earth observation without polar training data.
A manipulator on Mars without Mars training data.*

Each admits the same structure.

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
export MAPILLARY_TOKEN=<token>
make demo
```

###### Read

`README.md` &nbsp;·&nbsp; `docs/writeup.pdf` &nbsp;·&nbsp; `notebooks/01_walkthrough.ipynb`

###### Submission

SoTA Commission I — Minimal-Shot Autonomy &nbsp;·&nbsp; May 2026
