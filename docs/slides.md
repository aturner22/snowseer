# Slide deck outline — Snow-Underlay

> Marp / Slidev compatible. Each `---` is a slide.

---

# Snow-Underlay
### Cross-season image matching for snow-plough autonomy

SoTA Commission I — Minimal-Shot Autonomy

---

## The problem

- Snow ploughs need to scale on demand. Hard to staff year-round.
- Self-driving stacks are trained on dry roads.
- A plough operates in the regime that data deliberately excludes.
- The hardest perception failure: **the road itself is invisible**.

---

## The trick

- Don't learn what a road looks like under snow.
- For nearly every road in the developed world we have a **clear-season prior image** of it (Street View, Mapillary, prior fleet captures).
- Match the live snowy frame to the clear prior.
- Transfer the road segmentation through the alignment.

---

## The pipeline

```
[snow query] ─┐
              ├─> DISK + LightGlue ─> RANSAC homography
[clear prior] ┘                            │     (ground-plane biased)
              └─> Segformer (Cityscapes) ─> road mask
                                                │
                                                ▼
                                       warp + overlay
```

Three pretrained models, all frozen, all trained on clear imagery only.

---

## Minimal-shot integrity (the load-bearing claim)

| Component | Pretrained on | Snow in training? |
|---|---|---|
| DISK | MegaDepth | **No** |
| LightGlue | MegaDepth | **No** |
| Segformer-B0 | Cityscapes | **No** |

Snow appears only at **inference time**, as the runtime input.

---

## Naive baseline → motivation

- Same Cityscapes segmenter, applied directly to the snow frame.
- Fragmented, shifted, or missing road predictions.
- → Cross-season transfer is the cheapest fix.

*[insert side-by-side from outputs/heroes/<id>__naive_baseline.png ↔ <id>__overlay.png]*

---

## Hero results

*[insert 3-4 outputs/heroes/<id>__panel.png from Östersund, Tromsø, Rovaniemi]*

For each: `inliers = N`, `ground-plane bias used = True/False`.

---

## Honest failure case

*[insert one outputs/heroes/<id>__panel.png with worst inlier count]*

Heavy snow, no surviving structure, the system declines an overlay rather than producing a wrong one.

---

## "Simulation environment" framing

- Mapillary as the open-world substrate.
- Fetcher pulls geo-paired snow/clear imagery from 6 known-snowy regions.
- Agent traverses pairs one frame at a time.
- Random start coordinates → "randomised scenario generation" bonus.

---

## What this does *not* claim

- Not a 3D drivable surface estimate.
- Not robust to a blizzard.
- Not a substitute for lidar.
- Just a road **prior** — cheap, snow-naive, surprisingly correct.

---

## Future direction

1. Piecewise-affine warp (drop the planar-scene assumption).
2. Temporal smoothing via odometry.
3. Plough's own prior captures replace Mapillary.

---

## Code, notebook, demo

- Repo: `<github URL>`
- Notebook: `notebooks/01_walkthrough.ipynb` — runs end-to-end on CPU.
- Streamlit demo: `uv run streamlit run demo/streamlit_app.py`
- Video: `<URL>`
