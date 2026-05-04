# Snow-Underlay

**Cross-season image matching for snow-plough autonomy.**

A submission to [SoTA Commission I — Minimal-Shot Autonomy](https://sotaletters.substack.com/p/sota-commission-i-minimal-shot-autonomy).

---

## The pitch

Self-driving systems are trained on clear-weather imagery. A snow plough operates in the regime that data deliberately excludes — roads buried under snow, lane markings invisible, curbs and hydrants hidden. Asking a model trained on dry asphalt to know where the road is when the road is white is asking too much.

But we already have, for almost every road in the developed world, a **clear-season image of it** (Street View, Mapillary, prior fleet captures). This system uses the clear image as a geometric prior:

1. Pull the live snowy frame from the plough camera.
2. Pull a clear-season prior of the same location (open Mapillary imagery, indexed by GPS).
3. Run a generic feature matcher (SuperPoint / DISK + LightGlue) across the two — the matcher anchors on surviving structure: buildings, signs, poles, rooflines.
4. Estimate a homography biased toward the **ground plane** (lower-image matches only).
5. Run a generic Cityscapes-trained road segmenter on the **clear** prior — never on the snow frame.
6. Warp the road mask through the homography onto the snowy frame.

The plough now knows where the road is, and where it isn't, even though it cannot see the road.

## Minimal-shot integrity guarantee

This is the load-bearing claim:

| Component | Pretraining data | Snowy frames in training? |
| --- | --- | --- |
| DISK feature extractor | MegaDepth (outdoor scenes) | **No** |
| LightGlue matcher | MegaDepth | **No** |
| Segformer-B0 | Cityscapes (clear-weather European driving) | **No** |

**Snowy imagery enters the system only at inference time, as the runtime input.** No model weights are ever updated on snowy data.

## Setup

```bash
uv sync --python 3.12
```

## Running

You need a free Mapillary API token from <https://www.mapillary.com/dashboard/developers>:

```bash
export MAPILLARY_TOKEN=<your token>
```

Then:

```bash
# 1. Pull paired snowy/clear frames from open Mapillary imagery.
uv run python -m data.fetch_mapillary

# 2. Run the full pipeline on every pair, caching outputs.
uv run python -m src.pipeline

# 3. Browse the results.
uv run streamlit run demo/streamlit_app.py
```

A single pair can be run end-to-end in isolation:

```bash
uv run python -m src.pipeline --pair-id <region>__<snowID>__<clearID>
```

## Outputs (per pair)

`outputs/heroes/<pair_id>__panel.png` — the headline 3-panel figure: snowy frame, clear frame with detected road, snowy frame with warped road overlay.

`outputs/heroes/<pair_id>__matches.png` — feature correspondence visualization (inliers green, outliers red).

`outputs/heroes/<pair_id>__overlay.png` — the snow frame with road overlay only, for use in slide/video.

`outputs/heroes/<pair_id>__naive_baseline.png` — what happens when you run the same Cityscapes road segmenter directly on the snowy frame, no cross-season prior. This is the failure mode the system avoids.

## Repo layout

```
data/
  fetch_mapillary.py        # primary data path
  pairs/<id>/{snow.jpg, clear.jpg, meta.json}
src/
  matching.py               # DISK + LightGlue
  homography.py             # RANSAC, ground-plane biased
  segmentation.py           # Segformer (Cityscapes, frozen)
  overlay.py                # warp + blend + viz
  pipeline.py               # ties everything together
notebooks/
  01_walkthrough.ipynb      # end-to-end narrative
demo/
  streamlit_app.py          # judges-clickable viewer (cached outputs)
outputs/heroes/             # generated figures
docs/
  writeup.md                # ≤2-page write-up
```

## Honest limitations

- Homography assumes a near-planar scene. When matches concentrate on building façades rather than the road plane, the warp can drift. The lower-image-half restriction mitigates this but does not eliminate it.
- Heavy snow (frosted trees, ground completely white, low contrast) starves the matcher of usable structure. The system will fail gracefully (low inlier count) rather than confidently produce a wrong overlay.
- Mapillary contributor coverage in snowy regions is uneven. The pipeline is dataset-agnostic — any source of geo-tagged image pairs works.

A deliberate failure case is included in `outputs/heroes/` and called out in the notebook.
