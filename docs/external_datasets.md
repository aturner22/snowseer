---
title: "External snow datasets — acquisition and integration"
subtitle: "How to bring a new physical scene into the pipeline"
---

The pipeline is **substrate-agnostic** — any geo-tagged snow video plus any geo-tagged clear-weather imagery of the same coordinates can be fed through the same `make track` flow. This document is the user-side guide for the four academic / public datasets we surveyed for non-Boreas scenes.

> Every dataset listed below requires a free sign-up and a manual download — the dataset hosts don't expose programmatic mirrors. Once unpacked under `data/external/<name>/`, the rest of the integration is automatic: a small `data/fetch_<name>.py` module maps the source's directory layout into our `data/video/tracks/<id>/{snow,summer}/` shape, then `make oracle TRACK=<id>` followed by `make track TRACK=<id>` produces the standard 6-mp4 + stills bundle exactly the same way the canonical track does.

> **Minimal-shot integrity** is preserved across all of them: every learned component (DISK, LightGlue, Mask2Former) was pretrained on data that contains no snow, and we never train or fine-tune. The snow imagery from these datasets enters only as runtime input to the matcher.

---

## ACDC — Adverse Conditions Dataset with Correspondences

### Source

- **Site**: <https://acdc.vision.ee.ethz.ch/>
- **Paper**: Sakaridis et al., *ACDC: The Adverse Conditions Dataset with Correspondences for Robust Semantic Driving Scene Perception*, ICCV 2021.
- **Provenance**: ETH Zurich Computer Vision Lab (Christos Sakaridis et al.).
- **Coverage**: 4006 adverse-condition images split equally across fog / night / rain / snow, plus matching clear-condition reference images of the same scenes. Predominantly urban Switzerland (Zurich + surrounds).
- **License**: research / non-commercial (CC BY-NC-SA 4.0 in current versions). Cite the ICCV 2021 paper in any publication.

### Steps

1. Visit <https://acdc.vision.ee.ethz.ch/> → click **Download**. Fill the form (name, affiliation, email). Approval is typically instant or within a few hours. Keep the confirmation email.

2. After login, the download grid lists per-condition zips. The minimum set is:

   - `rgb_anon_trainval_snow.zip` — snow images and their paired clear-condition references. ~250 MB.
   - `gt_trainval_snow.zip` *(optional)* — pixel-level Cityscapes-class annotations. Useful for the analysis notebook's integrity audit; not required for the pipeline. ~50 MB.

   Skip `rgb_anon_test_snow.zip` and the other conditions (fog / night / rain) unless you intend to extend the demo into those regimes.

3. Unpack into `data/external/acdc/`:

   ```bash
   mkdir -p data/external/acdc
   cd data/external/acdc
   unzip ~/Downloads/rgb_anon_trainval_snow.zip
   # optional:
   unzip ~/Downloads/gt_trainval_snow.zip
   ```

4. The expected layout after unzip is roughly:

   ```
   data/external/acdc/
   └── rgb_anon/
       └── snow/
           ├── train/
           │   ├── GOPR0122/   ← one directory per physical scene
           │   │   ├── frame_00.png
           │   │   ├── frame_01.png
           │   │   └── …  (typically 4 consecutive frames)
           │   └── …
           └── val/
   ```

   Each `GOPR*` directory is a paired multi-frame clip in the same physical scene. The corresponding clear-condition reference image lives alongside (file naming varies by dataset version; the integration script handles both layouts).

5. Drop a note that the unzip succeeded:

   ```bash
   touch data/external/acdc/.ready
   ```

   The integration script uses this as a sentinel.

### Integration

`data/fetch_acdc.py` maps each `GOPR*` clip into our format:

```
data/video/tracks/acdc_snow_<scene>/
├── snow/
│   ├── frames/         ← copied from rgb_anon/snow/<split>/<scene>/
│   ├── camera_poses.csv  ← built from EXIF GPS where available;
│   │                       otherwise the per-frame Mapillary closeto
│   │                       call uses the dataset's own GPS metadata
│   └── window.json
└── summer/
    ├── frames/         ← per-frame Mapillary closeto pulls
    │                     (radius 50 m, captured June–September, lat/lon
    │                      from each snow frame)
    ├── camera_poses.csv
    └── window.json
```

After the integration script lands the data:

```bash
make oracle TRACK=acdc_snow_<scene>     # audit gate before any cache build
make track  TRACK=acdc_snow_<scene>     # full 6-mp4 + stills bundle
```

The `make oracle` step is mandatory for every new track. It samples per-frame summer priors and verifies (a) at least one usable prior within 30 m, and (b) per-prior road segmentation is non-degenerate. If either fails on too many frames the candidate is rejected without spending cache compute.

---

## MUSES — Multi-Sensor Semantic Perception in Adverse Conditions

### Source

- **Site**: <https://muses.vision.ee.ethz.ch/>
- **Paper**: Brödermann et al., *MUSES: The Multi-Sensor Semantic Perception Dataset for Driving Under Uncertainty*, ECCV 2024.
- **Provenance**: ETH Zurich Computer Vision Lab (same group as ACDC, sibling dataset).
- **Coverage**: 2500 multi-sensor scenes (camera + lidar + radar + event-camera) under adverse weather, including snow. Same Swiss / Zurich-region geography as ACDC.
- **License**: research / non-commercial. Cite ECCV 2024.

### Steps

1. <https://muses.vision.ee.ethz.ch/> → Download → register.

2. Grab `frame_camera_snow.zip` (the camera-only snow split is sufficient for our use; lidar / radar / event are extras). ~400 MB.

3. Unpack into `data/external/muses/`:

   ```bash
   mkdir -p data/external/muses
   unzip ~/Downloads/frame_camera_snow.zip -d data/external/muses/
   touch data/external/muses/.ready
   ```

4. The MUSES layout has explicit `metadata.json` per scene including GPS — easier integration than ACDC because no EXIF probing is needed.

### Integration

Same shape as ACDC: `data/fetch_muses.py` produces `data/video/tracks/muses_<scene>/{snow,summer}/`, then `make oracle` + `make track`.

---

## BDD100K — Berkeley DeepDrive 100K

### Source

- **Site**: <https://www.bdd100k.com/>
- **Paper**: Yu et al., *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning*, CVPR 2020.
- **Coverage**: 100k driving images / video clips across the US, with weather tags including `snowy`. Diverse cities (NY, SF, etc.) — the broadest geographic diversity of the four.
- **License**: BAIR research / non-commercial.

### Steps

1. <https://www.bdd100k.com/> → Get Data → BDD Login → register an academic account.

2. We want the **video clips** (not the still-image labels), filtered by `weather: snowy`:

   - `bdd100k_videos_train.zip` is the canonical set. **Big — ~1.8 TB total**, so don't grab the whole thing. Use the labels JSON to filter snow clips first:

     ```bash
     mkdir -p data/external/bdd100k
     # Step 1: grab labels.json (small) and filter to snow-tagged clips
     wget <bdd_labels_url> -O data/external/bdd100k/labels.json
     # Step 2: write the snow-clip ID list (helper TBD)
     # Step 3: download only those clip mp4s (each ~5 MB)
     ```

   The integration script writes a `snow_clip_ids.txt` from the labels JSON; you then download only the listed clips. Total budget is more like a few GB if we cap at ~50 candidate clips.

3. Unpack into `data/external/bdd100k/videos/`:

   ```bash
   data/external/bdd100k/
   ├── videos/        ← snow-tagged clip mp4s
   ├── labels.json
   └── .ready
   ```

### Integration

BDD100K clips don't always have GPS at the per-frame level — typically only the start coordinate. The fetcher uses the start coord + ego-motion estimation to interpolate GPS along the clip, then runs the per-frame Mapillary closeto query exactly as for ACDC / MUSES.

This is a higher-friction path than ACDC / MUSES; it's listed as a fallback if the European candidates fail.

---

## Audit gate (mandatory for every candidate)

Before any cache-build compute is committed, every external candidate is required to pass three checks. Without ground truth this is a human-in-the-loop step.

1. **Sample the snow data.** Pull ≤ 10 frames; eyeball for: snowfall actually visible (not just slush), camera mounted forward / roof (not out the side of a bus), scene has features the matcher can anchor on (buildings, signs, fence wires, masonry corners), no severe lens occlusion or motion blur.

2. **Sample the per-frame summer priors.** Pull the Mapillary closest-summer image for each of the 10 sampled snow frames. Verify they're geographically + visually plausible. Reject if priors are pointing the wrong way or the closest summer capture is significantly displaced.

3. **Run `make oracle` on a 50-frame slice.** Read the per-frame `n_good_priors` and per-prior segmentation coverage. Confirm the candidate window before any full cache build.

If any of (1), (2), (3) fail, the candidate is archived under `_archive/data/audit_<id>/` with a one-line rejection reason and we move to the next.

The audit-gate discipline is itself part of the contribution: **never demonstrate the pipeline on data that has no chance of matching, and never trust a "candidate window" before its priors have been segmented and inspected.** The `make oracle` target is the codified version of this rule. (For the audit log of candidates that *failed* the gate during this submission, see the analysis notebook's "What we tried" section.)

---

## Beyond these four

The pipeline accepts any geo-tagged snow imagery; these four are the academic candidates that fit our 3-day window. Other plausible substrates we did not try in this submission (each would follow the same `data/external/<name>/` + `data/fetch_<name>.py` pattern):

- **CADC** (Canadian Adverse Driving Conditions) — 50–100 frame scenes with full sensor calibration. Cleanest data; longest integration tail.
- **DENSE / SeeingThroughFog** (Daimler) — dense adverse-weather coverage. Heavier sensor suite.
- **Public dashcam YouTube** — abundant snowy footage but GPS extraction is unreliable.
- **Operator's own captures** — the production case. Same pipeline; tighter geometric prior; no contributor-coverage gaps.

The substrate question is interchangeable. The geometric correspondence is the load-bearing piece.
