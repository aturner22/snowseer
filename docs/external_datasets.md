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

### What the paper guarantees

The ACDC paper confirms three things relevant to us, with direct quotes:

> *"Our dataset and benchmark are publicly available at <https://acdc.vision.ee.ethz.ch>"*

> *"Our camera also provides GPS readings, which allow us to establish image-level correspondences between adverse-condition and normal-condition recordings."*

So per-image GPS metadata is in the dataset, and each adverse image has a paired clear-condition reference established by the authors via dynamic-programming matching of the GPS sequences. Snow split: 1000 images (400 train + 100 val + 500 test).

### What I know about the portal

The site at <https://acdc.vision.ee.ethz.ch/> is a Vue.js single-page app backed by an authenticated **benchmark portal**: registered users browse a list of "packages" (zips uploaded by the admins), request access to the ones they need, and receive a per-package download via the SPA after the access request is approved. The package filenames are admin-set rather than predictable — they follow the conventions the paper hints at (`rgb_anon` for images, `gt_anon` for ground truth, condition-tagged) but the exact names I cannot produce without logging in. The paper itself lists the dataset URL but does not specify package filenames.

### Steps

1. Go to <https://acdc.vision.ee.ethz.ch/> → click **Register** (top-right). Fill the form (name, affiliation, email). Confirm the email link. Approval is typically same-day.

2. Log in. Navigate to **Downloads** (top nav) — this is where the package list lives.

3. **In the package list, look for the snow-related rows.** Likely one or both of:
   - A row with **rgb_anon + snow** in its name → request access. This contains the snow images with the matched clear-condition references per frame.
   - A row with **gt_anon + snow** if you also want the pixel annotations. We don't strictly need them for the pipeline; useful for the notebook's integrity audit.

   You may need to click "Request access" on each row. A second email arrives once an admin grants it (usually quick, sometimes a few hours).

4. Once granted, click **Download** on the snow row(s). The SPA will pull the zip via `/api/downloadPackage/...` with your auth token. Save to `~/Downloads/`.

5. **Tell me what filename(s) you got** + paste the top-level `unzip -l <file>.zip | head -30` listing. I'll write `data/fetch_acdc.py` against the actual layout, not an invented one. The paper guarantees the GoPro frame naming (`GOPR<NNNN>_frame_<NNNNNN>_rgb_anon.png`) but the directory wrapper varies.

6. Unpack into `data/external/acdc/` and drop a sentinel:

   ```bash
   mkdir -p data/external/acdc
   unzip ~/Downloads/<the-actual-filename>.zip -d data/external/acdc/
   touch data/external/acdc/.ready
   ```

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

1. Visit <https://muses.vision.ee.ethz.ch/> → Download → register. Approval is typically same-day (same group as ACDC).

2. From the portal pick the **camera-only snow split** if it's offered separately; otherwise the full snow archive (lidar / radar / event sensors are extras we don't use). I haven't verified specific zip filenames against the live portal — paste the file list back to me once you've downloaded and I'll write `data/fetch_muses.py` against the actual layout.

3. Unpack into `data/external/muses/`:

   ```bash
   mkdir -p data/external/muses
   unzip ~/Downloads/<muses-snow-zip>.zip -d data/external/muses/
   touch data/external/muses/.ready
   ```

4. The MUSES paper describes per-scene metadata including GPS, so integration should be cleaner than EXIF-probing. Confirmation pending the actual file tree.

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

## CADC — Canadian Adverse Driving Conditions

**Recommended fallback.** CADC is the cleanest non-Boreas snow scene candidate for this pipeline because the dataset is direct-download (no benchmark portal), GPS / INS poses are bundled per scene, and multi-camera calibration is included.

### Source

- **Site**: <http://cadcd.uwaterloo.ca/>
- **Paper**: Pitropov et al., *Canadian Adverse Driving Conditions Dataset*, IJRR 2021.
- **Devkit**: <https://github.com/mpitropov/cadc_devkit> — includes `download_cadcd.py` for direct scene downloads.
- **Coverage**: 7,000 annotated frames across multiple winter weather conditions, Region of Waterloo, ON, Canada. Multi-camera (8 Ximea cameras) + Velodyne VLP-32C lidar + Novatel OEM638 GNSS+INS.
- **License**: research-permissive (consult site for exact terms).
- **Sample scenes**: the devkit demos `2019_02_27_0027` and `2019_02_27_0033` — reasonable starter scenes.

### Steps

1. Clone the devkit:

   ```bash
   git clone https://github.com/mpitropov/cadc_devkit ~/cadc_devkit
   ```

2. Use `download_cadcd.py` to grab one or two scenes. Example (consult devkit help):

   ```bash
   python ~/cadc_devkit/download_cadcd.py --date 2019_02_27 --scene 0027 --target data/external/cadc/
   ```

3. After download, paste the actual top-level layout back to me (`tree -L 3 data/external/cadc/`) and I will write `data/fetch_cadc.py` against the real structure. The paper guarantees per-frame GPS in the GNSS+INS log; the front-facing camera is conventionally one of the eight, and the calibration files include intrinsics + extrinsics. The exact file naming I cannot quote without seeing the actual download.

4. Drop a sentinel:

   ```bash
   touch data/external/cadc/.ready
   ```

### Why this is the cleanest fallback

- **Direct download** (no portal access dance, unlike ACDC / MUSES).
- **GPS bundled** at multi-Hz cadence, so per-frame Mapillary `closeto` summer-prior queries work out of the box.
- **Snow-specific** by design (the dataset's whole point is adverse winter conditions).
- **Forward-facing dashcam geometry** is genuinely there (Ximea cameras mounted on the Autonomoose vehicle), so the pipeline's "where is the road in front of the camera" question is well-posed.

The only friction is the manual download step; the pipeline integration is straightforward once a scene is on disk.

---

## Beyond these candidates

The pipeline accepts any geo-tagged snow imagery; the candidates above are the academic ones that fit our 3-day window. Other plausible substrates we did not try in this submission (each would follow the same `data/external/<name>/` + `data/fetch_<name>.py` pattern):

- **DENSE / SeeingThroughFog** (Daimler) — dense adverse-weather coverage. Heavier sensor suite.
- **Public dashcam YouTube** — abundant snowy footage but GPS extraction is unreliable. Russian-style "Registrator" dashcams write GPS to a sidecar log; most other YouTube clips do not.
- **Operator's own captures** — the production case. Same pipeline; tighter geometric prior; no contributor-coverage gaps.

The substrate question is interchangeable. The geometric correspondence is the load-bearing piece.
