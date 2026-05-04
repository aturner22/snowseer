# Methodology audit — v0.1 baseline

Tag: `v0.1-baseline` (commit `af5dcdd`).

## Summary

The v0.1 baseline runs end-to-end and produces 5 strong overlays out of 16 pairs. The remaining 11 pairs are a mix of mid-quality (drift / dashboard pollution / lighting), content-mismatched, and graceful failures. Three concrete bugs surfaced. Two are fixable in <50 lines of code (this audit applies them); the third requires a curation step (Phase B.1) and an iterative-refinement / segmenter-upgrade pass (Phase C).

## Class-id verification

Loaded `nvidia/segformer-b0-finetuned-cityscapes-512-1024` and inspected `model.config.id2label`. Class 0 is **road**, as the code assumed. There is no off-by-one bug in `src/segmentation.py`.

```
0: road
1: sidewalk
2: building
…
18: bicycle
```

## Per-pair visual review

Each pair was tagged after viewing its `__panel.png` (or `__matches.png` for the failure cases).

| Bucket | Count | Pairs |
| --- | ---:| --- |
| ✅ Strong (same location, clean overlay) | 5 | `kiruna_se__173943…` (red Falun houses, 47i), `kiruna_se__474352…` (brick building, 74i), `kiruna_se__837293…` (snow-banked residential, 43i), `rovaniemi_fi__1268529…` (tunnel interior, 62i), `rovaniemi_fi__26341928…` (tunnel interior, 98i) |
| ⚠ Mid-quality (same location; drift, dashboard pollution, or large lighting change) | 6 | `rovaniemi_fi__1263019…` (Revontuli tunnel-entrance, 6i, slight drift); `rovaniemi_fi__1379006…` (tunnel-exit drift, 6i); `kiruna_se__146117…`, `kiruna_se__301760…`, `kiruna_se__518714…`, `kiruna_se__865461…` (windshield-blocked or night-vs-day) |
| ❌ Content mismatch (Mapillary heading is wrong; pairs are not actually the same scene) | 3 | `rovaniemi_fi__1362765…`, `rovaniemi_fi__1457451…` (opposing carriageways of the same divided highway), `rovaniemi_fi__1548379…` (different street despite tight GPS) |
| 💀 Graceful failure (matcher correctly rejects) | 2 | `kiruna_se__1132166…` (4 matches, 0 inliers — snow-at-night vs daytime), `rovaniemi_fi__2119841…` (motion-blurred night, 0 inliers) |

## Concrete bugs found

### Bug 1 (FIXED in this audit) — RANSAC ground-plane window included the dashboard
`src/homography.py` previously kept matches where `y >= 0.5 * H`. The bottom 10–15 % of every Mapillary image (when uploaded from a car) is the **dashboard**, not the ground plane. Different cars have different dashboards; matching dashboard-vs-dashboard pollutes the homography fit with non-corresponding features. Fix: added `dashboard_y_frac = 0.85` upper bound, so the ground-plane window is `y ∈ [0.5 H, 0.85 H]`.

### Bug 2 (FIXED in this audit) — Segformer over-predicts class 0 on dashboards
The blue dashboard region of clear images is regularly classified as 'road' by Segformer-B0-Cityscapes — flat colour, lower image position, and a class distribution dominated by road in Cityscapes. The road mask therefore extends into the dashboard, and when warped, drags the overlay onto a region of the snow image that is not the road. Fix: `src/segmentation.py` now zeros the road mask below the same `dashboard_y_frac = 0.85` band. Same cutoff on both sides keeps the geometry consistent.

### Bug 3 (NOT FIXED YET — addressed in Phase B.1) — Mapillary heading metadata is unreliable
At least 3 of 16 pairs have ≤17.5° heading delta but obviously different visual content (opposing carriageways of the same divided road, or different streets at the same lat/lng). Mapillary's `compass_angle` reflects the device's compass at capture, which can be wildly out of phase with the actual camera orientation (phone in the wrong pocket, mount loose, etc.). No fix at the metadata level is possible — it requires a content-level pair sanity check. That's Phase B.1.

## Code re-read

Re-read `matching.py`, `homography.py`, `segmentation.py`, `overlay.py`, `pipeline.py`. Beyond the two bugs above, no other defects found:

- DISK keypoints are returned as `(x, y)` (not `(i, j)`) — confirmed by reading `kornia.feature.DISK.forward` source. We pass them straight through; consistent.
- `cv2.findHomography(src=kpts0, dst=kpts1)` returns `H_snow→clear`. To warp the *clear-image* road mask into the snow image we need `H_clear→snow`, hence the `np.linalg.inv(H)` in `pipeline.py`. Convention is consistent.
- `cv2.warpPerspective` takes `M` as the public src→dst transform (it inverts internally). We pass `H_inv` (=`H_clear→snow`) where src is the clear-image mask. Consistent.
- Image resize before matching: pipeline rescales both images to a common `max_dim = 1024` *before* matching. Both images are scaled independently of each other — that's fine, the homography is in pixel space of the resized images, and we never compose with the original-resolution coords. No leak.

## Drift cases — diagnosis

Two pairs (`rovaniemi_fi__1263019…` Revontuli tunnel-entrance, `rovaniemi_fi__1379006…` tunnel-exit) are visibly drifted despite the ground-plane bias having been engaged. The cause is **scene non-planarity**: tunnel walls and ceilings dominate the image structure, the lower-image-half restriction *almost* selects the ground but the surviving inliers are still too few (6 each) for RANSAC to robustly disambiguate the road plane from the tunnel-floor-plus-walls hybrid. Fix is **iterative refinement** (Phase C.1): warp the road mask using the initial H, subset matches to those inside the warped road region, refit. Empirically this is what the user-original-intuition called "wiggling the image until features line up."

## Action items

| ID | Description | Phase |
| --- | --- | ---:|
| 1 | Restrict ground-plane window upper bound to `y ≤ 0.85 H` (anti-dashboard) | A.2 ✅ |
| 2 | Zero the road mask below `y = 0.85 H` (anti-dashboard segmenter pollution) | A.2 ✅ |
| 3 | Content-level pair sanity check; reject pairs with `inliers < 15` | B.1 |
| 4 | Sweden recovery: widen bboxes 600 m → 1500 m, add Sundsvall / Gällivare / Luleå / Reykjavík | B.2 |
| 5 | Iterative homography refinement on road region | C.1 |
| 6 | Try Mask2Former-Cityscapes / softmax-thresholded mask / SAM2 click fallback | C.2 |
| 7 | IoU(naive, overlay) numeric on every panel; identity-homography baseline | D |
| 8 | Polish viz, Streamlit, video, slides, README banner | E |
| 9 | Submission package | F |
