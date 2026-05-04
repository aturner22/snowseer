# Hero outputs index — v0.3 (user-curated)

**Three-stage human-in-the-loop curation:**

1. **Automated snow-quality pre-filter** (`src/snow_quality.py`) — Laplacian sharpness, brightness, lower-half edge density. **125 → 95 (76 %)** pass all three thresholds.
2. **Spatial+heading dedup** (50 m / 40°) drops sequential frames from the same drive: **125 → 63 distinct clusters.**
3. **Manual snow-quality curation** (`demo/curate_snow.py`) — user accepts only frames matching a real plough camera (sharp, well-framed, daytime or well-lit): **63 → 27 manually accepted.**
4. **Pipeline + auto post-curation** (RANSAC inliers ≥ 15 under ground-plane bias, with iterative road-mask refinement and largest-connected-component cleanup): **27 → 19 auto-accepted, 8 auto-rejected.**
5. **Manual result rating** (`demo/curate_results.py`) — user rates each overlay panel on a 4-point scale: **10 GREAT, 4 OKAY, 8 NOT_GOOD, 5 AWFUL.**

**Demo set = GREAT + OKAY = 14 heroes.** These are what the video and slides ship with. Inlier count alone is not a reliable predictor of overlay quality — `kiruna_se__191430` with 238 inliers was rated NOT_GOOD; `kiruna_se__245577` with 17 inliers was rated GREAT.

## 🌟 GREAT (10) — the demo set

| Pair ID | Inliers |
| --- | ---:|
| `gallivare_se__1113124103239974` | 128 |
| `lulea_se__1235981388376274` | 101 |
| `gallivare_se__724743419870843` | 83 |
| `gallivare_se__1107706673896225` | 51 |
| `kiruna_se__173943764513956` | 47 |
| `gallivare_se__7240084582787704` | 40 |
| `gallivare_se__345189905208531` | 28 |
| `kiruna_se__1661386470711759` | 25 |
| `kiruna_se__4126780297420111` | 25 |
| `kiruna_se__245577317324651` | 17 |

## 🟡 OKAY (4) — supporting

| Pair ID | Inliers |
| --- | ---:|
| `kiruna_se__5529843027088716` | 299 |
| `gallivare_se__1074104257224613` | 90 |
| `gallivare_se__406156588778534` | 72 |
| `rovaniemi_fi__856601484101706` | 7 |

## 🟠 NOT_GOOD (8) — overlay drifts visibly despite high inlier count

`kiruna_se__191430299489835` (238i), `gallivare_se__948371859886087` (173i), `lulea_se__1416606716782053` (80i), `gallivare_se__432410789201829` (33i), `lulea_se__2293127254488678` (21i), `rovaniemi_fi__851945714561628`, `lulea_se__1448494693308466`, `rovaniemi_fi__1263019079098044` (Revontuli tunnel-entrance — the long-running honest drift case).

## 💀 AWFUL (5) — included only as evidence the system has limits

`lulea_se__1663156397999851` (15i), `rovaniemi_fi__1457451495738922`, `rovaniemi_fi__941648498258328`, `kiruna_se__469089707711944`, `ostersund_e45_se__771172498949499`. Several of these are content-mismatched pairs (Mapillary heading metadata is wrong) where the matcher found false-positive correspondences.

## What the user-curation revealed

The headline finding: **the auto-curation inlier threshold is not a reliable predictor of overlay quality.** A pair with 238 RANSAC inliers can still warp the road mask onto the wrong region of the snow image (because most of the inliers are on building façades, the homography aligns the buildings rather than the road plane). A pair with 17 inliers can produce a clean overlay if those inliers are concentrated on the ground plane and the road geometry is simple.

This is honest and worth documenting in the writeup. The auto-curator catches obvious failures (low matches, low inliers, content mismatch), but the final demo curation requires human visual judgement on the actual overlay output. We built `demo/curate_results.py` as the bridge between automated metrics and final demo material.
