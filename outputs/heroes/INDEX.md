# Hero outputs index

Curated selection from `outputs/heroes/*__panel.png`. The pipeline ran on 16 Mapillary pairs from Kiruna (Sweden) and Rovaniemi (Finland). Inlier counts are after RANSAC under the ground-plane-biased homography.

## Strong successes (use as hero shots in slides / video)

| Pair ID | Inliers | Ground plane | What it shows |
| --- | ---:| ---:| --- |
| `kiruna_se__173943764513956__2572648156371424` | 47 | ✓ | Red Falun-style houses; ground fully buried in snow; overlay precisely tracks the rightward sweep of the road. **The poster image.** |
| `kiruna_se__474352240535542__372889160790900` | 74 | ✓ | Brick residential building; partial snow visibility; clean overlay. |
| `kiruna_se__837293103527079__4180807925287533` | 43 | ✓ | Snow-banked residential street; overlay correctly threads between the snow piles. |
| `rovaniemi_fi__1268529072009484__1950094872070678` | 62 | ✓ | Tunnel interior; works even in low-feature environments (low-snow, but useful for showing the matcher does not need scene texture variety). |
| `rovaniemi_fi__26341928928771658__1533316530937158` | 98 | ✓ | Tunnel interior, second example. Highest inlier count of the run. |

## Honest drift case

| Pair ID | Inliers | Ground plane | What it shows |
| --- | ---:| ---:| --- |
| `rovaniemi_fi__1379006517334210__1856349351185152` | 6 | ✗ | Tunnel exit; overlay covers most of the road but extends onto a non-road area on the right. The lower-half restriction did not engage (too few low-half matches), so the homography is biased by the tunnel structure rather than the ground plane. |
| `rovaniemi_fi__1263019079098044__275111850989861` | 6 | ✓ | Revontuli tunnel entrance; gold visual pair (Feb 2026 ↔ Jul 2020), but only 6 lower-half matches survived RANSAC. The overlay is approximately correct but slightly drifted. Good honest mid-quality example. |

## Graceful failures (system declines rather than hallucinates)

| Pair ID | Matches | Inliers | What it shows |
| --- | ---:| ---:| --- |
| `kiruna_se__1132166577296546__430202249821980` | 4 | 0 | Snow-at-night vs clear daytime; matcher finds 4 candidate matches, RANSAC rejects all. **No overlay is produced** — the safety-positive failure mode for a plough. |
| `rovaniemi_fi__2119841382121712__231758195414969` | 7 | 0 | Motion-blurred night driving vs autumn daytime; same safety-positive failure. |

## Suggested narrative ordering for the video

1. Kiruna red houses (best success — set the story)
2. Kiruna snow-banked residential (success in heavier snow)
3. Kiruna brick building (success in lighter snow)
4. Rovaniemi tunnel entrance (mid-quality / partially correct)
5. Kiruna 1132166 graceful failure (honest limit)
