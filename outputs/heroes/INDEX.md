# Hero outputs index — v0.2

39 Mapillary pairs total across 4 regions (Kiruna, Rovaniemi, Östersund-E45, Tromsø). After content-level curation (RANSAC ≥ 15 inliers under the ground-plane-biased homography, with optional iterative road-mask refinement), **13 pairs are accepted**. The accept rate (33 %) is dominated by Mapillary's heading-metadata unreliability, not by any model failure.

## Strong successes (top picks for slides / video)

| Pair ID | Inliers | Refined | What it shows |
| --- | ---:| ---:| --- |
| `kiruna_se__191430299489835__1372566446455224` | **238** | – | Wide curved intersection; ground fully buried in snow; overlay precisely tracks the rightward sweep including the curb. **The new poster image.** |
| `rovaniemi_fi__26582409111354697__510356658196654` | 101 | – | Tunnel interior; high-confidence alignment in a low-snow but feature-rich environment. |
| `rovaniemi_fi__26341928928771658__1533316530937158` | 98 | – | Tunnel interior, second example. |
| `kiruna_se__511790163186080__336351897821514` | 83 | – | Residential street with parked car in the snow frame; overlay still threads the road past the obstacle. |
| `kiruna_se__474352240535542__372889160790900` | 74 | – | Brick residential building. |
| `rovaniemi_fi__1653107918206813__1495016130852680` | 73 | – | Night-vs-day intersection — dramatic lighting change, pipeline still aligns. |
| `rovaniemi_fi__1268529072009484__1950094872070678` | 55 | – | Tunnel interior. |
| `kiruna_se__173943764513956__2572648156371424` | 47 | – | Red Falun-style houses; ground fully buried in snow; overlay cleanly tracks the road. |
| `rovaniemi_fi__172183294726470__850381322354524` | 47 | – | Highway under a bridge, night-vs-day. |
| `kiruna_se__837293103527079__4180807925287533` | 43 | – | Snow-banked residential street; overlay correctly threads between the snow piles. |
| `rovaniemi_fi__516661399688105__513584009819912` | 41 | – | Cleanly-aligned residential. |
| `rovaniemi_fi__1362765774852728__7505659766205853` | 33 | – | Highway under bridge — *content-borderline* (tight inlier count but visually recognisable as the same road from a slightly different lane). |
| `rovaniemi_fi__1548379636222507__1624262148147376` | 27 | – | Same highway corridor, second crossing. |
| `rovaniemi_fi__1606074692919351__1020825555126586` | 26 | – | Night residential intersection, low-light. |
| `kiruna_se__294581095485214__169070348464227` | 19 | – | Borderline accepted — ground-plane bias engaged, low inlier count but visually consistent. |

## Honest drift cases (rejected by curator; useful as "limit of one-shot homography" exhibits)

| Pair ID | Inliers | Notes |
| --- | ---:| --- |
| `rovaniemi_fi__1263019079098044__275111850989861` | 6 | **Revontuli shopping centre tunnel entrance** — Feb 2026 snow ↔ Jul 2020 clear, 0.45 m apart. The matcher anchors on the tunnel structure but only 6 lower-half matches survive RANSAC. The overlay is approximately correct with a small lateral drift. Iterative refinement does not help here — the snow side has no usable road-surface features. |
| `rovaniemi_fi__1379006517334210__1856349351185152` | 6 | Tunnel exit, similar drift profile. |

## Graceful failures (matcher correctly declines)

| Pair ID | Matches | Inliers | What it shows |
| --- | ---:| ---:| --- |
| `kiruna_se__1132166577296546__430202249821980` | 4 | 0 | Snow-at-night vs daytime; matcher finds 4 candidate matches, RANSAC rejects all. **No overlay produced.** Safety-positive failure mode for a plough. |
| `rovaniemi_fi__2119841382121712__231758195414969` | 7 | 0 | Motion-blurred night driving vs autumn daytime; same safety-positive failure. |

## Content-mismatched (rejected — Mapillary heading metadata wrong)

3 Rovaniemi pairs (`1457451…`, plus the borderline `1362765…` and `1548379…`) and many of the Östersund-E45 pairs are at the same lat/lng + heading but visually different scenes — opposing carriageways of the same divided highway, or different-side-streets at the same intersection node. The curator rejects these via the inlier threshold; the audit calls them out as evidence that **GPS+compass alone isn't enough — content-level sanity is required**.

## Suggested narrative ordering for the video

1. Kiruna 191430 wide-intersection (poster — set the story)
2. Kiruna 173943 red Falun houses (full snow burial)
3. Kiruna 837293 snow-banked residential (overlay threads between snow piles)
4. Rovaniemi 1653107 night-intersection (dramatic lighting change)
5. Rovaniemi 1263019 Revontuli tunnel-entrance (mid-quality / drift case — honesty)
6. Kiruna 1132166 graceful failure (honest limit)
