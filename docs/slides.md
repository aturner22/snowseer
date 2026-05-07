---
marp: true
theme: snow-underlay
paginate: true
header: "Snow-Underlay  ·  SoTA Commission I — Minimal-Shot Autonomy"
footer: "Constants as the bridge — in motion"
---

<!-- _class: title -->
<!-- _paginate: false -->

# Constants as the bridge

## Minimal-shot autonomy, in motion

###### Snow-Underlay  ·  SoTA Commission I  ·  May 2026

---

<!-- _class: image-right -->

![bg right:55%](../outputs/video/boreas_2021_01_26/stills/overlay__t001p0.jpg)

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

![bg right:55%](../outputs/video/boreas_2021_01_26/stills/snow_naive_overlay__t005p0.jpg)

## Off-the-shelf segmentation, applied directly:

a confident red mask
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

## The recipe (per snow frame)

<div class="recipe">

1. &nbsp;Pull the **live snowy frame** from the plough's camera.
2. &nbsp;Pull a **clear-season prior** of the same coordinates (Boreas paired summer drives).
3. &nbsp;**Match** the two with a frozen feature matcher.
4. &nbsp;Estimate a **homography**, biased toward the ground plane.
5. &nbsp;Run a Cityscapes road segmenter — **on the clear prior only**.
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
| **Boreas** &nbsp;*(IJRR '23)* | Snow + paired summer captures | — (CC BY 4.0) |

Every learned component is **frozen**.
Snow appears only at inference, as the runtime input.

---

## In motion

A 15-second snow drive on a buried Toronto residential street.
Per snow frame: K=3 nearest summer priors, snow→summer match,
warp the segmenter's road mask back, fuse, EMA-smooth.

The cross-season principle survives motion.

###### outputs/video/boreas_2021_01_26/overlay.mp4 — 15 s, 1024×856, ~7 MB

---

<!-- _class: full-bleed -->

![bg fit](../outputs/video/boreas_2021_01_26/stills/overlay__t005p0.jpg)

---

## What we tried that didn't work

> A pair with **3–7×** more inliers can still produce a worse video.

**Synthetic priors** from past snow frames matched dramatically better in stills.
In motion, each frame's slightly-too-large mask seeded the next frame's prior;
the road overlay drifted outward into bushes within seconds. *Positive feedback loop.*

**Optical-flow propagation** between matched keyframes: vanishing-point flow
stretches the previous mask outward at every step. Same outcome, different mechanism.

###### EMA on the binary mask, α = 0.4, was what survived the motion test. Failure modes are evidence too.

---

## Minimal-shot integrity

| Claim | Status |
|-------|:------:|
| Zero snowy frames touch any model weights | ✓ |
| Zero snowy frames touch any annotation pipeline | ✓ |
| Snow appears only as runtime input | ✓ |
| Pretrained matcher · pretrained segmenter · classical RANSAC | ✓ |
| Reproducible from a clean clone with one command (`make reproduce`) | ✓ |

> The only handle we offered the model on the snow regime
> was the clear prior of the same place.

---

## One channel, not a snowplough

The output answers *where the road **should be***. **Not** *where to drive*.

A snow-covered car parked on the road would still sit inside the green overlay. The pipeline has no notion of obstacles, drivable surface, or 3D geometry — and that is the **scope**, not a bug.

This is **one channel** in a fuller stack. It feeds alongside lidar, depth estimation, and obstacle detection. It does not replace them.

> The contribution we are demonstrating is the *move*: how to extend a model from a data-rich regime into an adjacent data-poor one through a learned-invariant constant. **Snow on a road is the instance we built. The structure transfers.**

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
make reproduce
```

###### Read

`README.md` &nbsp;·&nbsp; `docs/writeup.md` (render with `make writeup`) &nbsp;·&nbsp; `docs/index.html` (Pages)

###### Submission

SoTA Commission I — Minimal-Shot Autonomy &nbsp;·&nbsp; May 2026 &nbsp;·&nbsp; Boreas dataset CC BY 4.0

---

<!-- ─────────────────────────────────────────────────────────────────────── -->
<!-- The slides above are the deck (rendered to slides.pdf via `make slides`). -->
<!-- The section below is the storyboard for the externally-edited submission -->
<!-- video. It's NOT a slide; Marp will render it as additional pages but the -->
<!-- editor reads it as plain markdown when planning the cut.                  -->
<!-- ─────────────────────────────────────────────────────────────────────── -->

# Appendix · submission-video plan

> **Title**: *Constants as the bridge — in motion.*
> **Length**: 90 s (target). Acceptable range 60–120 s.
> **Format**: 1080p MP4, H.264 / yuv420p, ≤ 50 MB.
> **Composed externally** (Premiere / Final Cut / DaVinci / etc.); this section is the editor's brief.

## Asset inventory

Every asset below is produced by a `make` command. No one-offs.

### Moving overlay clips (15 s each, 1024 × 856)

| Asset | Path | Produced by |
|---|---|---|
| **Canonical overlay** (snow + green road, headline) | `outputs/video/boreas_2021_01_26/overlay.mp4` | `make reproduce` |
| Snow input \| overlay (sidebyside) | `outputs/video/boreas_2021_01_26/sidebyside.mp4` | `make assets` |
| Snow \| naive (red) \| overlay (3-panel) | `outputs/video/boreas_2021_01_26/snow_naive_overlay.mp4` | `make assets` |
| Snow \| overlay \| naive (alt order) | `outputs/video/boreas_2021_01_26/snow_overlay_naive.mp4` | `make assets` |
| 4-panel quad (snow / summer+road / overlay / naive) | `outputs/video/boreas_2021_01_26/quad.mp4` | `make assets` |
| Robustness (same intersection, different snowfall): 2025-02-15 | `outputs/video/boreas_2025_02_15/overlay.mp4` | `make oracle TRACK=boreas_2025_02_15 && make reproduce-track TRACK=boreas_2025_02_15` |
| External scene (TBD — work in progress) | `outputs/video/<external_track>/overlay.mp4` | `make reproduce-track TRACK=<external_track>` |

### Stills

`outputs/video/<track>/stills/<layout>__t<NNNN>.jpg` — extracted at 1.0 s, 5.0 s, 10.0 s, 14.0 s of every rendered mp4. Produced by `make extract-stills TRACK=<id>` (auto-run by `make assets`).

### Static-stills precursor (B-roll for the principle)

`outputs/heroes/<pair_id>__panel.png` (4-column 2×2: snow / clear+mask / overlay / naive), `outputs/heroes/<pair_id>__matches.png` (correspondences), `outputs/heroes/<pair_id>__overlay.png` (snow with green road overlay), `outputs/heroes/<pair_id>__naive_baseline.png` (snow with red naive mask), and `outputs/audit/contact_sheet.png` (27-row sheet, one row per demo pair; in single-prior mode each row is snow / overlay / naive). Produced by `make stills` (single-prior, default; `make stills-multi` for the Phase J fusion ablation).

### Music

Bensound — *Slow Motion* (free with attribution). File at `_archive/assets/audio/music.mp3`. Source: <https://www.bensound.com/royalty-free-music>. Attribution: `Music: "Slow Motion" by Bensound · bensound.com`.

### Typography (if title cards added in the editor)

Fonts ship at `docs/_assets/fonts/` (OFL): **Inter Regular** (headlines / labels), **EB Garamond Regular / Italic** (body / attributions), **JetBrains Mono Regular** (code / file paths).

Palette: cream `#f6f3ee` · charcoal `#1c1c1c` · rust `#b34a25` · mute `#8a8780` · road green `#2e9c56` · naive red `#dc3c32`.

## Storyboard (10 beats, ≈ 90 s)

Music starts at beat 1 and runs continuously, fading out under beat 10.

1. **Hook** *(0:00–0:08)* — snow query still, slow zoom 100 → 110 %. *"Where is the road?"* + EB Garamond italic narration *"A snow plough's job is short. Keep the road clear. The catch — while it's doing it, the road is invisible."* Cut on first sustained piano note.

2. **The failure** *(0:08–0:16)* — naive-baseline still (red mask) hard-cut after the snow query, hold 4 s, then dissolve to `snow_naive_overlay.mp4` so the red drift is *seen*. *"Off-the-shelf segmentation, applied directly:"* Dip-to-white transition.

3. **27 million miles** *(0:16–0:24)* — cream card, large EB Garamond italic centred:
> 27 million miles of road. The long tail of conditions any of them can be in is *longer than the road itself.*

4. **The move** *(0:24–0:34)* — split-screen still: left half snow query, right half summer prior, from `quad__t005p0.jpg` (top-left and top-right). *"Same road. Last July. Match what stays the same."* Cross-dissolve.

5. **The recipe** *(0:34–0:42)* — cream card, six numbered Inter lines, animate each on after 0.6 s.

6. **THE REVEAL** *(0:42–0:58)* — full-bleed `overlay.mp4`, 1.0× from t=0 to t=15. Lower-third caption fades in t=2 s, out t=12 s: *"Cross-season road overlay. Snow, frame by frame. Nothing trained on snow."* Silence on the long shot — music carries the moment.

7. **Why it works** *(0:58–1:06)* — matches viz from a strong static pair (`outputs/heroes/<pair>__matches.png`), slow pan. *"The matcher anchors on what the season can't change: buildings, signs, rooflines, distant horizons."*

8. **Architecture** *(1:06–1:14)* — cream card, 4-row Inter table:
   - DISK · MegaDepth · no snow
   - LightGlue · MegaDepth · no snow
   - USAC-MAGSAC · classical RANSAC
   - Mask2Former · Cityscapes · no snow
   Footer: *"Every learned component frozen. Snow is the runtime input only."*

9. **Generalising** *(1:14–1:22)* — cream card, large EB Garamond italic centred:
> Where data is missing, find a regime where it isn't. Identify what is constant. Transfer through the constant.
   Narration cycles through three examples (low-light medical / polar / Mars).

10. **Sign-off** *(1:22–1:30)* — cream card, single Inter line: *"Constants as the bridge."* Subtitle with credits (project name, Boreas attribution, music attribution). Music fades over 6 s. Final 1 s on a clean cream frame.

## Music timing

- Bensound *Slow Motion* opens with a soft piano single note at ≈ 0:01. Start the clip there, not on the file's t=0.
- Natural breaths around 0:30 and 1:00 — align beat 4 → beat 5 cut to one of those breaths.
- Final fade-out: 6 s, starting at video t=1:24.

## Voice-over options

Narration above is written for *on-screen text only*. If recorded VO is preferred: narrate verbatim, calm and unhurried, ~2 wpm slower than conversational. Voice at -16 LUFS, music ducked to -22 LUFS under VO. If silent / text-only, slow each cream card by ~30 % so the reader has time.

## Required asset checklist

```bash
# ≤ 6 GB download + ~50 min on Mac CPU. Produces the headline.
make reproduce

# ≤ 5 min total — reuses caches.
make assets

# Robustness alt (same Glen Shields intersection, different snowfall).
# Always run the oracle first to verify priors exist + summer segmentation is good.
make oracle TRACK=boreas_2025_02_15
make reproduce-track TRACK=boreas_2025_02_15
make extract-stills TRACK=boreas_2025_02_15

# ≤ 30 min for static-stills precursor (Mapillary pulls + pipeline).
make stills

# (optional) Stage GitHub Pages assets at the same time:
make pages-assets
```

After all runs, verify:

- `outputs/video/boreas_2021_01_26/overlay.mp4` exists (~7 MB, canonical)
- `outputs/video/boreas_2021_01_26/{sidebyside,snow_naive_overlay,snow_overlay_naive,quad}.mp4` all exist
- `outputs/video/boreas_2021_01_26/stills/` has ≥ 20 JPEGs
- `outputs/video/boreas_2025_02_15/overlay.mp4` exists (robustness, oracle-verified window)
- `outputs/heroes/` has 27 panel PNGs (one per demo pair)
- `outputs/audit/contact_sheet.png` exists
- `_archive/assets/audio/music.mp3` is on disk (if title cards desired)

## Editor notes

- **Don't add hard cuts on busy frames of the overlay clip.** The road overlay's mask shifts subtly between frames; cuts mid-overlay-jiggle look like editing artefacts. Cut only on transitions between beats.
- **The naive baseline (red) is intentionally ugly.** Don't smooth it. Its job is to demonstrate the wrong answer in red against the right answer in green.
- **The cream card colour is precise** — `#f6f3ee`. Most editing software defaults to pure white; use the hex.
- **Run-time tweak**: if the overlay clip's first or last second looks weak in motion, trim it. The clip is rendered at full 15 s but 12–13 s usable is fine.
- **Closing card lingers.** Give the *Constants as the bridge* line at least 4 s on screen. The whole video earns that line; don't rush past it.
