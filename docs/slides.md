# Submission video — script, storyboard, and asset inventory

> **Title**: *Constants as the bridge — in motion.*
> **Length**: 90 seconds (target). Acceptable range: 60–120 s.
> **Audience**: SoTA Commission I judges + general technical reader.
> **Format**: 1080p MP4, H.264 / yuv420p, ≤ 50 MB.
> **Composed externally** (Premiere / Final Cut / DaVinci / Capcut / etc.); this file is the editor's brief.

---

## 1 · Thesis (the line everything serves)

> *We don't have to learn what a road looks like under snow. The road's appearance has changed; its position in space has not. Match what stays the same, and the rest comes for free.*

Every beat below points back to that line. If a clip doesn't earn it, cut.

---

## 2 · Asset inventory (paths)

Every asset below is produced by a `make` command in this repo. No one-offs.

### 2.1 — Moving overlay clips (15 s each, 1024 × 856)

| Asset | Path | Produced by |
|---|---|---|
| **Canonical overlay** (snow + green road overlay, headline visual) | `outputs/video/boreas_2021_01_26/overlay.mp4` | `make reproduce` |
| Snow input \| overlay (sidebyside) | `outputs/video/boreas_2021_01_26/sidebyside.mp4` | `make assets` |
| Snow \| naive (red) \| overlay (3-panel) | `outputs/video/boreas_2021_01_26/snow_naive_overlay.mp4` | `make assets` |
| Snow \| overlay \| naive (alt order) | `outputs/video/boreas_2021_01_26/snow_overlay_naive.mp4` | `make assets` |
| Snow \| summer prior + road \| overlay \| naive (quad, 2×2) | `outputs/video/boreas_2021_01_26/quad.mp4` | `make assets` |
| Alt: 2024-12-23 dusk residential | `outputs/video/boreas_2024_12_23/overlay.mp4` | `make reproduce-track TRACK=boreas_2024_12_23` |
| Alt: 2025-02-15 active snowfall | `outputs/video/boreas_2025_02_15/overlay.mp4` | `make reproduce-track TRACK=boreas_2025_02_15` |

### 2.2 — Stills (extracted at t = 1.0 s, 5.0 s, 10.0 s, 14.0 s of each clip)

| Asset directory | Produced by |
|---|---|
| `outputs/video/<track>/stills/<layout>__t<NNNN>.jpg` | `make extract-stills TRACK=<id>` (auto-run by `make assets`) |

Useful as: title-card backgrounds, B-roll, before/after stills, thumbnail.

### 2.3 — Static-stills precursor (the cross-season principle on individual pairs)

These are the v1.x demo's hero panels. Use as supporting B-roll to establish the principle before the moving-overlay reveal.

| Asset | Path | Produced by |
|---|---|---|
| Per-pair 4-panel figure (snow / clear+mask / overlay / naive) | `outputs/heroes/<pair_id>__panel.png` | `make stills` |
| Per-pair single overlay (snow + green) | `outputs/heroes/<pair_id>__overlay.png` | `make stills` |
| Per-pair feature-correspondence viz (matches.png) | `outputs/heroes/<pair_id>__matches.png` | `make stills` |
| 14-row contact sheet (one row per hero) | `outputs/audit/contact_sheet.png` | `make stills` |

Recommended hero pair to feature: `gallivare_se__1113124103239974__202392698419785` (highest-inlier GREAT pair, snow-banked road with a parking-restriction sign).

### 2.4 — Reference / supporting visuals

- The plough's *failure* shot (snow + red naive baseline) lives at `outputs/heroes/<id>__naive_baseline.png` — use as the "this is what fails" beat.
- Boreas vehicle / route information for credits: `data/video/tracks/boreas_2021_01_26/track.json` (carries the CC BY 4.0 attribution string).

### 2.5 — Music

Bensound — *Slow Motion* (free with attribution).
File on disk (kept for editor convenience, not part of the canonical pipeline): `_archive/assets/audio/music.mp3`.
Source / re-download: <https://www.bensound.com/royalty-free-music>
Attribution to bake into end card: `Music: "Slow Motion" by Bensound — bensound.com`.

### 2.6 — Typography (if title cards are added in editor)

Fonts ship with the repo at `assets/fonts/` (OFL-licensed):
- **Inter Regular** — headlines, captions, on-screen labels
- **EB Garamond Regular / Italic** — body text, attributions
- **JetBrains Mono Regular** — code / file-path overlays if used

Palette (in case the editor wants to match the repo's visual identity, see `docs/style/style.md`):
- Background `#f6f3ee` (warm cream)
- Body text `#1c1c1c` (charcoal)
- Accent `#b34a25` (rust)
- Mute `#8a8780` (warm grey)
- Road overlay (green) `#2e9c56`
- Naive failure (red) `#dc3c32`

---

## 3 · Storyboard (10 beats, ≈ 90 s)

Each beat is one cut. Music starts at beat 1 and runs continuously, fading out under beat 10.

### Beat 1 — Hook *(0:00 – 0:08)*

- **Visual**: still from the canonical snow query (no overlay yet) — `outputs/video/boreas_2021_01_26/stills/overlay__t01p0.jpg`. Hold for 2 s, then slow zoom in to ~110 % over 6 s.
- **On-screen text** (Inter, large, centre): "Where is the road?"
- **Narration / text card** (EB Garamond italic, 1.5 s after the headline):
  > *A snow plough's job is short. Keep the road clear. The catch — while it's doing it, the road is invisible.*
- **Transition**: cut on the music's first sustained piano note.

### Beat 2 — The failure *(0:08 – 0:16)*

- **Visual**: snow + red-naive baseline still (`outputs/heroes/<one of the heroes>__naive_baseline.png`) hard-cut after the snow query. Hold 4 s, then dissolve to the moving 3-panel `snow_naive_overlay.mp4` for the second 4 s (so the red naive prediction is *seen drifting* as the camera moves).
- **On-screen text**: "Off-the-shelf segmentation, applied directly:"
- **Narration**: *The model's confident answer is red — wrong. It hasn't seen snow.*
- **Transition**: dip to white (1 frame) on the cut.

### Beat 3 — The 27 million miles *(0:16 – 0:24)*

- **Visual**: cream background card. Large EB Garamond italic text (centred):
  > 27 million miles of road.
  > The long tail of conditions any of them can be in
  > is *longer than the road itself.*
- **Narration**: silence, let the type breathe.
- **Transition**: subtle horizontal-line wipe, rust accent.

### Beat 4 — The move *(0:24 – 0:34)*

- **Visual**: split-screen still — left half snow query, right half summer prior, both 50 % width. Use frames at the same UTM coordinate from `outputs/video/boreas_2021_01_26/stills/quad__t05p0.jpg` (top-left = snow, top-right = summer).
- **On-screen text**: "Same road. Last July. Match what stays the same."
- **Narration**: *The road's appearance changed. Its position in space did not. Buildings, signs, rooflines — they're still where they always were.*
- **Transition**: cross-dissolve.

### Beat 5 — The recipe *(0:34 – 0:42)*

- **Visual**: cream card again. Six numbered lines, Inter, large. Animate each line on after 0.6 s.
  1. Live snowy frame.
  2. Clear-season prior of the same coordinates.
  3. Match the two with a frozen feature matcher.
  4. Fit a homography, biased toward the ground plane.
  5. Segment the road on the *clear* prior only.
  6. Warp the mask onto the snowy frame.
- **Narration**: *Six steps. Nothing in the pipeline has been trained on snow.*
- **Transition**: rust accent line wipes to next.

### Beat 6 — THE REVEAL *(0:42 – 0:58)*

- **Visual**: full-bleed `outputs/video/boreas_2021_01_26/overlay.mp4`, played 1.0× from t=0 to t=15. **This is the headline shot.** Hold the entire clip.
- **On-screen text**: lower-third caption that fades in at t=2 s of clip and out at t=12 s:
  > *Cross-season road overlay. Snow, frame by frame. Nothing trained on snow.*
- **Narration**: silence on the long shot. Music carries the moment.
- **Transition**: fade to next on the last frame of the overlay clip.

### Beat 7 — Why it works *(0:58 – 1:06)*

- **Visual**: the matches viz from a strong static pair, e.g. `outputs/heroes/<pair>__matches.png`. Hold 2 s; pan slowly across.
- **On-screen text**: "The matcher anchors on what the season can't change: \n buildings, signs, rooflines, distant horizons."
- **Narration**: *A frozen feature matcher. A frozen segmenter. Classical RANSAC. The composition is the contribution.*
- **Transition**: cut.

### Beat 8 — Architecture (the receipts) *(1:06 – 1:14)*

- **Visual**: cream card with a 4-row table, Inter:
  - DISK · MegaDepth · no snow
  - LightGlue · MegaDepth · no snow
  - USAC-MAGSAC · classical RANSAC
  - Mask2Former · Cityscapes · no snow
- **On-screen footer**: "Every learned component frozen. Snow is the runtime input only."
- **Narration**: silence.
- **Transition**: dissolve.

### Beat 9 — Generalising *(1:14 – 1:22)*

- **Visual**: cream card with the closing line, large EB Garamond italic, centred:
  > *Where data is missing,*
  > *find a regime where it isn't.*
  > *Identify what is constant.*
  > *Transfer through the constant.*
- **Narration**: *Snow on a road is one example. Polar imaging without polar training data. Low-light medical imaging without low-light training data. A manipulator on Mars without Mars training data. Each admits the same structure.*
- **Transition**: gentle fade.

### Beat 10 — Sign-off *(1:22 – 1:30)*

- **Visual**: cream card. Single line, Inter, centred:
  > *Constants as the bridge.*
- **Subtitle / credits** (EB Garamond italic, smaller, lower):
  > Snow-Underlay · SoTA Commission I · May 2026
  > Boreas dataset (UTIAS-ASRL, CC BY 4.0)
  > Music: "Slow Motion" by Bensound · bensound.com
- **Music**: fade out across these 8 seconds.
- **Final hold**: 1 s on a fully-cream frame before end.

---

## 4 · Music timing notes

- Bensound *Slow Motion* opens with a soft piano single note at ≈ 0:01 — start the clip there, not on the file's t=0.
- The piece has natural breaths around 0:30 and 1:00 — align beat 4 → beat 5 cut to one of those breaths.
- Final fade-out: 6 s, starting at video t=1:24.

---

## 5 · Voice-over options

The narration above is written for *on-screen text only* — no recorded VO required. If a voice-over is preferred, narrate verbatim from the storyboard (calm, unhurried, about 2 wpm slower than conversational speed). Voice should sit at -16 LUFS, music ducked to -22 LUFS under the VO.

If silent / text-only is preferred, slow each cream card by ~30 % so the reader has time.

---

## 6 · Required asset checklist (run before opening the editor)

```bash
# ≤ 6 GB download + ~50 min on Mac CPU. Produces the headline.
make reproduce

# ≤ 5 min per layout × 4 extra layouts + ≤ 5 min stills extraction.
make assets

# ≤ 30 min compute total for both alts (sequential — never parallel on ≤ 16 GB).
make reproduce-track TRACK=boreas_2024_12_23
make reproduce-track TRACK=boreas_2025_02_15
make extract-stills TRACK=boreas_2024_12_23
make extract-stills TRACK=boreas_2025_02_15

# ≤ 30 min for static-stills precursor (Mapillary pulls + pipeline).
make stills

# (Optional) Stage GitHub Pages assets at the same time:
make pages-assets
```

After all four runs:
- [ ] `outputs/video/boreas_2021_01_26/overlay.mp4` exists and is ~7 MB
- [ ] `outputs/video/boreas_2021_01_26/{sidebyside,snow_naive_overlay,snow_overlay_naive,quad}.mp4` all exist
- [ ] `outputs/video/boreas_2021_01_26/stills/` has ≥ 20 JPEGs
- [ ] `outputs/video/boreas_2024_12_23/overlay.mp4` and `outputs/video/boreas_2025_02_15/overlay.mp4` exist (these are the alts)
- [ ] `outputs/heroes/` has 14 panel PNGs
- [ ] `outputs/audit/contact_sheet.png` exists
- [ ] `_archive/assets/audio/music.mp3` is on disk (the music bed)

If anything is missing, the corresponding `make` command will (re)produce it deterministically.

---

## 7 · Notes for the editor

- **Don't add hard cuts on busy frames of the overlay clip.** The road overlay's mask shifts subtly between frames; cuts mid-overlay-jiggle look like editing artefacts. Cut only on transitions between beats.
- **The naive baseline (red) is intentionally ugly.** Don't smooth it. Its job is to demonstrate the wrong answer in red, set against the right answer in green.
- **The cream card colour is precise** — `#f6f3ee`. Most editing software defaults to pure white (`#ffffff`); use the hex.
- **Run-time tweak**: if the overlay clip's first or last second looks weak in motion, trim it. The clip is rendered at full 15 s but 12–13 s usable is fine.
- **Closing card lingers** — give the *Constants as the bridge* line at least 4 s on screen. The whole video earns that line; don't rush past it.
