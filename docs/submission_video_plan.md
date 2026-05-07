---
title: "Snow-Underlay — submission video plan"
subtitle: "Beat-by-beat shot list, narration script, asset paths"
target: "90–120 s, 1080p, H.264 / yuv420p, ≤ 50 MB"
---

> **Submission**: SoTA Commission I — Minimal-Shot Autonomy &nbsp;·&nbsp; Deadline 2026-05-10
> **Composed externally** (Premiere / Final Cut / DaVinci / etc.); this document is the editor's brief.

---

## Constraints from the brief

- Length: 1–5 minutes; **target 90–120 s** (Devpost winning patterns favour 90 s).
- Must include **at least one failure case** shown visually.
- Show approach + system operation; the brief explicitly welcomes "hacky unfinished projects" alongside polished ones.
- Judging criteria: **technical excellence**, **novelty**, **feasibility**, **adherence to brief**.

The video has one job: convince a judge in the first 15 seconds to keep watching, then deliver the principle and a defensible result before they drift.

---

## Story arc (7 beats)

The arc moves: **problem (failure) → principle → reveal (success) → honesty (failure 2) → why it works → generality → sign-off**. The two failure beats earn the success beat in between them.

| # | Beat | Time | Length |
|---|---|---|---|
| 1 | The failure that motivates everything | 0:00–0:15 | 15 s |
| 2 | The principle (cream card) | 0:15–0:30 | 15 s |
| 3 | The reveal (canonical overlay) | 0:30–0:55 | 25 s |
| 4 | A failure case (mandatory per brief) | 0:55–1:10 | 15 s |
| 5 | Why it works (matches viz) | 1:10–1:25 | 15 s |
| 6 | Generalising (three regimes) | 1:25–1:45 | 20 s |
| 7 | Sign-off | 1:45–2:00 | 15 s |

Total: **120 s**. If the cut runs long, beat 6 contracts first (drop one regime), then beat 5.

---

## Beat 1 · The failure that motivates everything (0:00–0:15)

**Visual.** Hard-cut from black to a snow query frame. Hold for 1 s on the raw snow frame so the viewer registers what the camera sees. Cross-dissolve into `outputs/video/boreas_2021_01_26/snow_naive_overlay.mp4` from t=2 s, played in real time. The naive (red) mask appears on the right panel, painting confidently across buildings, sky, road — everything.

**Narration (text on screen, EB Garamond italic, lower-third).**

> *Off-the-shelf perception, applied directly to a snow plough's camera, says the entire scene is sky.*

**Cuts.** No cuts inside this beat. Hold the contrast.

**Why this hooks.** A self-driving stack failing visibly + confidently is the perfect 15-second hook. Devpost-style winning patterns: lead with the *problem*, not the architecture. The judge sees the long-tail failure mode the brief was written to address, before they hear our name.

**Assets.**

| Asset | Path | Notes |
|---|---|---|
| Snow query opener | `outputs/video/boreas_2021_01_26/stills/overlay__t005p0.jpg` | crop the green overlay out of frame for the first 1 s if your editor allows; or use `outputs/heroes/<one_great_pair>__snow.png` |
| Naive failure clip | `outputs/video/boreas_2021_01_26/snow_naive_overlay.mp4` | 3-panel layout — show only the *naive* panel by zooming in, or use the snow + naive frames extracted from `stills/snow_naive_overlay__t*.jpg` |

---

## Beat 2 · The principle (0:15–0:30)

**Visual.** Cream card (`#f6f3ee`). Centred EB Garamond italic. Cross-dissolve in over 0.5 s.

**Text on screen** (each line on its own beat, ≈ 1.5 s apart):

> Minimal-shot autonomy needs *generalisation*, not *memorisation*.
>
> For almost every regime where autonomy fails for lack of data,
> there is an *adjacent* regime where data exists,
> and where the parts that matter are *the same*.

**Voice-over (optional).** If recorded: calm, unhurried. Same lines.

**Cut.** Hard cut to beat 3 on a piano breath if music is present.

**Assets.** Cream card, no media.

---

## Beat 3 · The reveal (0:30–0:55)

**Visual.** Full-bleed `outputs/video/boreas_2021_01_26/overlay.mp4`, played from t=2 s to t=15 s (the cleanest stretch). The green road overlay is the only visual; no crop, no labels.

**Hold the silence on screen for the first 5 s.** Let the green mask track the buried road silently — the judge realises what they're watching.

**Text on screen** (lower-third, fades in at 0:38, out at 0:50):

> *Cross-season road overlay. Snow, frame by frame. Nothing trained on snow.*

**This is the moment the video earns.** No music swell, no graphic flourish; the tracking is the demonstration.

**Assets.**

| Asset | Path |
|---|---|
| Canonical overlay | `outputs/video/boreas_2021_01_26/overlay.mp4` |

---

## Beat 4 · A failure case (0:55–1:10) — *mandatory per brief*

**Pick one** of these; whichever is visually clearest after rendering:

**Option A — boreas_2025_02_15 jumpy frames (recommended).** A second snowfall on the same Toronto loop; the segmentation is wider on this window and 1-prior frames produce mask jumps. Show 6–8 seconds of the unsmoothed clip from `outputs/video/boreas_2025_02_15/overlay.mp4` over a stretch where you can see a mask jump.

**Option B — synthetic-priors drift.** If you re-run with `--synthetic-priors 2` (the rejected experiment), the road overlay drifts outward over 5–10 s as the feedback loop kicks in. Bigger visual but requires a separate render.

**Option C — naive-only cuts back in.** Re-show 3 seconds of `snow_naive_overlay.mp4` after the reveal as a contrast — "this is what was happening the whole time without the bridge". Simpler, but already used in beat 1.

**Narration (text on screen).**

> *The pipeline does not always succeed. Here, the matcher's assumptions break — and we know to detect it before we ship a frame.*

**Why this beat matters.** The brief explicitly asks for failure cases. Showing one *honestly* is a winning-pattern signal, not a weakness. It also frames the window oracle (`make oracle`) as an explicit answer: "here is how we detect this case before any cache build".

**Assets.**

| Asset | Path |
|---|---|
| (Option A) Jumpy clip | `outputs/video/boreas_2025_02_15/overlay.mp4` (6–8 s slice from a known jumpy stretch) |
| (Option B) Drift clip | run `--synthetic-priors 2` over a 50-frame window; render with the existing pipeline; capture |
| (Option C) Naive baseline | `outputs/video/boreas_2021_01_26/snow_naive_overlay.mp4` (reuse) |

---

## Beat 5 · Why it works (1:10–1:25)

**Visual.** `outputs/video/boreas_2021_01_26/matches.mp4` — snow + best summer prior with a small subset of correspondence lines (green) flickering between gate posts, fence wires, masonry corners. Play in real time.

**Narration (text on screen).**

> *The matcher anchors on what the season has not changed.*

**Optional layer.** A subtle rectangle highlights one matched feature pair (a gatepost, say), held for 2 s, then dissolves.

**Assets.**

| Asset | Path |
|---|---|
| Matches viz | `outputs/video/boreas_2021_01_26/matches.mp4` |
| (Optional) static still | `outputs/heroes/<best_pair>__matches.png` |

---

## Beat 6 · Generalising (1:25–1:45)

**Visual.** Three quick beats, each ≈ 5 s. Each is a cream card with a still + one line of text. Cross-dissolves between.

**6a — Polar Earth observation** (1:25–1:30).
- Background still: a satellite-imagery polar still (any open-license polar EO image works; if no time, a stylised globe + ice graphic in the cream colour palette).
- Text: *Polar Earth observation. The orbital schedule is the constant.*

**6b — Low-light medical imaging** (1:30–1:35).
- Background still: a fundus / retinal / endoscopy still (Wikipedia public-domain endoscopy is fine).
- Text: *Low-light medical imaging. The patient's anatomy is the constant.*

**6c — Off-Earth manipulation** (1:35–1:40).
- Background still: a Mars rover or robot manipulator still (NASA public-domain).
- Text: *Off-Earth manipulation. The robot's geometry is the constant.*

**Closing line** (1:40–1:45):

> *Find what stays the same. Walk across.*

**Why this beat matters.** This is what positions the contribution as a *primitive*, not "a snow-plough thing". Three regimes that are clearly not snow ploughs, each with the same structural shape — that earns the architectural framing.

**Assets.** External public-domain stills (NASA / Wikipedia / NOAA). If under deadline pressure, use solid-colour cream cards with the text only — the framing carries.

---

## Beat 7 · Sign-off (1:45–2:00)

**Visual.** Cream card. Centred large EB Garamond italic.

> *Constants as the bridge.*

Below, smaller (Inter regular):

```
github.com/aturner22/snowseer
```

**Closing card** holds for 4 s. Music fades over the final 6 s.

---

## Asset inventory (single source of truth)

Every asset below is produced by a `make` command on a clean clone. No one-offs.

### Moving clips (all 1024 × 856, 10 fps)

| Clip | Path | Produced by |
|---|---|---|
| Canonical overlay (the headline) | `outputs/video/boreas_2021_01_26/overlay.mp4` | `make reproduce` |
| Canonical sidebyside | `outputs/video/boreas_2021_01_26/sidebyside.mp4` | `make track TRACK=boreas_2021_01_26` |
| Canonical 3-panel (snow / naive / overlay) | `outputs/video/boreas_2021_01_26/snow_naive_overlay.mp4` | same |
| Canonical 3-panel (snow / overlay / naive) | `outputs/video/boreas_2021_01_26/snow_overlay_naive.mp4` | same |
| Canonical quad | `outputs/video/boreas_2021_01_26/quad.mp4` | same |
| Canonical matches | `outputs/video/boreas_2021_01_26/matches.mp4` | same |
| Robustness alt | `outputs/video/boreas_2025_02_15/overlay.mp4` | `make track TRACK=boreas_2025_02_15` |
| (TBD) Different-scene clip | `outputs/video/<id>/overlay.mp4` | `make track TRACK=<id>` |

### Stills

`outputs/video/<track>/stills/<layout>__t<NNNN>.jpg` — extracted at 1.0 s, 5.0 s, 10.0 s, 14.0 s of every rendered clip. Produced by `make track TRACK=<id>` (auto-runs `extract_assets`).

### Static-stills precursor (B-roll for the principle)

`outputs/heroes/<pair_id>__<layout>.png` — 27 demo pairs × 15 layouts each. Produced by `make stills`. Useful for beat 6 backgrounds if you prefer in-pipeline imagery to external stills.

### Music

Bensound *Slow Motion* (free with attribution). File at `_archive/assets/audio/music.mp3`. Source: <https://www.bensound.com/royalty-free-music>. Attribution line for closing card: *Music: "Slow Motion" by Bensound · bensound.com*.

### Typography (for cream cards)

Fonts ship at `docs/_assets/fonts/` (OFL): **Inter Regular** (headlines / labels), **EB Garamond Regular / Italic** (body / quotes), **JetBrains Mono Regular** (URLs / paths).

Palette: cream `#f6f3ee` · charcoal `#1c1c1c` · rust `#b34a25` · mute `#8a8780` · road green `#2e9c56` · naive red `#dc3c32`.

---

## Music cue points

Bensound *Slow Motion* is roughly 2:30 long; the first 2:00 lines up with the video.

- **0:00**: music starts (skip the file's silent t=0–t=1, start on the first piano note).
- **0:30**: a natural breath in the music — align the cut from beat 2 (cream card) → beat 3 (reveal) here.
- **1:00**: another breath — align beat 4 (failure) → beat 5 (matches viz) here.
- **1:54**: begin 6-s music fade-out.

If recorded VO is added: voice at -16 LUFS, music ducked to -22 LUFS under VO.

---

## Editor notes (read before the first cut)

- **Don't add hard cuts on busy frames of the canonical overlay clip.** The road mask shifts subtly between frames; cuts mid-jiggle look like editing artefacts. Cut only on transitions between beats.
- **The naive (red) baseline is intentionally ugly.** Don't smooth it. Its job is to demonstrate the wrong answer in red against the right answer in green.
- **Cream is `#f6f3ee` exactly.** Most editing software defaults to pure white; use the hex.
- **Hero clip can run shorter.** If the canonical clip's first or last second looks weak, trim. 12–13 s of usable clip is fine for beat 3.
- **Closing card lingers.** Give the *Constants as the bridge* line at least 4 s on screen. The whole video earns that line.
- **Resolution check.** Boreas mp4s are 1024 × 856. If exporting to 1920 × 1080, letterbox or crop — don't upscale (it artefacts on the green overlay's anti-aliased edges).

---

## Pre-flight asset checklist

```bash
# Once these all exist on disk, the editor has everything they need.
# All paths assume `cd /path/to/snowseer` first.

ls outputs/video/boreas_2021_01_26/{overlay,sidebyside,snow_naive_overlay,snow_overlay_naive,quad,matches}.mp4
ls outputs/video/boreas_2021_01_26/stills/*.jpg
ls outputs/video/boreas_2025_02_15/overlay.mp4
ls outputs/heroes/*__panel.png        # static-stills B-roll
ls _archive/assets/audio/music.mp3    # if title cards desired
ls docs/_assets/fonts/*.ttf           # Inter, EB Garamond, JetBrains Mono
```

If anything is missing, run:

```bash
make track TRACK=boreas_2021_01_26    # canonical bundle
make track TRACK=boreas_2025_02_15    # robustness alt
make stills                           # static-stills B-roll
```

---

## What this video deliberately doesn't do

- **No hyped intro.** No "what if" rhetorical question, no "imagine you're a snow plough" framing. The brief readers are technical; the failure beat (beat 1) does the hook job in 15 s without rhetoric.
- **No leaderboard numbers.** This is a qualitative project; quoted percentages would be cherry-picked. The cross-season pipeline tracks the road; the naive baseline doesn't. Show, don't tabulate.
- **No live talking head.** Calm voice-over (optional) or text-only; no on-camera presenter. The work is the visual.
- **No Bensound stinger or sound effects** beyond the licensed track. Restraint reads as confidence.
