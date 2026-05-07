# Changelog

All notable changes to Snow-Underlay are recorded here. Format inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows the project's `vN-suffix` tagging convention rather than semver.

## [Unreleased] — `video` branch (Phase L close-out)

> Working towards: SoTA Commission I — Minimal-Shot Autonomy submission, deadline 2026-05-10.

### Pre-flight oracle + alt-track recovery (2026-05-07 morning)

- **`src/video_runtime/window_oracle.py`** — pre-flight "is this demo-able?" check. Per snow frame: KD-tree query for K nearest summer poses (rejected if > distance threshold) + Mask2Former coverage check on each candidate prior (rejected if foreground road coverage < threshold). Output: per-frame report + longest contiguous demo-able run. Two modes:
  - default: full segmentation oracle on the loaded snow window
  - `--poses-only`: lite mode reading the FULL parent `camera_poses.csv` (pre-windowing); useful for re-windowing decisions before fetching frames.
- `make oracle TRACK=<id>` Makefile target. Hard rule: **never run `make reproduce-track` without a satisfied oracle first**.
- `src/video_runtime/fetch_track.py` gains `--snow-start-s/--snow-end-s` overrides so an oracle-chosen window can be fetched without editing the TRACKS registry.
- `src/video_runtime/pipeline_v.run_track` now honours `start/end/stride` when loading from cache (slices `cached_results` before smoothing). The cache holds the full original matching pass; callers wanting a sub-range get exactly that. Existing callers passing `start=0, end=None, stride=1` are unaffected.
- **`boreas_2024_12_23` retired** — same Glen Shields loop as canonical, plus the original window picked snow indices outside summer trajectory's spatial coverage (~75 % had zero priors possible by construction). Local cache + mp4s archived under `_archive/outputs/video/boreas_2024_12_23/` for inspection. Pages assets dropped.
- **`boreas_2025_02_15` re-windowed** via the pose-only oracle. New snow indices `5000..5350` (time 500–535 s in the snow sequence). Full segmentation oracle reports 100 % demo-able with score 1.112 — higher than canonical's 0.565. Robustness clip: same Glen Shields intersection as canonical, different snowfall and time-of-day. Cache build in flight.
- **L.6.C Mapillary external-scene scripts archived** to `_archive/data/` as work-in-progress: `find_snow_for_video.py` (broader recon with per-frame summer-prior probe), `fetch_mapillary_video.py` (Mapillary→Boreas-track bridge), `preview_candidates.py` (sample-thumbnail audit + HTML preview). The Tromsø candidate was a side-of-bus camera view (caught in user audit before any compute committed). Restoring once Boreas alt is shipping.

### Repo tidying (2026-05-07)

- **Comprehensive archive sweep** — moved `outputs/audit/`, `outputs/heroes/`, retired alt tracks, Phase K ablation videos (`_ablation/`, `_v1_canonical_K3/`), old K=5 / stride-3 caches, the Finnish winter dataset (4.7 GB), `_match_test/`, dev preview scripts to `_archive/`. Working tree dropped from 6.2 GB to 1.0 GB under `outputs/`.
- `demo/` Streamlit scripts archived (one-shot data curation tools, not on canonical pipeline). `make stream` removed.
- `.github/workflows/lint.yml` removed — CI was generating failure-email noise. `make test` (smoke tests) retained for local use.
- Standing directory-tree audit checklist added to plan §L.7.bis.
- `make tidy` Makefile target — reproducible cleanup of `__pycache__/`, `.DS_Store`, `.ruff_cache/`, scratch logs.

### Added

- Single-prior mode for the static-stills pipeline (`src/pipeline.py`). New `--max-priors` CLI flag (default `1` = v1 narrative; `5` = Phase J multi-prior fusion ablation; `0` = unlimited). The K = 1 path skips the three fusion-variant overlays and the priors strip — only `__matches.png`, `__naive_baseline.png`, `__overlay.png`, `__panel.png` are written.
- `make stills-multi` Makefile target wraps the K = 5 fusion ablation. `make stills` remains the default v1 narrative.
- `src/audit.py` now detects single-prior mode per pair (multi-prior fusion outputs absent → falls back to a 3-column `snow / overlay / naive` row instead of the 5-column multi-prior layout).
- `pipeline_v` (video runtime) gains observability and resume:
  - Per-frame `print(..., flush=True)` so long matching passes are visible in real time.
  - Atomic checkpoint to `_cache_<tag>.partial.pkl` every 50 processed frames (via `os.replace` of a `.tmp`).
  - Resume on restart: reads the partial cache and skips frames already done.
  - ETA log every 10 processed frames.
- `make assets` target — render-all-layouts + extract-stills for the canonical track in one command.
- `make extract-stills TRACK=<id>` target — extracts JPEGs at preset timestamps (1.0, 5.0, 10.0, 14.0 s) from every mp4 under `outputs/video/<id>/`.
- `make pages-assets` target — copies finalised stills (and locally-only mp4s) into `docs/_assets/<track>/` for the GitHub Pages site.
- `make reproduce-everything` master target — `reproduce` + alt tracks + assets + stills + writeup PDF + Pages assets.
- `src/video_runtime/extract_assets.py` module — ffmpeg-driven still extraction at preset timestamps.
- GitHub Pages site at `docs/index.html` + `docs/style/site.css`. Charcoal / cream / rust palette, EB Garamond + Inter via Google Fonts CDN. Sections: hero, principle, pipeline, architecture, watch-it-work (3 tracks), negative findings, static-stills precursor, integrity table, honest limits, reproduce, generalising. mp4s use poster-image placeholders (the actual mp4s remain gitignored — Pages renders the t = 5 s frame instead).
- "Scope of the contribution" framing in README, writeup, Pages, and slides: this is *one channel* of a fuller stack, answering *where the road should be*, not *where to drive*. The contribution is the **move** (transferring knowledge across regimes via a learned-invariant constant), not a turnkey snowplough perception system.
- Plan: `~/.claude/plans/i-have-an-idea-bubbly-haven.md` consolidated.
- Memory entries (`~/.claude/projects/.../memory/`):
  - `feedback_phase_l_long_runs.md` — flush + checkpoint discipline for long passes
  - `feedback_phase_l_git_hygiene.md` — no large or binary files in git, period

### Changed

- `.gitignore` tightened defensively: blanket `**/*.{mp4,mov,zip,tar,tar.gz,7z,pkl,parquet}`; `data/video/`, `outputs/video/`, `docs/*.pdf`, `data/**/*.csv`, `outputs/**/*.csv` all excluded. The repo now contains source only.
- README — new "video extension" lead, repo-layout fixes (compose_final.py + audio archived), pair-count clarification (27 reviewed, 14 GREAT+OKAY headline + 13 review-pool).
- `docs/slides.md` — Marp deck rewritten for visual rhythm (16 slides, alternating text and full-bleed imagery) plus a markdown video-plan appendix at the end (storyboard for the externally-edited submission video, not a slide deck).
- `docs/writeup.md` — essay updated for the video extension; new "What we extended (and what we tried that didn't work)" section documenting synth-prior + optical-flow rejections.
- `docs/index.html` — embedded `<video>` elements replaced with `<img>` poster JPEGs (per project policy: no large binaries in git). Local reproduction via `make reproduce` produces the actual mp4s.
- `render_all_layouts.py` — drops the cache-tag suffix from output filenames (stable layout names: `overlay.mp4`, `sidebyside.mp4`, etc., one per layout per track).

### Removed (untracked from index)

- `docs/{slides,writeup}.pdf` — render locally with `make pdfs`.
- `outputs/video/_match_test/*.png` (~8 MB total) — debug renders from K.1.
- `data/video/recon/*.csv`, `data/video/recon/summary.json`, `_thumbs/*.jpg` — Mapillary reconnaissance outputs.
- `data/video/tracks/**/{calib/, camera_poses.csv, window.json, track.json}` (~50 MB total) — fetched Boreas track metadata; regenerated by `make video-fetch TRACK=<id>`.

### Archived (`_archive/`, gitignored)

- `notebooks/02_video_walkthrough.ipynb` — narrative redundant with the writeup essay + Pages site.
- `src/video.py`, `src/video_runtime/compose_final.py`, `assets/audio/music.mp3` — submission-video composition; user composes externally.
- `data/curated_pairs.v1.json`, schema-migration scripts — Phase J multi-prior schema break.
- Earlier walkthrough notebook (Phase H), `outputs/demo.mp4`, `docs/audit.md`.

### Remote

- `origin` → `https://github.com/aturner22/snowseer.git` (added 2026-05-06; was previously empty). `video` branch pushed; subsequent commits push as they land.

---

## [v1.2-multi-prior-experiment] — 2026-05-04

### Added

- Multi-prior fetcher: K nearest summer captures per snow pair, fused via three strategies (union with edge-erosion, weighted soft-average by inlier count, hard majority vote).
- `src/fuse.py` — fusion strategies + foreground crop.
- Streamlit fusion-ablation curator (`demo/curate_results.py` — now archived).
- Audit contact sheet 5-column per-pair layout + per-prior strip.

### Decision

After empirical review the gain was real but small (a couple of pairs promoted, a couple regressed) at substantial narrative cost. Branch retained as future direction; `main` rolled back to v1.1-single-prior.

---

## [v1.1-single-prior] — 2026-05-03

### Added

- 4-column hero panel (snow / clear+mask / overlay / naive) with title + subtitle.
- Naive baseline (Cityscapes segmenter applied directly to snow) for visual comparison.
- IoU metrics (overlay vs naive, overlay vs identity).

### Tagged baseline

The v1 narrative referenced throughout the v2 video extension's "static-stills precursor" framing.

---

## [v1.0] — 2026-04-30

### Added

- Initial submission-ready demo: 14 GREAT+OKAY hand-curated pairs over Northern Sweden / Finland from Mapillary v4.
- DISK + LightGlue matcher, USAC-MAGSAC homography with ground-plane bias, Mask2Former-Cityscapes segmenter, alpha-blended overlay.
- Streamlit viewer over cached results.
- README + writeup essay + Marp deck.

---

## [v0.x] — earlier

- v0.1 baseline (Phase A): initial proof of concept.
- v0.2: snow pre-filter (`src/snow_quality.py`), Streamlit curator, Mask2Former-tiny + confidence threshold.
- v0.3: connected-component cleanup, fetcher region expansion (15 cold-climate cities), 4-panel contact sheet.
