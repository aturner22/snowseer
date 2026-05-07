# Snow-Underlay — reproducible build commands.
#
# Canonical fresh-clone workflow:
#
#     uv sync --python 3.12
#     export MAPILLARY_TOKEN=<token from https://www.mapillary.com/dashboard/developers>
#     make reproduce
#
# That produces outputs/video/boreas_2021_01_26/overlay.mp4 — a 15 s
# cross-season road overlay on a snowy Toronto drive (Boreas dataset,
# CC BY 4.0). Compute: ~6 GB download + ~50 min matching on Mac CPU.
#
# Other entry points:
#
#   make track TRACK=<id> [TRACK_START=N TRACK_END=N]  full pipeline, any track
#   make reproduce-track TRACK=<id>         alias for the above (default 0..350)
#   make stills                             static-prior precursor (single-prior
#                                           v1 narrative; 27 demo pairs from
#                                           data/demo_pairs.json)
#   make oracle TRACK=<id>                  pre-flight gate before any cache build

.PHONY: help track reproduce reproduce-track reproduce-track-alts reproduce-everything \
        assets extract-stills pages-assets submission-bundle test oracle demo notebook \
        stills stills-fetch stills-pipeline stills-audit stills-multi \
        slides writeup pdfs \
        video-fetch video-render video-augment clean dist-clean

# ─── Master entry points ────────────────────────────────────────────────────

CANONICAL_TRACK := boreas_2021_01_26
CANONICAL_TAG   := canonical

help:
	@echo "Snow-Underlay  ·  make targets"
	@echo ""
	@echo "  Reproduce the canonical clip:"
	@echo "    make reproduce                      Full pipeline on the canonical track:"
	@echo "                                          cache + 5 layouts + stills"
	@echo "                                          → outputs/video/boreas_2021_01_26/"
	@echo "    make reproduce-track TRACK=<id>     Same pipeline on any registered track"
	@echo "    make track TRACK=<id> TRACK_START=N TRACK_END=N"
	@echo "                                        Lower-level — explicit window override"
	@echo ""
	@echo "  Static-prior demo (the v1.x narrative):"
	@echo "    make stills                         Single-prior (default): 27 Mapillary demo pairs"
	@echo "                                          → outputs/heroes/{matches,naive_baseline,overlay,panel}.png"
	@echo "    make stills-multi                   Multi-prior fusion ablation (K=5, Phase J)"
	@echo "    make demo SNOW=<jpg> PRIOR=<jpg>    Run the pipeline on any (snow, prior) pair"
	@echo "                                          → outputs/demo/<id>__<layout>.png (15 layouts)"
	@echo ""
	@echo "  Documentation:"
	@echo "    make pdfs                           Render docs/{slides,writeup}.pdf (gitignored)"
	@echo "    make slides                         Marp deck"
	@echo "    make writeup                        Pandoc essay PDF"
	@echo "    make notebook                       Re-execute docs/analysis.ipynb in place"
	@echo ""
	@echo "  Pre-flight oracle (verify priors + summer segmentation BEFORE cache compute):"
	@echo "    make oracle TRACK=<id> [STRIDE=10]  Check demo-ability + suggest candidate windows"
	@echo ""
	@echo "  Asset bundles (slides plan + Pages):"
	@echo "    make assets                         Render canonical clip's full asset bundle"
	@echo "                                          (5 layouts + extracted stills)"
	@echo "    make extract-stills TRACK=<id>      Extract JPEGs at preset timestamps from"
	@echo "                                          all mp4s in outputs/video/<id>/"
	@echo "    make pages-assets                   Copy canonical mp4s + stills into docs/_assets/"
	@echo "                                          for GitHub Pages deployment"
	@echo "    make submission-bundle              Stage submission/ dir (PDFs + mp4s, local-only)"
	@echo "    make reproduce-everything           Full bundle: canonical + all alts + stills +"
	@echo "                                          static panels + writeup PDF + pages assets"
	@echo ""
	@echo "  Lower-level Phase K targets:"
	@echo "    make video-fetch TRACK=<id>         Pull snow + summer Boreas track"
	@echo "    make video-render TRACK=<id> MODE=<m>  Render one layout from cache"
	@echo "    make video-augment TRACK=<id> CACHE=<tag>  Build naive + summer panels cache"
	@echo ""
	@echo "  Tests:"
	@echo "    make test                           Smoke tests (import graph + CLI shape; no compute)"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean                          Remove generated outputs (heroes/audit/frames)"
	@echo "    make dist-clean                     Also remove cached pair downloads"

TRACK_START ?= 0
TRACK_END   ?= 350

# Single deterministic full-pipeline target. Every track produces the same
# output shape under outputs/video/<id>/:
#   _cache_<tag>.pkl  _aug_<tag>.pkl
#   overlay.mp4  sidebyside.mp4  snow_naive_overlay.mp4
#   snow_overlay_naive.mp4  quad.mp4
#   stills/<layout>__t<NNNN>.jpg
track:
	@if [ -z "$(TRACK)" ]; then echo "usage: make track TRACK=<id> [TRACK_START=N TRACK_END=N]"; exit 2; fi
	uv run python -m src.video_runtime.fetch_track --track $(TRACK)
	uv run python -m src.video_runtime.render_all_layouts \
	    --track $(TRACK) --cache-tag $(CANONICAL_TAG) \
	    --start $(TRACK_START) --end $(TRACK_END) --stride 1 \
	    --K 3 --ema-alpha 0.4
	uv run python -m src.video_runtime.extract_assets --track $(TRACK)
	@echo ""
	@echo "Done. outputs/video/$(TRACK)/ — overlay.mp4 + 4 alternate layouts + stills/"

# Canonical reproducer — what a clean clone runs to make the headline clip.
reproduce:
	$(MAKE) track TRACK=$(CANONICAL_TRACK) TRACK_START=100 TRACK_END=250

# Same pipeline on any other registered track.
reproduce-track:
	@if [ -z "$(TRACK)" ]; then echo "usage: make reproduce-track TRACK=<id>"; exit 2; fi
	$(MAKE) track TRACK=$(TRACK)

# ─── Static-prior quick test ────────────────────────────────────────────────

# Default `make stills` cleans heroes/ first so stale multi-prior fusion
# outputs from a previous run don't contaminate the single-prior audit.
stills: clean-heroes stills-fetch stills-pipeline stills-audit
	@echo ""
	@echo "Static stills built (single-prior, v1.x narrative). See outputs/heroes/"
	@echo "and outputs/audit/contact_sheet.png."

stills-fetch:
	uv run python -m data.fetch_mapillary --curated-only

# Default: single-prior (v1.x narrative). The pipeline writes the v1 outputs only:
#   __matches.png  __naive_baseline.png  __overlay.png  __panel.png
stills-pipeline:
	uv run python -m src.pipeline --max-priors 1

# Multi-prior fusion ablation (Phase J). K = 5 priors per pair, three fusion
# strategies (union / weighted / majority) compared side-by-side. Adds:
#   __overlay_union.png  __overlay_weighted.png  __overlay_majority.png  __priors.png
# Substantially slower than single-prior (matching cost scales with K).
stills-multi: clean-heroes stills-fetch
	uv run python -m src.pipeline --max-priors 5
	uv run python -m src.audit
	@echo ""
	@echo "Multi-prior fusion ablation built. See outputs/audit/contact_sheet.png"
	@echo "for the 5-column per-pair comparison."

stills-audit:
	uv run python -m src.audit

# ─── Analysis notebook ─────────────────────────────────────────────────────

# Re-execute the analysis notebook in place. Loads DISK + LightGlue +
# Mask2Former and runs the worked-example cells. First run can take a
# few minutes for HF cache warm-up; subsequent runs ~30s.
notebook:
	uv run jupyter nbconvert \
	    --to notebook --execute --inplace \
	    --ExecutePreprocessor.timeout=600 \
	    docs/analysis.ipynb

# ─── Live interactive demo ─────────────────────────────────────────────────

# Run the cross-season pipeline on any (snow, prior) image pair. Outputs
# the full 15-layout per-pair set under OUT (default outputs/demo/).
demo:
	@if [ -z "$(SNOW)" ] || [ -z "$(PRIOR)" ]; then \
	    echo "usage: make demo SNOW=<snow.jpg> PRIOR=<clear.jpg> [OUT=outputs/demo]"; exit 2; \
	fi
	uv run python -m src.pipeline \
	    --snow "$(SNOW)" --prior "$(PRIOR)" \
	    --out-dir $(or $(OUT),outputs/demo)

# ─── Asset bundles (slides plan + GitHub Pages) ─────────────────────────────

assets:
	$(MAKE) track TRACK=$(CANONICAL_TRACK) TRACK_START=100 TRACK_END=250

extract-stills:
	@if [ -z "$(TRACK)" ]; then echo "usage: make extract-stills TRACK=<id>"; exit 2; fi
	uv run python -m src.video_runtime.extract_assets --track $(TRACK)

pages-assets:
	@mkdir -p docs/_assets/canonical/stills
	@if [ -f outputs/video/$(CANONICAL_TRACK)/stills/overlay__t005p0.jpg ]; then \
	    cp -f outputs/video/$(CANONICAL_TRACK)/stills/overlay__t005p0.jpg \
	          docs/_assets/canonical/stills/overlay__t005p0.jpg; \
	    echo "  staged canonical hero still"; \
	else \
	    echo "  [SKIP] outputs/video/$(CANONICAL_TRACK)/stills/overlay__t005p0.jpg missing — run 'make assets' first"; \
	fi

# Master "produce every shippable artefact" target. Slow.
reproduce-everything: reproduce reproduce-track-alts assets stills writeup pages-assets
	@echo ""
	@echo "Reproduce-everything complete. The canonical + alts + stills + static"
	@echo "panels + writeup PDF + Pages assets are all up to date."

reproduce-track-alts:
	$(MAKE) reproduce-track TRACK=boreas_2025_02_15

# ─── Oracle (pre-flight before any cache build) ─────────────────────────────

# Verify a track has a demo-able window before committing cache compute.
# Checks UTM distance to nearest summer prior + summer-prior road-segmentation
# coverage; reports candidate windows.
oracle:
	@if [ -z "$(TRACK)" ]; then echo "usage: make oracle TRACK=<id> [STRIDE=10]"; exit 2; fi
	uv run python -m src.video_runtime.window_oracle --track $(TRACK) \
	    $(if $(STRIDE),--stride $(STRIDE)) \
	    --out-json outputs/video/$(TRACK)/_oracle.json

# ─── Lower-level Phase K targets ────────────────────────────────────────────

video-fetch:
	@if [ -z "$(TRACK)" ]; then echo "usage: make video-fetch TRACK=<track_id>"; exit 2; fi
	uv run python -m src.video_runtime.fetch_track --track $(TRACK)

video-render:
	@if [ -z "$(TRACK)" ]; then echo "usage: make video-render TRACK=<id> MODE=<overlay|sidebyside|snow_naive_overlay|snow_overlay_naive|quad>"; exit 2; fi
	@if [ -z "$(MODE)" ]; then echo "usage: make video-render TRACK=<id> MODE=<overlay|...>"; exit 2; fi
	uv run python -m src.video_runtime.render --track $(TRACK) --mode $(MODE)

video-augment:
	@if [ -z "$(TRACK)" ]; then echo "usage: make video-augment TRACK=<id> CACHE=<tag>"; exit 2; fi
	@if [ -z "$(CACHE)" ]; then echo "usage: make video-augment TRACK=<id> CACHE=<tag>"; exit 2; fi
	uv run python -m src.video_runtime.augment --track $(TRACK) --cache-tag $(CACHE)

# ─── Documentation ──────────────────────────────────────────────────────────

pdfs: slides writeup

# docs/slides.md has two halves: the Marp slide deck (renders to slides.pdf)
# and a markdown video-plan appendix at the end (read as plain markdown by
# the editor).
slides:
	npx -y --package=@marp-team/marp-cli@latest -- marp docs/slides.md \
	    -o docs/slides.pdf --allow-local-files --theme-set docs/style/marp.css

writeup:
	pandoc docs/writeup.md -o docs/writeup.pdf -V geometry:margin=2cm \
	    --pdf-engine=xelatex \
	    --variable=mainfont:"EB Garamond" \
	    --variable=sansfont:"Inter" \
	    --variable=monofont:"JetBrains Mono"

# ─── Submission bundle (manual, local-only — not in git) ───────────────────

# Copies the submission deliverables into ./submission/ so the user has a
# single directory to upload from. The bundle is local-only — it contains
# binaries (mp4s, PDFs) that we do NOT commit. Run AFTER `make assets`,
# `make pages-assets`, and `make pdfs`.
.PHONY: submission-bundle
submission-bundle:
	@mkdir -p submission
	@echo "  → copying writeup + slides PDFs"
	@cp -f docs/writeup.pdf submission/writeup.pdf 2>/dev/null || echo "  ! writeup.pdf missing — run make writeup first"
	@cp -f docs/slides.pdf submission/slides.pdf 2>/dev/null || echo "  ! slides.pdf missing — run make slides first"
	@echo "  → copying canonical overlay clip"
	@cp -f outputs/video/$(CANONICAL_TRACK)/overlay.mp4 submission/overlay.mp4 2>/dev/null || echo "  ! overlay.mp4 missing — run make reproduce first"
	@echo "  → copying canonical 5-layout asset bundle"
	@for f in sidebyside snow_naive_overlay snow_overlay_naive quad; do \
	    cp -f outputs/video/$(CANONICAL_TRACK)/$$f.mp4 submission/$$f.mp4 2>/dev/null || true; \
	done
	@echo "  → copying alt-track headlines (every track with a built overlay.mp4)"
	@for track in boreas_2025_02_15; do \
	    cp -f outputs/video/$$track/overlay.mp4 submission/$$track.mp4 2>/dev/null || true; \
	done
	@echo "  → copying source repo URL"
	@echo "https://github.com/aturner22/snowseer" > submission/REPO.txt
	@echo "Submitted to SoTA Commission I — Minimal-Shot Autonomy" >> submission/REPO.txt
	@echo ""
	@echo "Submission bundle staged at ./submission/. Contents:"
	@ls -la submission/

# ─── Hygiene helpers ──────────────────────────────────────────────────────

# Wipe outputs/heroes/ stills before a fresh single-prior run, so that
# stale multi-prior fusion outputs (overlay_union/weighted/majority,
# priors strip) from a previous run don't contaminate the audit's per-pair
# detection (audit infers "single-prior" from the *absence* of those
# files). Idempotent. Safe to call before either single-prior or
# multi-prior runs.
.PHONY: clean-heroes
clean-heroes:
	rm -f outputs/heroes/*.png outputs/heroes/*.jpg outputs/heroes/summary.json
	@echo "  outputs/heroes/ cleaned"

# Wipe outputs/audit/ before a fresh contact-sheet generation. Useful when
# switching between single-prior and multi-prior modes.
.PHONY: clean-audit
clean-audit:
	rm -rf outputs/audit
	@echo "  outputs/audit/ cleaned"

# ─── Tests (smoke / CLI shape only — no model loading) ────────────────────

# Light tests that never load model weights or touch GPU. Verify import
# graph + argparse shape + Make targets + .gitignore policy. Fast (< 5 s)
# and safe to run during cache builds.
test:
	uv run python tests/test_smoke.py

# ─── Cleanup ────────────────────────────────────────────────────────────────

clean:
	rm -rf outputs/heroes/*.png outputs/heroes/*.jpg outputs/heroes/summary.json
	rm -rf outputs/audit outputs/pipeline_run.log
	rm -rf outputs/video/*/frames

dist-clean: clean
	rm -rf data/pairs
	rm -rf outputs/video/*/_cache_*.pkl outputs/video/*/_aug_*.pkl
