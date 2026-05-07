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
#   make reproduce-all-layouts TRACK=<id>   — render all 5 layout variants
#   make reproduce-track TRACK=<id>         — full pipeline on a different track
#   make stills                             — static-prior quick test (the original
#                                             14-pair Mapillary demo, panels only)
#
# Project layout: src/video_runtime/ is the per-frame video pipeline; src/
# top level is the static-prior pipeline used by `make stills`. Legacy code
# is under _archive/ (gitignored).

.PHONY: help reproduce reproduce-all-layouts reproduce-track reproduce-track-alts \
        reproduce-all-layouts-canonical extract-stills-canonical reproduce-everything \
        assets extract-stills pages-assets submission-bundle test tidy oracle \
        stills stills-fetch stills-pipeline stills-audit stills-multi \
        slides writeup pdfs \
        video-fetch video-render video-augment video-recon clean dist-clean

# ─── Master entry points ────────────────────────────────────────────────────

CANONICAL_TRACK := boreas_2021_01_26
CANONICAL_TAG   := canonical

help:
	@echo "Snow-Underlay  ·  make targets"
	@echo ""
	@echo "  Reproduce the canonical clip:"
	@echo "    make reproduce                      One 15s overlay clip from boreas_2021_01_26"
	@echo "                                          → outputs/video/boreas_2021_01_26/overlay.mp4"
	@echo "    make reproduce-all-layouts TRACK=<id>  All 5 layouts (overlay/sidebyside/3-panel x2/quad)"
	@echo "    make reproduce-track TRACK=<id>     Full pipeline on a non-canonical track"
	@echo ""
	@echo "  Static-prior demo (the v1.x narrative):"
	@echo "    make stills                         Single-prior (default): 14 curated Mapillary pairs"
	@echo "                                          → outputs/heroes/{matches,naive_baseline,overlay,panel}.png"
	@echo "    make stills-multi                   Multi-prior fusion ablation (K=5, Phase J)"
	@echo ""
	@echo "  Documentation:"
	@echo "    make pdfs                           Render docs/{slides,writeup}.pdf (gitignored)"
	@echo "    make slides                         Marp deck (slides + video-plan appendix)"
	@echo "    make writeup                        Pandoc essay PDF"
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
	@echo "    make video-recon CITY=<name>        Mapillary winter sequence reconnaissance"
	@echo ""
	@echo "  Tests:"
	@echo "    make test                           Smoke tests (import graph + CLI shape; no compute)"
	@echo ""
	@echo "  Repo hygiene:"
	@echo "    make tidy                           Remove stale logs + .DS_Store + .ruff_cache (safe)"
	@echo "    make clean                          Remove generated outputs (heroes/audit/frames)"
	@echo "    make dist-clean                     Also remove cached pair downloads"

# Canonical reproducer — what the user runs from a clean clone.
# Pulls the boreas_2021_01_26 snow+summer windows (~1.4 GB), builds the
# matching cache (~50 min), renders the overlay layout (~1 min).
reproduce:
	uv run python -m src.video_runtime.fetch_track --track $(CANONICAL_TRACK)
	uv run python -m src.video_runtime.render --track $(CANONICAL_TRACK) \
	    --start 100 --end 250 --stride 1 --K 3 \
	    --temporal ema --ema-alpha 0.4 \
	    --cache-tag $(CANONICAL_TAG) --out-name overlay.mp4
	@echo ""
	@echo "Done. Watch outputs/video/$(CANONICAL_TRACK)/overlay.mp4."

# Render all 5 layouts (overlay / sidebyside / 3-panel x2 / quad) for a track.
# Reuses the matching cache + augmentation cache; adds ~5 min of render time.
reproduce-all-layouts:
	@if [ -z "$(TRACK)" ]; then echo "usage: make reproduce-all-layouts TRACK=<id>"; exit 2; fi
	uv run python -m src.video_runtime.render_all_layouts \
	    --track $(TRACK) --cache-tag $(CANONICAL_TAG) \
	    --start 100 --end 250 --stride 1 --K 3 --ema-alpha 0.4

# Full pipeline on a non-canonical track.
reproduce-track:
	@if [ -z "$(TRACK)" ]; then echo "usage: make reproduce-track TRACK=<id>"; exit 2; fi
	uv run python -m src.video_runtime.fetch_track --track $(TRACK)
	uv run python -m src.video_runtime.render --track $(TRACK) \
	    --start 0 --end 350 --stride 1 --K 3 \
	    --temporal ema --ema-alpha 0.4 \
	    --cache-tag $(CANONICAL_TAG) --out-name overlay.mp4
	@echo ""
	@echo "Done. Watch outputs/video/$(TRACK)/overlay.mp4."

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

# ─── Asset bundles (slides plan + GitHub Pages) ─────────────────────────────

# Render the canonical track in all 5 layouts and extract preset stills.
# Single command, single track. Reuses any existing matching + augment caches.
assets: reproduce-all-layouts-canonical extract-stills-canonical
	@echo ""
	@echo "Canonical asset bundle ready under outputs/video/$(CANONICAL_TRACK)/."

reproduce-all-layouts-canonical:
	uv run python -m src.video_runtime.render_all_layouts \
	    --track $(CANONICAL_TRACK) --cache-tag $(CANONICAL_TAG) \
	    --start 100 --end 250 --stride 1 --K 3 --ema-alpha 0.4

extract-stills-canonical:
	uv run python -m src.video_runtime.extract_assets --track $(CANONICAL_TRACK)

# Per-track variant (alts).
extract-stills:
	@if [ -z "$(TRACK)" ]; then echo "usage: make extract-stills TRACK=<id>"; exit 2; fi
	uv run python -m src.video_runtime.extract_assets --track $(TRACK)

# Copy finalised mp4s + a representative still per track into docs/_assets/
# so the GitHub Pages site (docs/index.html) can reference them via stable
# relative paths. Run this after `make assets` (and after `make reproduce-track`
# for any alt tracks you want on Pages).
pages-assets:
	@mkdir -p docs/_assets/canonical
	@if [ -d outputs/video/$(CANONICAL_TRACK) ]; then \
	    for f in overlay sidebyside snow_naive_overlay snow_overlay_naive quad; do \
	        if [ -f outputs/video/$(CANONICAL_TRACK)/$$f.mp4 ]; then \
	            cp -f outputs/video/$(CANONICAL_TRACK)/$$f.mp4 docs/_assets/canonical/$$f.mp4; \
	        fi; \
	    done; \
	    if [ -d outputs/video/$(CANONICAL_TRACK)/stills ]; then \
	        mkdir -p docs/_assets/canonical/stills; \
	        cp -f outputs/video/$(CANONICAL_TRACK)/stills/*.jpg docs/_assets/canonical/stills/ 2>/dev/null || true; \
	    fi; \
	    echo "  copied canonical assets"; \
	fi
	@for track in boreas_2024_12_23 boreas_2025_02_15; do \
	    if [ -d outputs/video/$$track ]; then \
	        mkdir -p docs/_assets/$$track; \
	        for f in overlay sidebyside snow_naive_overlay snow_overlay_naive quad; do \
	            if [ -f outputs/video/$$track/$$f.mp4 ]; then \
	                cp -f outputs/video/$$track/$$f.mp4 docs/_assets/$$track/$$f.mp4; \
	            fi; \
	        done; \
	        if [ -d outputs/video/$$track/stills ]; then \
	            mkdir -p docs/_assets/$$track/stills; \
	            cp -f outputs/video/$$track/stills/*.jpg docs/_assets/$$track/stills/ 2>/dev/null || true; \
	        fi; \
	        echo "  copied $$track assets"; \
	    fi; \
	done
	@echo ""
	@echo "Pages assets staged under docs/_assets/. Commit and push to deploy."

# Master "produce every shippable artefact" target. Slow.
reproduce-everything: reproduce reproduce-track-alts assets stills writeup pages-assets
	@echo ""
	@echo "Reproduce-everything complete. The canonical + alts + stills + static"
	@echo "panels + writeup PDF + Pages assets are all up to date."

# Helper: run reproduce-track for each known alt sequentially. Sequential is
# critical — running two cache builds in parallel on a Mac with ≤16 GB RAM
# guarantees swap thrashing and silently degrades both.
reproduce-track-alts:
	$(MAKE) reproduce-track TRACK=boreas_2024_12_23
	$(MAKE) reproduce-track TRACK=boreas_2025_02_15

# ─── Oracle (pre-flight before any cache build) ─────────────────────────────

# `make oracle TRACK=<id>` — verify the track has a demo-able window before
# committing cache compute. Checks (a) UTM distance to nearest summer prior
# and (b) summer-prior road-segmentation coverage. Reports candidate windows
# (longest contiguous runs where every frame has ≥ 1 acceptable prior).
#
# Hard rule (per memory feedback_phase_l_window_oracle.md): never run
# `make reproduce-track TRACK=<id>` without a satisfied oracle pass first.
# Cost: dominated by the segmentation cache (~5–10 min on Mac CPU for a
# typical 350-frame track + K=3 priors). Cached aggressively per summer
# frame.
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

video-recon:
	@if [ -z "$(CITY)" ]; then echo "usage: make video-recon CITY=<name|all>"; exit 2; fi
	uv run python -m data.find_snow_sequences --city $(CITY)

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
	@echo "  → copying alt-track headlines"
	@for track in boreas_2024_12_23 boreas_2025_02_15; do \
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

# ─── Repo hygiene ─────────────────────────────────────────────────────────

# `make tidy` — reproducible physical cleanup of the working tree.
# Removes generated logs, .DS_Store, .ruff_cache, __pycache__. Does NOT
# touch outputs/video/<track>/_cache_*.pkl (matching cache, expensive to
# regenerate) or outputs/video/<track>/*.mp4 (renders). Run after a long
# session where stale logs / pycache directories accumulated.
#
# For deeper archival cleanup (move stale / experimental outputs to
# _archive/), see _archive/REFACTOR_NOTES.md or do it manually — these
# moves are deliberate decisions, not automatable.
tidy:
	@echo "  removing stale logs + debug renders + macOS / cache cruft"
	@rm -rf outputs/video/_match_test/
	@rm -f outputs/video/_*.log
	@rm -f outputs/_*.log
	@rm -f outputs/fetch_*.log outputs/pipeline_run.log
	@rm -f outputs/.DS_Store outputs/heroes/.gitkeep
	@find . -name '.DS_Store' -not -path './.venv/*' -not -path './.git/*' -delete 2>/dev/null || true
	@rm -rf .ruff_cache/
	@find . -type d -name '__pycache__' -not -path './.venv/*' -not -path './.git/*' -not -path './_archive/*' -exec rm -rf {} + 2>/dev/null || true
	@echo "  done. \`make clean\` for a deeper sweep."

# ─── Cleanup ────────────────────────────────────────────────────────────────

clean:
	rm -rf outputs/heroes/*.png outputs/heroes/*.jpg outputs/heroes/summary.json
	rm -rf outputs/audit outputs/pipeline_run.log
	rm -rf outputs/video/*/frames

dist-clean: clean
	rm -rf data/pairs
	rm -rf outputs/video/*/_cache_*.pkl outputs/video/*/_aug_*.pkl
