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
        assets extract-stills pages-assets \
        stills stills-fetch stills-pipeline stills-audit stream writeup notebook pdfs \
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
	@echo "  Static-prior quick test (legacy single-prior path):"
	@echo "    make stills                         Pull 14 curated Mapillary pairs, render panels"
	@echo "    make stream                         Open Streamlit viewer over cached static stills"
	@echo ""
	@echo "  Documentation:"
	@echo "    make writeup                        Render docs/writeup.pdf"
	@echo "    make notebook                       Re-execute notebooks/02_video_walkthrough.ipynb"
	@echo "                                          (docs/slides.md is a markdown plan, not a deck)"
	@echo ""
	@echo "  Asset bundles (slides plan + Pages):"
	@echo "    make assets                         Render canonical clip's full asset bundle"
	@echo "                                          (5 layouts + extracted stills)"
	@echo "    make extract-stills TRACK=<id>      Extract JPEGs at preset timestamps from"
	@echo "                                          all mp4s in outputs/video/<id>/"
	@echo "    make pages-assets                   Copy canonical mp4s + stills into docs/_assets/"
	@echo "                                          for GitHub Pages deployment"
	@echo "    make reproduce-everything           Full bundle: canonical + all alts + stills +"
	@echo "                                          static panels + writeup PDF + pages assets"
	@echo ""
	@echo "  Lower-level Phase K targets:"
	@echo "    make video-fetch TRACK=<id>         Pull snow + summer Boreas track"
	@echo "    make video-render TRACK=<id> MODE=<m>  Render one layout from cache"
	@echo "    make video-augment TRACK=<id> CACHE=<tag>  Build naive + summer panels cache"
	@echo "    make video-recon CITY=<name>        Mapillary winter sequence reconnaissance"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean                          Remove generated outputs"
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

stills: stills-fetch stills-pipeline stills-audit
	@echo ""
	@echo "Static stills built. See outputs/heroes/ and outputs/audit/contact_sheet.png."

stills-fetch:
	uv run python -m data.fetch_mapillary --curated-only

stills-pipeline:
	uv run python -m src.pipeline

stills-audit:
	uv run python -m src.audit

stream:
	uv run streamlit run demo/streamlit_app.py

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

notebook:
	uv run jupyter nbconvert --to notebook --execute notebooks/02_video_walkthrough.ipynb \
	    --output 02_video_walkthrough.ipynb --ExecutePreprocessor.timeout=900

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

# docs/slides.md is now a markdown submission-video plan (storyboard +
# asset inventory), not a Marp slide deck. No Marp render needed.
# `make pdfs` is just `make writeup` for backwards compatibility.
pdfs: writeup

writeup:
	pandoc docs/writeup.md -o docs/writeup.pdf -V geometry:margin=2cm \
	    --pdf-engine=xelatex \
	    --variable=mainfont:"EB Garamond" \
	    --variable=sansfont:"Inter" \
	    --variable=monofont:"JetBrains Mono"

# ─── Cleanup ────────────────────────────────────────────────────────────────

clean:
	rm -rf outputs/heroes/*.png outputs/heroes/*.jpg outputs/heroes/summary.json
	rm -rf outputs/audit outputs/pipeline_run.log
	rm -rf outputs/video/*/frames

dist-clean: clean
	rm -rf data/pairs
	rm -rf outputs/video/*/_cache_*.pkl outputs/video/*/_aug_*.pkl
