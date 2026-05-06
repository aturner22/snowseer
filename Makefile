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

.PHONY: help reproduce reproduce-all-layouts reproduce-track \
        stills stills-fetch stills-pipeline stills-audit stream pdfs slides writeup \
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
	@echo "    make pdfs                           Render docs/{slides,writeup}.pdf"
	@echo "    make slides                         Render only slides.pdf"
	@echo "    make writeup                        Render only writeup.pdf"
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

slides:
	npx -y --package=@marp-team/marp-cli@latest -- marp docs/slides.md \
	    -o docs/slides.pdf --allow-local-files --theme-set docs/style/marp.css

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
