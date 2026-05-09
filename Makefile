# Snowseer: reproducible build commands.
#
# Quick start (full reproduction, ~2 hours on Mac CPU):
#
#     uv sync --python 3.12
#     export MAPILLARY_TOKEN=<token>     # needed for the stills step
#     make reproduce
#
# `make reproduce` runs every shippable artefact sequentially: canonical
# clip → alt-track robustness clip → 18-pair static-stills precursor.
# For ad-hoc single runs, use `make track TRACK=<id>` or `make stills`
# directly.

.PHONY: help reproduce track stills pdfs test clean

CANONICAL_TRACK := boreas_2021_01_26
ALT_TRACK       := boreas_2025_02_15
CANONICAL_TAG   := canonical

help:
	@echo "Snowseer  ·  make targets"
	@echo ""
	@echo "  reproduce                   Full sequential build:"
	@echo "                                canonical clip + alt clip + 18-pair stills"
	@echo "                                (~2 h, needs MAPILLARY_TOKEN)"
	@echo ""
	@echo "  track TRACK=<id>            Full pipeline on one registered track"
	@echo "  stills                      Static-prior demo on 18 Mapillary pairs"
	@echo "                                (needs MAPILLARY_TOKEN)"
	@echo ""
	@echo "  pdfs                        Render docs/{slides,writeup}.pdf"
	@echo "  test                        Smoke tests (no compute)"
	@echo "  clean                       Wipe generated outputs"

# ─── Full reproduction (sequential; never run two compute jobs in parallel) ─

reproduce:
	@echo ""
	@echo "[$$(date +%H:%M:%S)] reproduce: step 1/3 — canonical clip ($(CANONICAL_TRACK))"
	$(MAKE) track TRACK=$(CANONICAL_TRACK)
	@echo ""
	@echo "[$$(date +%H:%M:%S)] reproduce: step 2/3 — alt clip ($(ALT_TRACK))"
	$(MAKE) track TRACK=$(ALT_TRACK)
	@echo ""
	@echo "[$$(date +%H:%M:%S)] reproduce: step 3/3 — 18-pair static-stills"
	$(MAKE) stills
	@echo ""
	@echo "[$$(date +%H:%M:%S)] reproduce: complete"
	@echo "  canonical + alt clips under outputs/toronto_video/"
	@echo "  static-stills under outputs/nordic_stills/"

# ─── Single-track / single-demo ancillaries ────────────────────────────────

track:
	@if [ -z "$(TRACK)" ]; then echo "usage: make track TRACK=<id>"; exit 2; fi
	@echo "[$$(date +%H:%M:%S)] track $(TRACK): fetch"
	@if [ -f data/video/tracks/$(TRACK)/snow/camera_poses.csv ]; then \
	    echo "[track] $(TRACK) already staged on disk; skipping fetch_track"; \
	else \
	    uv run python -m src.video_runtime.fetch_track --track $(TRACK); \
	fi
	@echo "[$$(date +%H:%M:%S)] track $(TRACK): matching cache + 5 layout renders"
	uv run python -m src.video_runtime.render_all_layouts \
	    --track $(TRACK) --cache-tag $(CANONICAL_TAG) \
	    --stride 1 --K 3 --ema-alpha 0.4
	@echo "[$$(date +%H:%M:%S)] track $(TRACK): extract stills"
	uv run python -m src.video_runtime.extract_assets --track $(TRACK)
	@echo "[$$(date +%H:%M:%S)] track $(TRACK): done"

# ─── Static-stills demo ─────────────────────────────────────────────────────

stills:
	@echo "[$$(date +%H:%M:%S)] stills: clean previous outputs"
	rm -f outputs/nordic_stills/*.png outputs/nordic_stills/*.jpg outputs/nordic_stills/summary.json
	@echo "[$$(date +%H:%M:%S)] stills: fetch 18 pairs from Mapillary"
	uv run python -m src.data.fetch_mapillary
	@echo "[$$(date +%H:%M:%S)] stills: run pipeline"
	uv run python -m src.pipeline
	@echo "[$$(date +%H:%M:%S)] stills: done"

# ─── Documentation ──────────────────────────────────────────────────────────

pdfs:
	npx -y --package=@marp-team/marp-cli@latest -- marp docs/slides.md \
	    -o docs/slides.pdf --allow-local-files --theme-set docs/style/marp.css
	pandoc docs/writeup.md -o docs/writeup.pdf -V geometry:margin=2cm \
	    --resource-path=docs:. \
	    --pdf-engine=xelatex \
	    --variable=mainfont:"EB Garamond" \
	    --variable=sansfont:"Inter" \
	    --variable=monofont:"JetBrains Mono"

# ─── Tests + cleanup ────────────────────────────────────────────────────────

test:
	uv run python tests/test_smoke.py

clean:
	rm -rf outputs/nordic_stills/*.png outputs/nordic_stills/*.jpg outputs/nordic_stills/summary.json
	rm -rf outputs/pipeline_run.log
	rm -rf outputs/toronto_video/*/frames
