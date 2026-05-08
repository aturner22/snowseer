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

.PHONY: help reproduce track stills pdfs notebook test clean

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
	@echo "  notebook                    Re-execute docs/analysis.ipynb in place"
	@echo "  test                        Smoke tests (no compute)"
	@echo "  clean                       Wipe generated outputs"

# ─── Full reproduction (sequential; never run two compute jobs in parallel) ─

reproduce:
	$(MAKE) track TRACK=$(CANONICAL_TRACK)
	$(MAKE) track TRACK=$(ALT_TRACK)
	$(MAKE) stills
	@echo ""
	@echo "Reproduce complete. Canonical + alt clips under outputs/toronto_video/,"
	@echo "static-stills panels under outputs/nordic_stills/."

# ─── Single-track / single-demo ancillaries ────────────────────────────────

track:
	@if [ -z "$(TRACK)" ]; then echo "usage: make track TRACK=<id>"; exit 2; fi
	@if [ -f data/video/tracks/$(TRACK)/snow/camera_poses.csv ]; then \
	    echo "[track] $(TRACK) already staged on disk; skipping fetch_track"; \
	else \
	    uv run python -m src.video_runtime.fetch_track --track $(TRACK); \
	fi
	uv run python -m src.video_runtime.render_all_layouts \
	    --track $(TRACK) --cache-tag $(CANONICAL_TAG) \
	    --stride 1 --K 3 --ema-alpha 0.4
	uv run python -m src.video_runtime.extract_assets --track $(TRACK)

# ─── Static-stills demo ─────────────────────────────────────────────────────

stills:
	rm -f outputs/nordic_stills/*.png outputs/nordic_stills/*.jpg outputs/nordic_stills/summary.json
	uv run python -m src.data.fetch_mapillary
	uv run python -m src.pipeline

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

notebook:
	uv run jupyter nbconvert \
	    --to notebook --execute --inplace \
	    --ExecutePreprocessor.timeout=600 \
	    docs/analysis.ipynb

# ─── Tests + cleanup ────────────────────────────────────────────────────────

test:
	uv run python tests/test_smoke.py

clean:
	rm -rf outputs/nordic_stills/*.png outputs/nordic_stills/*.jpg outputs/nordic_stills/summary.json
	rm -rf outputs/pipeline_run.log
	rm -rf outputs/toronto_video/*/frames
