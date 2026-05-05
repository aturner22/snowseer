# Snow-Underlay — one-command reproduction.
#
# Canonical fresh-clone workflow:
#   uv sync --python 3.12
#   export MAPILLARY_TOKEN=<token from https://www.mapillary.com/dashboard/developers>
#   make demo
#
# That produces, deterministically, the v1.0 demo: 14 user-curated pair
# panels, the contact sheet, the demo video, and summary.json.

.PHONY: help demo fetch pipeline audit video stream notebook pdfs slides writeup clean dist-clean video-fetch video-render

help:
	@echo "Snow-Underlay  ·  make targets"
	@echo ""
	@echo "  make demo        Pull the 14 curated pairs, run the pipeline, build the contact sheet, render the video"
	@echo "  make stream      Open the Streamlit demo viewer over cached outputs"
	@echo "  make notebook    Re-execute the walkthrough notebook end-to-end"
	@echo "  make pdfs        Render docs/{slides,writeup}.pdf"
	@echo "  make slides      Render only docs/slides.pdf"
	@echo "  make writeup     Render only docs/writeup.pdf"
	@echo ""
	@echo "  make fetch       Pull the 14 curated pairs from Mapillary"
	@echo "  make pipeline    Run the cross-season pipeline on the curated pairs"
	@echo "  make audit       Build outputs/audit/contact_sheet.png"
	@echo "  make video       Render outputs/demo.mp4"
	@echo ""
	@echo "  make clean       Remove generated outputs (keeps cached pairs and INDEX.md)"
	@echo "  make dist-clean  Also remove cached pair downloads"
	@echo ""
	@echo "  Phase K (video / pseudo-realtime, on the 'video' branch):"
	@echo "    make video-fetch TRACK=<id>           Pull snow clip + clear-season priors for a track"
	@echo "    make video-render TRACK=<id> MODE=<m> Render overlay | sidebyside | quad for a track"

demo: fetch pipeline audit video
	@echo ""
	@echo "Demo built. See outputs/heroes/, outputs/audit/contact_sheet.png, outputs/demo.mp4."

fetch:
	uv run python -m data.fetch_mapillary --curated-only

pipeline:
	uv run python -m src.pipeline

audit:
	uv run python -m src.audit

video:
	uv run python -m src.video --out outputs/demo.mp4

stream:
	uv run streamlit run demo/streamlit_app.py

notebook:
	uv run jupyter nbconvert --to notebook --execute notebooks/01_walkthrough.ipynb --output 01_walkthrough.ipynb --ExecutePreprocessor.timeout=900

pdfs: slides writeup

slides:
	npx -y --package=@marp-team/marp-cli@latest -- marp docs/slides.md -o docs/slides.pdf --allow-local-files --theme-set docs/style/marp.css

writeup:
	pandoc docs/writeup.md -o docs/writeup.pdf -V geometry:margin=2cm --pdf-engine=xelatex \
		--variable=mainfont:"EB Garamond" \
		--variable=sansfont:"Inter" \
		--variable=monofont:"JetBrains Mono"

clean:
	rm -rf outputs/heroes/*.png outputs/heroes/*.jpg outputs/heroes/summary.json
	rm -rf outputs/audit outputs/demo.mp4 outputs/pipeline_run.log

dist-clean: clean
	rm -rf data/pairs

# ─── Phase K — pseudo-realtime video pipeline ───────────────────────────────
# These targets only have meaning on the `video` branch. Stub bodies live in
# src/video_runtime/.

video-fetch:
	@if [ -z "$(TRACK)" ]; then echo "usage: make video-fetch TRACK=<track_id>"; exit 2; fi
	uv run python -m src.video_runtime.fetch_track --track $(TRACK)

video-render:
	@if [ -z "$(TRACK)" ]; then echo "usage: make video-render TRACK=<track_id> MODE=<overlay|sidebyside|quad>"; exit 2; fi
	@if [ -z "$(MODE)" ]; then echo "usage: make video-render TRACK=<track_id> MODE=<overlay|sidebyside|quad>"; exit 2; fi
	uv run python -m src.video_runtime.render --track $(TRACK) --mode $(MODE)
