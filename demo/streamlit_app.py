"""Streamlit demo viewer over cached pipeline outputs.

This app runs no models. It reads pre-generated figures from outputs/heroes/.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

OUT_DIR = Path("outputs/heroes")
PAIRS_DIR = Path("data/pairs")

st.set_page_config(page_title="Snow-Underlay", layout="wide")

st.title("Snow-Underlay")
st.caption(
    "Cross-season image matching for snow-plough autonomy. "
    "Click a location to see the pipeline output. "
    "Snow imagery is used only as a runtime input — no model weights are trained on it."
)

summary_path = OUT_DIR / "summary.json"
if not summary_path.exists():
    st.error(
        "No cached results found. Run `uv run python -m data.fetch_mapillary` then "
        "`uv run python -m src.pipeline` first."
    )
    st.stop()

summary = json.loads(summary_path.read_text())

with st.sidebar:
    st.header("Pair")
    options = [s["pair_id"] for s in summary]
    pair_id = st.selectbox("Location", options=options)

selected = next(s for s in summary if s["pair_id"] == pair_id)

# Metadata strip
meta_path = PAIRS_DIR / pair_id / "meta.json"
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    cols = st.columns(4)
    cols[0].metric("region", meta["region"])
    cols[1].metric("pair distance (m)", meta["distance_m"])
    cols[2].metric("heading delta (°)", meta["heading_delta_deg"])
    cols[3].metric("inliers", selected.get("n_inliers", "—"))

# Headline 3-panel figure
panel_path = OUT_DIR / f"{pair_id}__panel.png"
if panel_path.exists():
    st.image(str(panel_path), caption="Snowy query | clear prior + Cityscapes road | snow frame + warped road overlay", use_column_width=True)

st.subheader("Naive baseline")
st.caption("Same Cityscapes segmenter, applied directly to the snowy frame. No cross-season prior.")
naive_path = OUT_DIR / f"{pair_id}__naive_baseline.png"
if naive_path.exists():
    st.image(str(naive_path), use_column_width=True)
else:
    st.info("Naive baseline not yet generated for this pair.")

st.subheader("Feature correspondences")
st.caption("DISK + LightGlue. Green: RANSAC inliers under the ground-plane-biased homography. Red: rejected.")
matches_path = OUT_DIR / f"{pair_id}__matches.png"
if matches_path.exists():
    st.image(str(matches_path), use_column_width=True)
