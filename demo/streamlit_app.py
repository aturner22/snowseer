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
st.markdown(
    "**Cross-season image matching for snow-plough autonomy.** "
    "Match a live snowy frame against a clear-season prior of the same coordinates, "
    "transfer the road segmentation through the alignment. "
    "*Zero snowy frames touch any model weights — snow appears only at inference time, "
    "as the runtime input.*"
)

summary_path = OUT_DIR / "summary.json"
if not summary_path.exists():
    st.error(
        "No cached results found. Run `uv run python -m data.fetch_mapillary` then "
        "`uv run python -m src.pipeline` first."
    )
    st.stop()

summary = json.loads(summary_path.read_text())

# --- Sidebar: filtering and selection ---
with st.sidebar:
    st.header("Filter")
    show_only_accepted = st.checkbox("Accepted pairs only", value=True)
    pool = [s for s in summary if (not show_only_accepted) or s.get("accept")]
    if not pool:
        st.warning("No pairs match the current filter. Disable 'Accepted pairs only' to see rejects.")
        st.stop()

    st.header("Quick jumps")
    if st.button("→ Best (highest inliers)"):
        st.session_state["pair_idx"] = max(range(len(pool)), key=lambda i: pool[i].get("n_inliers", 0))
    rejects = [i for i, s in enumerate(pool) if not s.get("accept")]
    if rejects and st.button("→ A rejected/failure case"):
        st.session_state["pair_idx"] = rejects[0]
    refined_idx = next((i for i, s in enumerate(pool) if s.get("refined")), None)
    if refined_idx is not None and st.button("→ A refinement-rescued pair"):
        st.session_state["pair_idx"] = refined_idx

    st.header("Pair")
    options = [s["pair_id"] for s in pool]
    default = st.session_state.get("pair_idx", 0)
    default = min(default, len(options) - 1)
    pair_id = st.selectbox("Location", options=options, index=default)

selected = next(s for s in pool if s["pair_id"] == pair_id)
meta_path = PAIRS_DIR / pair_id / "meta.json"
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

# --- Top metric strip ---
cols = st.columns(6)
cols[0].metric("region", meta.get("region", "—"))
cols[1].metric("pair Δ distance (m)", meta.get("distance_m", "—"))
cols[2].metric("Δ heading (°)", meta.get("heading_delta_deg", "—"))
cols[3].metric("RANSAC inliers", selected.get("n_inliers", "—"))
cols[4].metric("refined?", "yes" if selected.get("refined") else "no")
cols[5].metric("accept?", "yes" if selected.get("accept") else "no")

iou_n = selected.get("iou_overlay_vs_naive")
iou_i = selected.get("iou_overlay_vs_identity")
if iou_n is not None or iou_i is not None:
    iou_cols = st.columns(2)
    if iou_n is not None:
        iou_cols[0].metric("IoU(overlay vs naive)", f"{iou_n:.2f}")
    if iou_i is not None:
        iou_cols[1].metric("IoU(overlay vs identity-warp)", f"{iou_i:.2f}")

# --- Headline 3-panel figure ---
panel_path = OUT_DIR / f"{pair_id}__panel.png"
if panel_path.exists():
    st.image(
        str(panel_path),
        caption="Snowy query | clear prior + Cityscapes road | snow frame + warped road overlay",
        use_column_width=True,
    )
else:
    st.info("This pair has no overlay (graceful failure — matcher correctly declined).")

# --- Naive baseline ---
st.subheader("Naive baseline — Cityscapes segmenter applied directly to snow")
st.caption(
    "No cross-season prior; no matching; no warp. The model has never seen snow during "
    "training; the failure mode is fragmented or absent road predictions."
)
naive_path = OUT_DIR / f"{pair_id}__naive_baseline.png"
if naive_path.exists():
    st.image(str(naive_path), use_column_width=True)
else:
    st.info("Naive baseline not yet generated for this pair.")

# --- Feature correspondences ---
st.subheader("Feature correspondences")
st.caption("DISK + LightGlue. Green: RANSAC inliers under the ground-plane-biased homography. Red: rejected.")
matches_path = OUT_DIR / f"{pair_id}__matches.png"
if matches_path.exists():
    st.image(str(matches_path), use_column_width=True)

with st.expander("Minimal-shot integrity guarantee"):
    st.markdown(
        """
| Component | Pretrained on | Snow in training? |
| --- | --- | --- |
| DISK feature extractor | MegaDepth | **No** |
| LightGlue matcher | MegaDepth | **No** |
| Segformer-B0 road segmenter | Cityscapes | **No** |

Snow appears only at inference time, as the runtime input. No model weights are
fine-tuned on snowy data anywhere in this codebase.
        """
    )
