"""Streamlit result-quality labeller.

After the pipeline has produced overlay panels, the user rates each one on a
4-point ordinal scale: GREAT / OKAY / NOT VERY GOOD / AWFUL. The labels are
persisted and used downstream to pick the demo heroes.

Usage:
    uv run streamlit run demo/curate_results.py

State persistence: data/manual_result_curation.json keyed by pair_id, each
value `{rating: "great"|"okay"|"not_good"|"awful", note: str}`.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

HEROES = Path("outputs/heroes")
PAIRS_DIR = Path("data/pairs")
DECISIONS_PATH = Path("data/manual_result_curation.json")

RATINGS = [
    ("great", "🌟 GREAT", "#1f9e3a"),
    ("okay", "🟡 OKAY", "#d4a017"),
    ("not_good", "🟠 NOT VERY GOOD", "#d96c1f"),
    ("awful", "💀 AWFUL", "#b22222"),
]
RATING_VALUES = {k: i for i, (k, _, _) in enumerate(RATINGS)}  # 0=great, 3=awful

st.set_page_config(page_title="Result curator", layout="wide")


@st.cache_data
def _load_results():
    """All pairs that have a panel.png (i.e., the pipeline produced an
    overlay). Sorted by inlier count, descending."""
    summary_path = HEROES / "summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text())
    out = []
    for s in summary:
        pid = s["pair_id"]
        panel = HEROES / f"{pid}__panel.png"
        if not panel.exists():
            continue
        meta_path = PAIRS_DIR / pid / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        out.append({
            "pair_id": pid,
            "panel_path": str(panel),
            "naive_path": str(HEROES / f"{pid}__naive_baseline.png"),
            "matches_path": str(HEROES / f"{pid}__matches.png"),
            "summary": s,
            "meta": meta,
        })
    out.sort(key=lambda p: -(p["summary"].get("n_inliers") or 0))
    return out


def _load_decisions() -> dict[str, dict]:
    if DECISIONS_PATH.exists():
        try:
            return json.loads(DECISIONS_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_decisions(d: dict[str, dict]) -> None:
    DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DECISIONS_PATH.write_text(json.dumps(d, indent=2))


def _decide(pair_id: str, rating: str, note: str = "") -> None:
    decisions = st.session_state.decisions
    decisions[pair_id] = {"rating": rating, "note": note}
    _save_decisions(decisions)
    st.session_state.idx = min(st.session_state.idx + 1, len(st.session_state.results) - 1)


# --- Initial state ---
if "results" not in st.session_state:
    st.session_state.results = _load_results()
if "decisions" not in st.session_state:
    st.session_state.decisions = _load_decisions()
if "idx" not in st.session_state:
    results = st.session_state.results
    decisions = st.session_state.decisions
    first_undecided = next((i for i, p in enumerate(results) if p["pair_id"] not in decisions), 0)
    st.session_state.idx = first_undecided

results = st.session_state.results
decisions = st.session_state.decisions

if not results:
    st.error(
        "No panels under outputs/heroes/. Run `uv run python -m src.pipeline` first "
        "(after the snow curator)."
    )
    st.stop()

# --- Top progress strip ---
n_total = len(results)
n_decided = len(decisions)
counts = {k: 0 for k, _, _ in RATINGS}
for v in decisions.values():
    r = v.get("rating")
    if r in counts:
        counts[r] += 1

cols = st.columns(6)
cols[0].metric("Reviewed", f"{n_decided} / {n_total}")
cols[1].metric("🌟 great", counts["great"])
cols[2].metric("🟡 okay", counts["okay"])
cols[3].metric("🟠 not good", counts["not_good"])
cols[4].metric("💀 awful", counts["awful"])
cols[5].metric("Remaining", n_total - n_decided)

st.divider()

# --- Current pair ---
idx = st.session_state.idx
if idx >= n_total:
    st.success("All results reviewed.")
    st.stop()

pair = results[idx]
pair_id = pair["pair_id"]
existing = decisions.get(pair_id, {})

# Header + per-pair metrics
hcols = st.columns([3, 1])
with hcols[0]:
    st.subheader(f"{idx + 1} / {n_total}  ·  {pair_id}")
s = pair["summary"]
m = pair["meta"]
mcols = st.columns(6)
mcols[0].metric("region", m.get("region", "—"))
mcols[1].metric("inliers", s.get("n_inliers", "—"))
mcols[2].metric("matches", s.get("n_matches", "—"))
mcols[3].metric("refined", "yes" if s.get("refined") else "no")
mcols[4].metric("IoU vs naive", (
    f"{s['iou_overlay_vs_naive']:.2f}" if s.get("iou_overlay_vs_naive") is not None else "—"
))
mcols[5].metric("IoU vs identity", (
    f"{s['iou_overlay_vs_identity']:.2f}" if s.get("iou_overlay_vs_identity") is not None else "—"
))

if existing:
    st.info(f"Previously rated: **{existing.get('rating', '?').upper()}** — {existing.get('note', '')}")

# Sidebar size controls (defined later in sidebar block but read here).
img_max_h = st.session_state.get("img_max_h", 700)


def _img(path_str: str) -> None:
    """Render an image scaled to fit `img_max_h`."""
    try:
        from PIL import Image as _PIL
        from pathlib import Path as _P
        if not _P(path_str).exists():
            st.info(f"(missing: {path_str})")
            return
        _w, _h = _PIL.open(path_str).size
        scale = img_max_h / _h if _h > img_max_h else 1.0
        st.image(path_str, width=int(round(_w * scale)))
    except Exception:
        st.image(path_str, use_column_width=True)


# Headline 3-panel
st.markdown("##### 3-panel: snowy query | clear prior + road | snow + overlay")
_img(pair["panel_path"])

# Side-by-side overlay vs naive (the contrast condition)
ncols = st.columns(2)
with ncols[0]:
    st.markdown("**Cross-season overlay (this work)** — derived from clear-season prior + cross-season alignment.")
    overlay_path = HEROES / f"{pair_id}__overlay.png"
    if overlay_path.exists():
        _img(str(overlay_path))
with ncols[1]:
    st.markdown("**Naive baseline** — same Cityscapes segmenter applied directly to the snow frame, no prior.")
    if Path(pair["naive_path"]).exists():
        _img(pair["naive_path"])

with st.expander("Feature correspondences (DISK + LightGlue, RANSAC inliers in green)"):
    if Path(pair["matches_path"]).exists():
        _img(pair["matches_path"])

note = st.text_input("Optional note (one line)", value=existing.get("note", ""))

# Rating buttons
st.markdown("**Rate this result:**")
bcols = st.columns(len(RATINGS) + 2)
for i, (key, label, color) in enumerate(RATINGS):
    with bcols[i]:
        if st.button(label, key=f"rating_{key}", use_container_width=True):
            _decide(pair_id, key, note)
            st.rerun()
with bcols[-2]:
    if st.button("⬅ BACK", key="back", use_container_width=True):
        st.session_state.idx = max(0, st.session_state.idx - 1)
        st.rerun()
with bcols[-1]:
    if st.button("➡ NEXT (no rating)", key="next", use_container_width=True):
        st.session_state.idx = min(n_total - 1, st.session_state.idx + 1)
        st.rerun()

# Sidebar: jumper, image height, summary
with st.sidebar:
    st.header("Display")
    st.slider("Image max height (px)", min_value=300, max_value=1400,
              value=st.session_state.get("img_max_h", 700), step=50, key="img_max_h")

    st.header("Jump to")
    jump_to = st.selectbox("pair_id", options=[p["pair_id"] for p in results], index=idx)
    if jump_to != results[idx]["pair_id"]:
        st.session_state.idx = next(i for i, p in enumerate(results) if p["pair_id"] == jump_to)
        st.rerun()

    st.divider()
    st.header("Top-rated so far")
    for key, label, color in RATINGS:
        rated = [pid for pid, v in decisions.items() if v.get("rating") == key]
        if not rated:
            continue
        st.markdown(f"**{label}** ({len(rated)})")
        for pid in rated[-12:]:
            st.text(f"  • {pid}")

    st.divider()
    if st.button("Reset all ratings"):
        st.session_state.decisions = {}
        _save_decisions({})
        st.session_state.idx = 0
        st.rerun()
