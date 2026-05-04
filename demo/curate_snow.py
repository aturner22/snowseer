"""Streamlit snow-frame curator.

Big-image accept/reject for the user. Pairs are presented in descending order
of composite snow-quality score (best candidates first), so the user can stop
early once enough have accepted.

Usage:
    uv run streamlit run demo/curate_snow.py

State persistence: data/manual_snow_curation.json keyed by pair_id, each
value `{verdict: "accept"|"reject"|"skip", note: str}`.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

PAIRS_DIR = Path("data/pairs")
DECISIONS_PATH = Path("data/manual_snow_curation.json")
TARGET_ACCEPTS = 12

st.set_page_config(page_title="Snow curator", layout="wide")


@st.cache_data
def _load_pairs():
    """All pairs ordered by composite snow-quality score (highest first)."""
    out = []
    for d in sorted(PAIRS_DIR.iterdir() if PAIRS_DIR.exists() else []):
        if not d.is_dir():
            continue
        snow = d / "snow.jpg"
        clear = d / "clear.jpg"
        meta = d / "meta.json"
        sq = d / "snow_quality.json"
        if not (snow.exists() and clear.exists()):
            continue
        sq_data = {}
        if sq.exists():
            try:
                sq_data = json.loads(sq.read_text())
            except json.JSONDecodeError:
                sq_data = {}
        meta_data = {}
        if meta.exists():
            try:
                meta_data = json.loads(meta.read_text())
            except json.JSONDecodeError:
                meta_data = {}
        out.append({
            "pair_id": d.name,
            "snow_path": str(snow),
            "clear_path": str(clear),
            "snow_quality": sq_data,
            "meta": meta_data,
        })
    out.sort(key=lambda p: -(p["snow_quality"].get("composite") or 0))
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


def _decide(pair_id: str, verdict: str, note: str = "") -> None:
    decisions = st.session_state.decisions
    decisions[pair_id] = {"verdict": verdict, "note": note}
    _save_decisions(decisions)
    st.session_state.idx = min(st.session_state.idx + 1, len(st.session_state.pairs) - 1)


# --- Initial state ---
if "pairs" not in st.session_state:
    st.session_state.pairs = _load_pairs()
if "decisions" not in st.session_state:
    st.session_state.decisions = _load_decisions()
if "idx" not in st.session_state:
    # Resume at the first un-decided pair.
    pairs = st.session_state.pairs
    decisions = st.session_state.decisions
    first_undecided = next(
        (i for i, p in enumerate(pairs) if p["pair_id"] not in decisions), 0
    )
    st.session_state.idx = first_undecided

pairs = st.session_state.pairs
decisions = st.session_state.decisions

if not pairs:
    st.error("No pairs under data/pairs/. Run the fetcher first.")
    st.stop()

# --- Top progress strip ---
n_total = len(pairs)
n_decided = len(decisions)
n_accepted = sum(1 for v in decisions.values() if v.get("verdict") == "accept")
n_rejected = sum(1 for v in decisions.values() if v.get("verdict") == "reject")
n_skipped = sum(1 for v in decisions.values() if v.get("verdict") == "skip")

cols = st.columns(5)
cols[0].metric("Reviewed", f"{n_decided} / {n_total}")
cols[1].metric("Accepted", n_accepted, delta=f"target {TARGET_ACCEPTS}")
cols[2].metric("Rejected", n_rejected)
cols[3].metric("Skipped", n_skipped)
cols[4].metric("Remaining", n_total - n_decided)

if n_accepted >= TARGET_ACCEPTS:
    st.success(f"Target reached — {n_accepted} pairs accepted. You can stop, or keep going for more.")

st.divider()

# --- Current pair ---
idx = st.session_state.idx
if idx >= n_total:
    st.success("All pairs reviewed.")
    st.stop()

pair = pairs[idx]
pair_id = pair["pair_id"]
existing = decisions.get(pair_id, {})

# Header
hcols = st.columns([3, 1])
with hcols[0]:
    st.subheader(f"{idx + 1} / {n_total}  ·  {pair_id}")
sq = pair.get("snow_quality") or {}
meta = pair.get("meta") or {}
mcols = st.columns(5)
mcols[0].metric("region", meta.get("region", "—"))
mcols[1].metric("composite quality", f"{(sq.get('composite') or 0):.2f}")
mcols[2].metric("sharpness", f"{(sq.get('sharpness') or 0):.0f}")
mcols[3].metric("brightness", f"{(sq.get('brightness') or 0):.0f}")
mcols[4].metric("edges", f"{(sq.get('edge_density') or 0):.3f}")

if existing:
    st.info(f"Previously decided: **{existing.get('verdict')}** — {existing.get('note','')}")

# Snow first, big.
st.markdown("##### Snow query frame  *(this is what the plough's camera would feed in)*")
st.image(pair["snow_path"], use_column_width=True)

with st.expander("Compare with the clear-season prior of the same coordinates", expanded=False):
    st.image(pair["clear_path"], use_column_width=True)

note = st.text_input("Optional note (one line)", value=existing.get("note", ""))

# Buttons
bcols = st.columns([1, 1, 1, 1, 1])
with bcols[0]:
    if st.button("✅ ACCEPT", key="accept", type="primary", use_container_width=True):
        _decide(pair_id, "accept", note)
        st.rerun()
with bcols[1]:
    if st.button("❌ REJECT", key="reject", use_container_width=True):
        _decide(pair_id, "reject", note)
        st.rerun()
with bcols[2]:
    if st.button("⏭ SKIP", key="skip", use_container_width=True):
        _decide(pair_id, "skip", note)
        st.rerun()
with bcols[3]:
    if st.button("⬅ BACK", key="back", use_container_width=True):
        st.session_state.idx = max(0, st.session_state.idx - 1)
        st.rerun()
with bcols[4]:
    if st.button("➡ NEXT", key="next", use_container_width=True):
        st.session_state.idx = min(n_total - 1, st.session_state.idx + 1)
        st.rerun()

# Sidebar: jumper to any pair_id, plus accepted list.
with st.sidebar:
    st.header("Jump to")
    jump_to = st.selectbox("pair_id", options=[p["pair_id"] for p in pairs], index=idx)
    if jump_to != pairs[idx]["pair_id"]:
        st.session_state.idx = next(i for i, p in enumerate(pairs) if p["pair_id"] == jump_to)
        st.rerun()

    st.divider()
    st.header("Accepted so far")
    accepted_ids = [pid for pid, v in decisions.items() if v.get("verdict") == "accept"]
    for a in accepted_ids[-20:]:
        st.text(f"• {a}")

    if st.button("Reset all decisions"):
        st.session_state.decisions = {}
        _save_decisions({})
        st.session_state.idx = 0
        st.rerun()
