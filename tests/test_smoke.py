"""Smoke tests — module imports + CLI shape.

These tests deliberately do NOT load model weights or run the matcher /
segmenter (those need ~9 GB of memory and 5+ s per pair). The goal is to
catch import errors and CLI regressions in CI without burning compute.

Run locally with: `uv run python -m pytest tests/`
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# When invoked as `python tests/test_smoke.py`, ensure imports of `src.*`
# resolve from the project root rather than the tests directory.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_pipeline_imports() -> None:
    """The static-stills pipeline modules import cleanly."""
    import src.fuse
    import src.homography
    import src.matching
    import src.overlay
    import src.pipeline
    import src.segmentation
    import src.snow_quality
    import src.audit

    assert callable(src.pipeline.run_pair)
    assert callable(src.pipeline.run_all)
    assert hasattr(src.pipeline, "PairResult")


def test_video_runtime_imports() -> None:
    """The video pipeline modules import cleanly."""
    import src.video_runtime.pipeline_v as pv
    import src.video_runtime.prior_pool
    import src.video_runtime.temporal
    import src.video_runtime.track
    import src.video_runtime.overlay_render
    import src.video_runtime.augment
    import src.video_runtime.extract_assets
    import src.video_runtime.fetch_track
    import src.video_runtime.render
    import src.video_runtime.render_all_layouts

    assert callable(pv.run_track)
    assert pv.CHECKPOINT_EVERY == 50
    assert pv.ETA_EVERY == 10


def test_pipeline_cli_max_priors_flag() -> None:
    """`--max-priors` flag is exposed and defaults to single-prior."""
    r = subprocess.run(
        [sys.executable, "-m", "src.pipeline", "--help"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, f"--help failed: {r.stderr}"
    assert "--max-priors" in r.stdout
    # Default = 1 (single-prior, v1.x narrative)
    assert "default 1" in r.stdout.lower()


def test_pipeline_cli_demo_flags() -> None:
    """`--snow` and `--prior` are exposed for the live `make demo` entry."""
    r = subprocess.run(
        [sys.executable, "-m", "src.pipeline", "--help"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0
    for flag in ("--snow", "--prior"):
        assert flag in r.stdout, f"missing flag {flag}"


def test_video_render_cli_args() -> None:
    """The video render CLI exposes the canonical args (track / mode /
    cache-tag / temporal). Schema check, no execution."""
    r = subprocess.run(
        [sys.executable, "-m", "src.video_runtime.render", "--help"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0
    for flag in ["--track", "--mode", "--cache-tag", "--temporal", "--ema-alpha", "--K"]:
        assert flag in r.stdout, f"missing flag {flag}"


def test_demo_pairs_well_formed() -> None:
    """`src/data/demo_pairs.json` parses + has the expected schema."""
    spec_path = ROOT / "src" / "data" / "demo_pairs.json"
    if not spec_path.exists():
        # Fresh-clone case: file ships in git but tolerate its absence.
        return
    spec = json.loads(spec_path.read_text())
    assert "pairs" in spec
    pairs = spec["pairs"]
    assert isinstance(pairs, list) and len(pairs) > 0
    for p in pairs:
        assert "pair_id" in p
        # Either v1 (snow_id, prior_ids) or v2 (priors list of dicts)
        assert ("snow_id" in p) or ("snow" in p)


def test_audit_module_layout() -> None:
    """audit.py keeps the contact-sheet entry point stable."""
    import src.audit
    assert callable(src.audit.main)
    # Single-prior detection lives in _build_row.
    assert callable(src.audit._build_row)


def test_makefile_targets_exist() -> None:
    """Critical Make targets are listed in the Makefile."""
    mk = (ROOT / "Makefile").read_text()
    for target in [
        "track:", "reproduce:", "reproduce-track:",
        "stills:", "stills-multi:", "assets:", "extract-stills:",
        "pages-assets:", "pdfs:",
    ]:
        assert target in mk, f"missing make target {target}"


def test_gitignore_keeps_binaries_out() -> None:
    """The .gitignore enforces the no-large-binaries policy."""
    gi = (ROOT / ".gitignore").read_text()
    for pattern in ["**/*.mp4", "**/*.pkl", "docs/*.pdf", "data/video/", "outputs/video/"]:
        assert pattern in gi, f"missing gitignore pattern {pattern}"


# ─── Standalone runner (no pytest dep needed) ─────────────────────────────
# Allows `python tests/test_smoke.py` and CI to execute these without
# adding pytest to the project dependencies. pytest still works if the
# user has it installed.

def _run_all() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    fails: list[str] = []
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except Exception as e:  # noqa: BLE001 — surface the failure verbatim
            print(f"  FAIL {t.__name__}: {e}")
            fails.append(t.__name__)
    print(f"\n{len(tests) - len(fails)}/{len(tests)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_run_all())
