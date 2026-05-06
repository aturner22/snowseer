# `docs/_assets/`

Final mp4 + still bundle for the [GitHub Pages site](../index.html).

This directory is **populated by** `make pages-assets`. Files are copies of the
canonical / alt-track renders under `outputs/video/<track>/` (which are
gitignored). The Pages site references them via stable relative paths
like `_assets/canonical/overlay.mp4`.

## Workflow

```bash
# 1. Build the canonical clip (~50 min compute, one-time):
make reproduce

# 2. Build the full canonical asset bundle (5 layouts + extracted stills):
make assets

# 3. (Optional) Build the alts:
make reproduce-track TRACK=boreas_2024_12_23
make reproduce-track TRACK=boreas_2025_02_15
make extract-stills TRACK=boreas_2024_12_23
make extract-stills TRACK=boreas_2025_02_15

# 4. Stage everything into docs/_assets/ for Pages:
make pages-assets

# 5. Commit + push. GitHub Pages serves from main /docs.
```

## Expected layout after `make pages-assets`

```
docs/_assets/
├── README.md                          (this file)
├── canonical/
│   ├── overlay.mp4
│   ├── sidebyside.mp4
│   ├── snow_naive_overlay.mp4
│   ├── snow_overlay_naive.mp4
│   ├── quad.mp4
│   └── stills/<layout>__t<NNNN>.jpg   (16 files: 4 layouts × 4 timestamps)
├── boreas_2024_12_23/
│   └── overlay.mp4 + (other layouts if rendered) + stills/
└── boreas_2025_02_15/
    └── overlay.mp4 + (other layouts if rendered) + stills/
```

mp4 file sizes are small (≤ 25 MB each); the full Pages bundle stays well
under the GitHub Pages 1 GB soft-cap.
