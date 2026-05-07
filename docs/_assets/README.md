# `docs/_assets/`

Still-image bundle for the [GitHub Pages site](../index.html).

This directory is **populated by** `make pages-assets`. Files are copies of the
canonical / alt-track renders under `outputs/video/<track>/` (which are
gitignored). The Pages site references the JPEGs via stable relative paths
like `_assets/canonical/stills/overlay__t005p0.jpg`.

## Policy

Only the **stills (small JPEGs, ≤ 500 KB each)** are committed to git. The
mp4s — even though `pages-assets` copies them here for local preview — are
gitignored under the project's `**/*.mp4` rule. The Pages site renders the
poster JPEG in lieu of the actual mp4. The user can decide later whether
to add the mp4s out-of-band (external video host, or a deliberate
large-file commit if Pages's per-file ≤ 100 MB allows).

## Workflow

```bash
# 1. Build the canonical clip (~50 min compute, one-time):
make reproduce

# 2. Build the full canonical asset bundle (5 layouts + extracted stills):
make assets

# 3. (Optional) Build the alts (each: ~50 min cache + ~25 min renders):
make reproduce-track TRACK=boreas_2024_12_23
make extract-stills TRACK=boreas_2024_12_23
make reproduce-track TRACK=boreas_2025_02_15
make extract-stills TRACK=boreas_2025_02_15

# 4. Stage everything into docs/_assets/ for Pages:
make pages-assets

# 5. Commit the new stills (mp4s are auto-ignored). Push.
#    GitHub Pages serves from main /docs.
git add docs/_assets/
git commit -m "assets: <track> stills"
git push
```

## Expected layout after `make pages-assets`

```
docs/_assets/
├── README.md                                 (this file)
├── canonical/
│   ├── overlay.mp4 + sidebyside.mp4 + ...    (gitignored)
│   └── stills/<layout>__t<NNN>p<N>.jpg       (~20 files: 5 layouts × 4 timestamps)
├── boreas_2024_12_23/
│   ├── overlay.mp4 + ...                     (gitignored)
│   └── stills/                               (24 files: 6 layouts × 4 ts)
└── boreas_2025_02_15/
    ├── overlay.mp4 + ...                     (gitignored)
    └── stills/                               (likewise)
```

## Filename convention

Stills are named `<layout>__t<NNN>p<N>.jpg` where `NNN.N` is the extraction
timestamp in seconds, formatted with leading zeros (`001p0` for 1.0 s,
`014p0` for 14.0 s). The `extract_assets` module uses `(1.0, 5.0, 10.0,
14.0)` by default; override with `--timestamps` to extract different
points.
