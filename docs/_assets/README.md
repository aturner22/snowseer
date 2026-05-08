# `docs/_assets/`

Committed assets used by the GitHub Pages site, the writeup PDF, and the
slide deck.

```
_assets/
├── fonts/    EB Garamond, Inter, JetBrains Mono (OFL).
└── media/    Stills (jpg/png) and demo clips (mp4).
```

Naming convention under `media/`:

- `toronto_<year>_<layout>.mp4` for video clips.
- `toronto_<year>_<layout>_t<NN>.jpg` for stills extracted at second `NN`.
- `nordic_<city>_<layout>.<ext>` for the static-stills precursor panels.

mp4s are gitignored under the project's `**/*.mp4` rule. JPEGs and PNGs
in this directory are committed so the Pages site and writeup PDF can
build from a clean clone without first running `make track`.
