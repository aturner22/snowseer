# `docs/assets/`

Committed assets used by the GitHub Pages site.

```
assets/
├── fonts/    EB Garamond, Inter, JetBrains Mono (OFL).
└── media/    Stills (jpg/png) and demo clips (mp4).
```

Naming convention under `media/`:

- `toronto_<year>_<layout>.mp4` for video clips.
- `toronto_<year>_<layout>_t<NN>.jpg` for stills extracted at second `NN`.
- `nordic_<city>_<layout>.<ext>` for the static-stills precursor panels.

mp4s are globally gitignored under the project's `**/*.mp4` rule, with an
explicit exception for this directory in `.gitignore` so the Pages site
can load them. Stills are committed without the exception.
