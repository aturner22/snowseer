# Visual identity

> *Constants as the bridge.* The aesthetic answers the brief: minimal, considered, infrastructure-coloured. One accent. Generous whitespace. No decoration that doesn't carry meaning.

## Palette

| Token | Hex | Where |
| --- | --- | --- |
| `bg` (warm off-white) | `#f6f3ee` | Background of every long-form artefact. README, slides, PDF, video safe-zone. |
| `text` (charcoal) | `#1c1c1c` | Body text. Hairline borders. Code. |
| `accent` (rust) | `#b34a25` | Section markers. Pull-quotes. The overlay panel's frame. The video's progress bar. **Used sparingly — it's the only colour with semantic weight.** |
| `mute` (warm grey) | `#8a8780` | Secondary captions, axis labels. |

The accent is plough-yellow infrastructure, not Christmas-sweater festive. Think rust on a snow plough's blade.

## Typography

| Element | Face | Notes |
| --- | --- | --- |
| Body text | EB Garamond | Long-form. Open Font License. Old-style figures where supported. |
| Headers | Inter | All weights 400 / 500 / 700. Tracking slightly negative on display sizes. |
| Captions / UI / code labels | Inter or JetBrains Mono | Mono only for code; sans for everything else short. |
| Code blocks | JetBrains Mono | 1.5× line-height. |

Source files live in `docs/_assets/fonts/`. Open Font License compatible; redistributable.

## Spacing

- Base body line-height: 1.55× (essay reading, not technical reference).
- Generous paragraph spacing (1.0em). Section breaks at 2.0em.
- Figure margins: 2.5em above and below. Captions tight under the figure (0.5em).
- Page margins (PDF): 28mm sides, 25mm top/bottom.

## Restraint rules

1. **One accent only.** Never two coloured highlights on the same page.
2. **Hairlines, not boxes.** Borders are 1px charcoal at 30 % opacity. No drop shadows.
3. **Headers argue.** Each section header should advance the argument, not just label content. "Constants as the bridge" beats "Methodology".
4. **Whitespace is content.** Don't fill it with chartjunk.
5. **The image carries.** When a figure is shown, the surrounding text is short.

## Where the identity is applied

- `README.md` — banner, run instructions, hero image
- `docs/writeup.md` — the essay (rendered to PDF via `make writeup` with EB Garamond body + Inter headers; PDF is gitignored)
- `docs/slides.md` — Marp deck via `docs/style/marp.css` + a markdown video-plan appendix
- `docs/index.html` + `docs/style/site.css` — GitHub Pages site (charcoal / cream / rust palette, Google-Fonts CDN for EB Garamond + Inter; serves the canonical static stills as poster placeholders)
- `outputs/heroes/<id>__panel.png` — figure typography (Inter caption labels), rust frame on the overlay column
- `outputs/video/<id>/{overlay,sidebyside,...}.mp4` — overlays use `--green` (`#2e9c56`) for the road and `--red` (`#dc3c32`) for the naive-baseline failure mode
- `demo/streamlit_app.py` — custom CSS injection (time-permitting)

## Reference

Inspiration: SOTA Letters (https://sotaletters.substack.com/) for the writing voice and minimal-monochrome layout. The accent colour is our own — SOTA Letters runs pure monochrome.
