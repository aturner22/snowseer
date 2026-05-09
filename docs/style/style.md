# Visual identity

Minimal, considered, infrastructure-coloured. One accent. Generous whitespace. No decoration that doesn't carry meaning.

## Palette

| Token | Hex | Where |
| --- | --- | --- |
| `bg` (warm off-white) | `#f6f3ee` | Background of every long-form artefact. |
| `text` (charcoal) | `#1c1c1c` | Body text. Hairline borders. Code. |
| `accent` (rust) | `#b34a25` | Section markers. Pull-quotes. Used sparingly: the only colour with semantic weight. |
| `mute` (warm grey) | `#8a8780` | Secondary captions, axis labels. |

The accent is plough-yellow infrastructure, not Christmas-sweater festive. Think rust on a snow plough's blade.

## Typography

| Element | Face | Notes |
| --- | --- | --- |
| Body text | EB Garamond | Long-form. Open Font License. Old-style figures where supported. |
| Headers | Inter | Weights 400 / 500 / 700. Tracking slightly negative on display sizes. |
| Captions / UI labels | Inter or JetBrains Mono | Mono for code only; sans for everything else short. |
| Code blocks | JetBrains Mono | 1.5x line-height. |

Source files live in `docs/assets/fonts/`. Open Font License compatible; redistributable.

## Spacing

- Base body line-height: 1.55x (essay reading, not technical reference).
- Generous paragraph spacing (1.0em). Section breaks at 2.0em.
- Figure margins: 2.5em above and below. Captions tight under the figure (0.5em).

## Restraint rules

1. **One accent only.** Never two coloured highlights on the same page.
2. **Hairlines, not boxes.** Borders are 1px charcoal at ~30% opacity. No drop shadows.
3. **Whitespace is content.** Don't fill it with chartjunk.
4. **The image carries.** When a figure is shown, the surrounding text is short.

## Where the identity is applied

- `docs/index.html` + `docs/style/site.css`: GitHub Pages site.
- `outputs/nordic_stills/<id>__{snow,clear,clear_with_mask,overlay,naive_baseline,matches}.png`: raw constituent images that the site composes into 2x2 grids in HTML.
- `outputs/toronto_video/<id>/{overlay,sidebyside,quad,...}.mp4`: overlays use green for the road and red for the naive baseline.
