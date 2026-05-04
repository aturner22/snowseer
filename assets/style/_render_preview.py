"""Render a one-shot preview of the visual identity (palette + type pairing).

Used once during Phase H.1 to sanity-check the locked aesthetic. Rerunnable.
Output: assets/style/preview.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
FONTS_DIR = ROOT / "assets/fonts"
OUT = ROOT / "assets/style/preview.png"

# Register local fonts so matplotlib can find them by family name.
for f in FONTS_DIR.glob("*.ttf"):
    fm.fontManager.addfont(str(f))

BG = "#f6f3ee"
TEXT = "#1c1c1c"
ACCENT = "#b34a25"
MUTE = "#8a8780"


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_axis_off()
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # — Wordmark (Inter, all caps, tracked) —
    ax.text(4, 53, "S N O W — U N D E R L A Y", fontfamily="Inter", fontweight=700, fontsize=11,
            color=TEXT, transform=ax.transData)
    ax.plot([4, 18], [51.5, 51.5], color=ACCENT, linewidth=1.8)

    # — Body specimen (EB Garamond) —
    body = (
        "Self-driving systems are trained on dry roads, deliberately. "
        "A snow plough operates in the regime that data excludes. The "
        "road is invisible — but it has not moved."
    )
    ax.text(4, 44.5, body, fontfamily="EB Garamond", fontsize=12,
            color=TEXT, wrap=True, va="top", linespacing=1.45)

    # — Section header (Inter Bold) —
    ax.text(4, 28.5, "Constants as the bridge", fontfamily="Inter",
            fontweight=700, fontsize=14, color=TEXT)
    ax.text(4, 25, "What stays the same when everything else changes.",
            fontfamily="EB Garamond", fontsize=10.5, color=MUTE, style="italic")

    # — Code block (JetBrains Mono) —
    ax.add_patch(Rectangle((4, 14), 50, 8, facecolor="#1c1c1c0a", edgecolor="none"))
    ax.text(5.2, 19.5, "uv run make demo", fontfamily="JetBrains Mono",
            fontsize=10, color=TEXT)
    ax.text(5.2, 17, "uv run streamlit run demo/streamlit_app.py",
            fontfamily="JetBrains Mono", fontsize=10, color=TEXT)

    # — Palette swatches —
    palette = [
        ("bg",     BG,     "#f6f3ee"),
        ("text",   TEXT,   "#1c1c1c"),
        ("accent", ACCENT, "#b34a25"),
        ("mute",   MUTE,   "#8a8780"),
    ]
    y0 = 4.5
    for i, (label, color, hex_str) in enumerate(palette):
        x = 4 + i * 24
        # Swatch
        ax.add_patch(Rectangle((x, y0), 5, 5, facecolor=color,
                               edgecolor=TEXT, linewidth=0.5))
        ax.text(x + 6.5, y0 + 3.3, label, fontfamily="Inter", fontweight=500,
                fontsize=10, color=TEXT)
        ax.text(x + 6.5, y0 + 1.2, hex_str, fontfamily="JetBrains Mono",
                fontsize=8.5, color=MUTE)

    # — Hairline (rust accent — used sparingly) —
    ax.plot([4, 96], [60, 60], color=ACCENT, linewidth=1.0, alpha=0.85,
            transform=ax.transData)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
