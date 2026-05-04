"""Mask warping + panel visualisation.

Visual identity (locked in `docs/style/style.md`):
    bg     #f6f3ee  (warm off-white)
    text   #1c1c1c  (charcoal)
    accent #b34a25  (rust — used semantically, not decoratively)
    mute   #8a8780  (warm grey, secondary captions)
Body font EB Garamond; headers / labels Inter; code JetBrains Mono.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# Identity tokens
BG = "#f6f3ee"
TEXT = "#1c1c1c"
ACCENT = "#b34a25"
MUTE = "#8a8780"

# Register local fonts on import so figure typography is consistent across runs.
_FONTS_DIR = Path(__file__).resolve().parents[1] / "assets/fonts"
if _FONTS_DIR.exists():
    for _f in _FONTS_DIR.glob("*.ttf"):
        try:
            fm.fontManager.addfont(str(_f))
        except Exception:
            pass


def warp_mask(
    mask_src: np.ndarray, H_src_to_dst: np.ndarray, dst_shape: tuple[int, int]
) -> np.ndarray:
    """Warp a binary mask from img0 to img1 using a homography img0 -> img1."""
    h, w = dst_shape
    warped = cv2.warpPerspective(
        mask_src.astype(np.uint8), H_src_to_dst, (w, h), flags=cv2.INTER_NEAREST
    )
    return warped


def keep_largest_component(mask: np.ndarray, *, min_area_px: int = 500) -> np.ndarray:
    """Reduce a binary mask to its single largest connected component.

    A snow plough cares about the *one* drivable surface directly in front of
    it, not the long tail of small islands that warp aliasing or segmenter
    noise can leave behind. This collapses the mask to that single component.
    Returns an empty mask if no component clears `min_area_px`.

    Connectivity: 8-way (matches typical road shape with diagonal pixels).
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask)
    # Component 0 is background — skip. Find the largest by area.
    areas = stats[1:, cv2.CC_STAT_AREA]
    if not len(areas):
        return np.zeros_like(mask)
    largest_idx = int(np.argmax(areas)) + 1  # +1 to account for skipped background
    largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])
    if largest_area < min_area_px:
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    out[labels == largest_idx] = 1
    return out


def alpha_blend(
    img: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 100),
    alpha: float = 0.45,
    edge_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Alpha-blend a binary mask onto the image with an optional outline.

    edge_color: explicit outline RGB. Default chooses a high-contrast variant
    of the fill color so the road boundary is visible on both snow and asphalt.
    """
    out = img.copy()
    color_arr = np.array(color, dtype=np.uint8)
    blended = (alpha * color_arr + (1 - alpha) * out[mask > 0]).astype(np.uint8)
    out[mask > 0] = blended
    if edge_color is None:
        # A darker, saturated version of the fill color contrasts on both white snow and gray road.
        edge_color = tuple(max(int(c) - 80, 0) for c in color)
    edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
    out[edges > 0] = np.array(edge_color, dtype=np.uint8)
    return out


def panel_figure(
    snowy: np.ndarray,
    clear: np.ndarray,
    clear_road_mask: np.ndarray,
    snowy_overlay: np.ndarray,
    snowy_naive: np.ndarray | None = None,
    title: str = "",
    subtitle: str = "",
    out_path: str | Path | None = None,
) -> None:
    """Save the headline 4-column comparison.

    Columns, left to right:
      1. Snowy query frame — what the plough's camera sees.
      2. Clear prior + road mask — what we know about the same coordinates.
      3. Snow frame with warped road overlay — the cross-season output (rust frame).
      4. Snow frame with naive direct-on-snow segmentation — the failure
         condition that motivates the cross-season approach.

    If `snowy_naive` is None the figure falls back to 3 columns.
    """
    n_cols = 4 if snowy_naive is not None else 3
    # Figure proportions: each panel ~16:9-ish in cell ratio. Wider panels +
    # added top space for the two-line header.
    fig = plt.figure(figsize=(5.0 * n_cols, 4.4), facecolor=BG)
    # GridSpec with a slim header band.
    gs = fig.add_gridspec(2, n_cols, height_ratios=[0.18, 1.0], hspace=0.05, wspace=0.04)
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.set_facecolor(BG)
    header_ax.set_axis_off()
    if title:
        header_ax.text(0.0, 0.85, title, fontfamily="Inter", fontweight=500,
                       fontsize=14, color=TEXT, ha="left", va="top",
                       transform=header_ax.transAxes)
    if subtitle:
        header_ax.text(0.0, 0.18, subtitle, fontfamily="EB Garamond",
                       fontsize=11, color=MUTE, ha="left", va="top",
                       style="italic", transform=header_ax.transAxes)

    axes = [fig.add_subplot(gs[1, i]) for i in range(n_cols)]
    for ax in axes:
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    label_kwargs = dict(
        fontfamily="Inter", fontsize=11.5, color=TEXT, pad=10, loc="left"
    )

    axes[0].imshow(snowy)
    axes[0].set_title("Snow query", **label_kwargs)

    axes[1].imshow(alpha_blend(clear, clear_road_mask, color=(80, 140, 170), alpha=0.42))
    axes[1].set_title("Clear prior  ·  road mask", **label_kwargs)

    axes[2].imshow(snowy_overlay)
    axes[2].set_title("Cross-season overlay", **label_kwargs)
    # Rust-accent frame on the load-bearing column.
    for spine_pos in ("top", "bottom", "left", "right"):
        axes[2].spines[spine_pos].set_visible(True)
        axes[2].spines[spine_pos].set_color(ACCENT)
        axes[2].spines[spine_pos].set_linewidth(2.2)

    if snowy_naive is not None:
        axes[3].imshow(snowy_naive)
        axes[3].set_title("Naive  ·  direct on snow", **label_kwargs)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BG, pad_inches=0.35)
    plt.close(fig)


# Backwards-compat alias; new code should use panel_figure.
three_panel_figure = panel_figure
