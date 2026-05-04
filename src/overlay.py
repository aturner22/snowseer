"""Mask warping + 3-panel visualization."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def warp_mask(
    mask_src: np.ndarray, H_src_to_dst: np.ndarray, dst_shape: tuple[int, int]
) -> np.ndarray:
    """Warp a binary mask from img0 to img1 using a homography img0 -> img1."""
    h, w = dst_shape
    warped = cv2.warpPerspective(
        mask_src.astype(np.uint8), H_src_to_dst, (w, h), flags=cv2.INTER_NEAREST
    )
    return warped


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


def three_panel_figure(
    snowy: np.ndarray,
    clear: np.ndarray,
    clear_road_mask: np.ndarray,
    snowy_overlay: np.ndarray,
    title: str = "",
    out_path: str | Path | None = None,
) -> None:
    """Save a 3-panel comparison: snowy | clear+mask | snowy+overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(snowy)
    axes[0].set_title("Snowy query frame")
    axes[1].imshow(alpha_blend(clear, clear_road_mask, color=(0, 200, 255), alpha=0.4))
    axes[1].set_title("Clear prior frame  (road mask from Segformer)")
    axes[2].imshow(snowy_overlay)
    axes[2].set_title("Snowy frame + warped road overlay")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
