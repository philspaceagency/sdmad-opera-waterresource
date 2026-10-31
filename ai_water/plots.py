"""The water-frequency figure: one per water body, ascending and descending passes combined.

White = never water, then five blue classes (<=5, 5-20, 20-50, 50-80, 80-100 % of the valid observations), a legend strip, scale
bar and north arrow. Pixels never observed (outside the outline, no valid pass) are left white. Cropped to the observed area.
"""
from __future__ import annotations

import numpy as np

BOUNDS = (0, 5, 20, 50, 80, 100)
COLORS = ("#ffffff", "#a6cdee", "#5ba4e5", "#2f78d4", "#1b3fa6", "#0a1a82")        # 0 %, then one per class
INK, FRAME = "#22303c", "#8a97a3"


def _nice_length(extent_m):
    """A round scale-bar length (m) of about a fifth of `extent_m`."""
    target = extent_m / 5.0
    for step in (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000):
        if step >= target * 0.7:
            return step
    return 100000


def class_index(freq):
    """0 for never-water, 1..5 for the classes above, -1 where never observed (255)."""
    f = np.asarray(freq)
    idx = np.digitize(f, BOUNDS[1:-1], right=True) + 1
    idx = np.where(f == 0, 0, idx)
    return np.where(f == 255, -1, idx)


def frequency_figure(path, freq, geobox, title, subtitle="", footer="", margin_px=20):
    """Write the frequency picture to `path` (PNG). `freq`: uint8 0-100 (255 = never observed) on `geobox` (metric CRS)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    idx = class_index(freq)
    seen = idx >= 0
    if not seen.any():
        raise ValueError("no observed pixel: nothing to draw")
    rows, cols = np.where(seen)
    r0, r1 = max(rows.min() - margin_px, 0), min(rows.max() + margin_px + 1, idx.shape[0])
    c0, c1 = max(cols.min() - margin_px, 0), min(cols.max() + margin_px + 1, idx.shape[1])
    img = idx[r0:r1, c0:c1]
    h, w = img.shape
    res = abs(float(geobox.transform.a))

    fig_w = 8.0
    dpi = float(np.clip(w / (fig_w * 0.92), 150, 400))                    # about one image pixel per raster pixel
    map_h = fig_w * 0.92 * h / w
    fig = plt.figure(figsize=(fig_w, map_h + 1.8), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0.04, 0.95 / fig.get_figheight(), 0.92, map_h / fig.get_figheight()])
    ax.imshow(np.ma.masked_less(img, 0), cmap=ListedColormap(COLORS), vmin=-0.5, vmax=len(COLORS) - 0.5, interpolation="nearest",
              aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(FRAME); s.set_linewidth(0.8)

    km = _nice_length(w * res)                                            # scale bar, lower left
    px = km / res
    x0, y0 = 0.035 * w, h * 0.965
    ax.add_patch(Rectangle((x0 - 0.01 * w, y0 - 0.075 * h), px + 0.02 * w, 0.075 * h + 0.01 * h, fc="white", ec="none", alpha=0.85, zorder=3))
    ax.plot([x0, x0 + px], [y0 - 0.012 * h] * 2, color=INK, lw=2.2, solid_capstyle="butt", zorder=4)
    ax.text(x0 + px / 2, y0 - 0.025 * h, f"{km / 1000:g} km" if km >= 1000 else f"{km:g} m", ha="center", va="bottom", fontsize=8,
            color=INK, zorder=4)
    ax.annotate("N", xy=(0.955, 0.945), xytext=(0.955, 0.865), xycoords="axes fraction", textcoords="axes fraction", ha="center",
                fontsize=10, fontweight="bold", color=INK, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))

    fig.text(0.04, 1.0 - 0.32 / fig.get_figheight(), title, fontsize=16, fontweight="bold", color=INK, va="center", ha="left")
    if subtitle:
        fig.text(0.04, 1.0 - 0.68 / fig.get_figheight(), subtitle, fontsize=9, color="#53606b", va="center", ha="left")

    lg = fig.add_axes([0.04, 0.30 / fig.get_figheight(), 0.50, 0.17 / fig.get_figheight()])          # legend strip
    lg.set_xlim(0, 7.0); lg.set_ylim(0, 1); lg.axis("off")
    lg.add_patch(Rectangle((0, 0), 1, 1, fc=COLORS[0], ec=FRAME, lw=0.8))
    for k in range(1, 6):
        lg.add_patch(Rectangle((1 + (k - 1), 0), 1, 1, fc=COLORS[k], ec=FRAME, lw=0.8))
    lg.text(0.5, -0.35, "0", ha="center", va="top", fontsize=8, color=INK)
    for k, b in enumerate(BOUNDS[1:], start=1):
        lg.text(1 + k, -0.35, f"{b}%", ha="center", va="top", fontsize=8, color=INK)
    lg.text(0, 1.45, "Water frequency", ha="left", va="bottom", fontsize=9.5, color=INK)
    if footer:
        fig.text(0.96, 0.30 / fig.get_figheight(), footer, fontsize=7, color="#7b8791", va="center", ha="right")
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path
