"""Shared style of the paper's figures (IEEE column 3.5 in, page 7.16 in)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

UAVSAR, S1, SAMEDAY = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1, 2, 3
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d2"
COLUMN, PAGE = 3.5, 7.16


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "image.interpolation": "none", "savefig.dpi": 600})


def panel_label(ax, label, color=INK, bg=None, corner="left"):
    box = dict(fc=bg, ec="none", pad=1.0) if bg else None
    x, ha = (0.01, "left") if corner == "left" else (0.99, "right")
    ax.text(x, 0.99, label, transform=ax.transAxes, fontsize=7, va="top", ha=ha,
            fontweight="bold", color=color, bbox=box)
