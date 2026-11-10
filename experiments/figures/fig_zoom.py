r"""Fig. 9 of the paper (page width): raw closures of both sensors in the zone of Fig. 1(b, c).

Rows: UAVSAR, Sentinel-1 on one matched triplet. Columns: (1) closure phase of every cell;
(2) zone closure $\hat\mu$ of every 100 m zone, no selection and no threshold; (3) evidence of a
closure, $-\log_{10}$ of the p-value of the test, continuous (the tick marks $\Pfa$).
Same color scale for both sensors in each column.

Input: ``box.npz`` of a run of ``experiments/sentinel1/zoom_closures.py``.

Example:
    uv run --group figures python experiments/figures/fig_zoom.py --run <runs>/runs/<zoom run> \
        --out zoom.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from style import INK, MUTED, PAGE, panel_label, style

POS, NEG, MID = "#2a78d6", "#e34948", "#f2f1ec"   # diverging poles (as Figs. 7b, 8)
DIV = LinearSegmentedColormap.from_list("closure", [NEG, MID, POS])
SEQ = LinearSegmentedColormap.from_list("evidence", [MID, "#8d8c86", INK])
SENSORS = (("uavsar", "UAVSAR (L)"), ("s1", "Sentinel-1 (C)"))


def raster(x, y, v, box, res, fill):
    """Mean of the values in each ``res`` pixel of the box (north-up); gaps up to ``fill`` px
    take the nearest value."""
    nx, ny = int(round((box[2] - box[0]) / res)), int(round((box[3] - box[1]) / res))
    j = np.clip(((x - box[0]) // res).astype(int), 0, nx - 1)
    i = np.clip(((box[3] - y) // res).astype(int), 0, ny - 1)
    s = np.bincount(i * nx + j, v, nx * ny)
    c = np.bincount(i * nx + j, None, nx * ny)
    with np.errstate(invalid="ignore"):
        img = (s / c).reshape(ny, nx)
    empty = np.isnan(img)
    dist, (ii, jj) = ndimage.distance_transform_edt(empty, return_indices=True)
    sel = empty & (dist <= fill)
    img[sel] = img[ii[sel], jj[sel]]
    return img


def zones_raster(keys, v, box, zone_m):
    e, n = keys // 1_000_000, keys % 1_000_000
    e0, n1 = int(box[0] // zone_m), int(np.ceil(box[3] / zone_m)) - 1
    nx = int(round((box[2] - box[0]) / zone_m))
    img = np.full((nx, nx), np.nan)
    ok = (e - e0 >= 0) & (e - e0 < nx) & (n1 - n >= 0) & (n1 - n < nx)
    img[(n1 - n)[ok], (e - e0)[ok]] = v[ok]
    return img


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(MUTED)
        s.set_linewidth(0.6)


def scale_bar(ax, n_px, size_m, km=1):
    frac = km * 1000 / size_m
    ax.plot([0.05, 0.05 + frac], [0.05, 0.05], transform=ax.transAxes, color=INK, lw=1.2,
            solid_capstyle="butt")
    ax.text(0.05 + frac / 2, 0.07, f"{km} km", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=6.5, bbox=dict(fc="white", ec="none", pad=0.5))


def main(args):
    style()
    z = np.load(Path(args.run) / "box.npz")
    box = z["box"]
    size = box[2] - box[0]
    fig = plt.figure(figsize=(PAGE, 5.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.05], wspace=0.06, hspace=0.12,
                          left=0.07, right=0.93, top=0.95, bottom=0.08)
    letters = iter("abcdef")
    ims = [None] * 3
    for i, (sensor, label) in enumerate(SENSORS):
        dates = ", ".join(f"{d[:4]}-{d[4:6]}-{d[6:]}" if "-" not in d else d
                          for d in map(str, z[f"{sensor}_dates"]))
        cell = raster(z[f"{sensor}_cell_x"], z[f"{sensor}_cell_y"],
                      np.degrees(z[f"{sensor}_cell_phi"]), box, args.res, args.fill)
        keys = z[f"{sensor}_zone_keys"]
        mu = zones_raster(keys, np.degrees(z[f"{sensor}_zone_mu"]), box, args.zone_m)
        ev = zones_raster(keys, -np.log10(z[f"{sensor}_zone_p"]), box, args.zone_m)
        det = (z[f"{sensor}_zone_p"] < args.pfa).mean()
        panels = ((cell, DIV, -args.phi_max, args.phi_max, "cells"),
                  (mu, DIV, -args.mu_max, args.mu_max, "zones"),
                  (ev, SEQ, 0, args.ev_max, f"zones, $p<P_{{FA}}$: {100 * det:.1f}%"))
        for j, (img, cmap, lo, hi, what) in enumerate(panels):
            ax = fig.add_subplot(gs[i, j])
            ims[j] = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            bare(ax)
            panel_label(ax, f"({next(letters)}) {label}, {what}", bg="white")
            if j == 0:
                ax.set_ylabel(dates, fontsize=6.5, color=MUTED)
                scale_bar(ax, img.shape[1], size)
    labels = (r"cell closure phase (°)", r"zone closure $\hat\mu$ (°)",
              r"evidence $-\log_{10} p$")
    for j, lab in enumerate(labels):
        cax = fig.add_subplot(gs[2, j])
        cb = fig.colorbar(ims[j], cax=cax, orientation="horizontal",
                          extend="both" if j < 2 else "max")
        cb.set_label(lab, color=INK)
        cb.outline.set_edgecolor(MUTED)
        if j == 2:
            cb.set_ticks([0, 1, -np.log10(args.pfa), 3, 4][: int(args.ev_max) + 2])
            cb.ax.axvline(-np.log10(args.pfa), color=POS, lw=1.0)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--res", type=float, default=10.0, help="Raster of the cells (m).")
    parser.add_argument("--fill", type=float, default=2.5, help="Gap filling (pixels).")
    parser.add_argument("--zone-m", type=float, default=100.0)
    parser.add_argument("--pfa", type=float, default=0.01)
    parser.add_argument("--phi-max", type=float, default=60.0)
    parser.add_argument("--mu-max", type=float, default=15.0)
    parser.add_argument("--ev-max", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=Path("zoom.pdf"))
    main(parser.parse_args())
