r"""UAVSAR and Sentinel-1 closures on the same zones (TGRS paper): two figures.

* ``--out-maps`` (page width): maps of the zone closure $\hat\mu$ of each sensor (columns) and
  segment (rows) for one matched triplet, on the 100 m zones coherent in both sensors, colored
  where the test detects a closure and gray where it does not. The maps are rotated so that the
  UAVSAR flight line is horizontal (angle of the principal axis of the zones of all segments);
  north arrow and scale bar give the orientation and the size.
* ``--data DIR``: tables and image layers of the pgfplots versions (``map_<seg>_<sensor>.png``,
  ``hist_<seg>.dat``, ``labels.tex`` with extents, shares, medians and the colormap).
* ``--out-widths`` (one column): distribution of the half-width $\Delta$ of the $1-\alpha$
  confidence arc of each sensor on the same zones, one panel per segment (log scale).

Input: the ``zones.npz`` of runs of ``experiments/sentinel1/cross_sensor.py`` (one per segment).

Example:
    uv run --group figures python experiments/figures/fig_xsensor.py \
        --run s1=<runs>/runs/<cross-sensor-s1 run> --run s2=<runs>/runs/<cross-sensor-s2 run> \
        --triplet s1=T035_20170928_20171022_20171115_t65 \
        --triplet s2=T035_20170928_20171022_20171115_t65 \
        --out-maps xsensor_maps.pdf --out-widths xsensor_widths.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Patch
from pgfdata import write_dat, write_macros
from style import COLUMN, GRID, INK, MUTED, PAGE, S1, UAVSAR, style

NEG, MID, POS = "#c0392b", "#f4f3ee", "#1f5fae"
UNDETECTED = "#b9b8b2"                         # zone tested, no closure detected
SWATH = "#f1f0eb"                              # inside the UAVSAR swath, not a common zone
AZ_M, RG_M = 0.6, 1.6655                       # UAVSAR SLC pixel spacing (azimuth, slant range)
CMAP = LinearSegmentedColormap.from_list("closure", [NEG, "#ef9a8a", MID, "#8db7e6", POS])
SENSORS = (("uavsar", "UAVSAR (L)"), ("s1", "Sentinel-1 (C)"))


def pairs(items):
    return dict(s.split("=", 1) for s in items)







def lookup(grid_keys, keys, values):
    """Values of the zones ``keys`` at the positions of ``grid_keys`` (nan where absent)."""
    order = np.argsort(keys)
    sk = keys[order]
    pos = np.clip(np.searchsorted(sk, grid_keys), 0, len(sk) - 1)
    found = (sk[pos] == grid_keys) & (grid_keys >= 0)
    out = np.full(grid_keys.shape, np.nan)
    out[found] = values[order][pos[found]]
    return out


def export_maps(folder, panels, norm, vmax):
    """PNG layer per panel (swath, undetected zones, closure colors; transparent elsewhere, row 0
    at range 0) and the macros of the pgfplots maps: extents (km), shares, colormap."""
    from matplotlib.colors import to_rgba
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    macros = {}
    for i, (s, ext, row, _) in enumerate(panels):
        seg = "ab"[i]
        macros[f"xmapaz{seg}"] = f"{ext[1]:.3f}"
        macros[f"xmaprg{seg}"] = f"{ext[2]:.3f}"
        for (swath, tested, closure, share), (sensor, _) in zip(row, SENSORS):
            rgba = np.zeros(swath.shape + (4,))
            rgba[swath] = to_rgba(SWATH)
            rgba[np.isfinite(tested)] = to_rgba(UNDETECTED)
            m = np.isfinite(closure)
            rgba[m] = CMAP(norm(closure[m]))
            plt.imsave(folder / f"map_{s}_{sensor}.png", rgba)
            macros[f"xmapshare{seg}{'u' if sensor == 'uavsar' else 's'}"] = f"{100 * share:.0f}"
    levels = norm.boundaries
    cols = [CMAP(norm((a + b) / 2)) for a, b in zip(levels[:-1], levels[1:])]
    macros["xmapcolormap"] = " ".join(
        "rgb255=({},{},{})".format(*(round(255 * v) for v in c[:3])) for c in cols)
    macros["xmapvmax"] = f"{vmax:g}"
    macros["xmapnlevels"] = str(len(cols))
    write_macros(folder, "labels", **macros)


def maps(args, runs, trip):
    """Zone results drawn on the UAVSAR radar grid (``radar_zone_keys.py``): azimuth along the
    horizontal axis, slant range along the vertical axis, as the amplitude images."""
    segs = list(runs)
    rk = pairs(args.radar_keys)
    norm = BoundaryNorm(np.arange(-args.vmax, args.vmax + 1e-9, 0.5), CMAP.N, extend="both")
    panels = []
    for s in segs:
        z, name = np.load(Path(runs[s]) / "zones.npz"), trip[s]
        r = np.load(rk[s])
        K = r["keys"].T                                    # rows: range, columns: azimuth
        step_az, step_rg = (float(v) for v in r["step"])
        ext = (0, K.shape[1] * step_az * AZ_M / 1e3, K.shape[0] * step_rg * RG_M / 1e3, 0)
        keys = z[f"{name}__keys"]
        row = []
        for sensor, _ in SENSORS:
            det = z[f"{name}__p_{sensor}"] < args.pfa
            mu = np.degrees(z[f"{name}__mu_{sensor}"])
            tested = lookup(K, keys, np.zeros(len(keys)))
            closure = lookup(K, keys, np.where(det, mu, np.nan))
            row.append((K >= 0, tested, closure, det.mean()))
        box = None
        if args.box:
            x0, x1, y0, y1 = (v / args.zone_m for v in args.box)
            e, n = K // 1_000_000, K % 1_000_000
            box = (K >= 0) & (e >= x0) & (e < x1) & (n >= y0) & (n < y1)
        panels.append((s, ext, row, box))
    aspect = (panels[0][1][2]) / panels[0][1][1]
    fig = plt.figure(figsize=(PAGE, 2 * (PAGE * 0.46) * aspect * len(segs) / 2 + 0.75))
    gs = fig.add_gridspec(len(segs), 3, width_ratios=[1, 1, 0.03], wspace=0.07, hspace=0.45,
                          left=0.055, right=0.935, top=0.93, bottom=0.17)
    if args.data:
        export_maps(args.data, panels, norm, args.vmax)
    letters = iter("abcdefgh")
    for i, (s, ext, row, box) in enumerate(panels):
        for j, ((swath, tested, closure, share), (_, label)) in enumerate(zip(row, SENSORS)):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(np.where(swath, 0.0, np.nan), extent=ext, aspect="equal",
                      cmap=LinearSegmentedColormap.from_list("s", [SWATH, SWATH]), vmin=0, vmax=1)
            ax.imshow(np.where(np.isfinite(tested), 0.0, np.nan), extent=ext, aspect="equal",
                      cmap=LinearSegmentedColormap.from_list("u", [UNDETECTED, UNDETECTED]),
                      vmin=0, vmax=1)
            im = ax.imshow(closure, extent=ext, aspect="equal", cmap=CMAP, norm=norm)
            if box is not None and box.any():
                ax.contour(box.astype(float), levels=[0.5], colors=INK, linewidths=1.0,
                           extent=(ext[0], ext[1], ext[3], ext[2]), origin="upper")
            ax.set_title(f"({next(letters)}) {label}, segment {s}", loc="left",
                         fontweight="bold", pad=3)
            ax.set_title(f"detected: {100 * share:.0f}%", loc="right", pad=3)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            ax.tick_params(length=2)
            if i == len(panels) - 1:
                ax.set_xlabel("azimuth (km)")
            if j == 0:
                ax.set_ylabel("range (km)")
            else:
                ax.set_yticklabels([])
    cax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(im, cax=cax, orientation="vertical", extend="both",
                      ticks=np.arange(-args.vmax, args.vmax + 1e-9, 1.0))
    cb.set_label(r"detected closure $\hat\mu$ (°)", color=INK)
    cb.outline.set_edgecolor(MUTED)
    cb.ax.tick_params(length=2)
    fig.legend(handles=[Patch(fc=UNDETECTED, ec="none", label="zone tested, no closure detected"),
                        Patch(fc=SWATH, ec=GRID, lw=0.5,
                              label="UAVSAR swath, zone not coherent in both sensors")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
               handlelength=1.2)
    fig.savefig(args.out_maps)
    print("wrote", args.out_maps)


def widths(args, runs, trip):
    segs = list(runs)
    fig, axes = plt.subplots(len(segs), 1, figsize=(COLUMN, 1.35 * len(segs) + 0.3), sharex=True)
    bins = np.logspace(np.log10(0.1), np.log10(180), 40)
    medians, ymax_all = {}, 0.0
    for i, (s, ax) in enumerate(zip(segs, np.atleast_1d(axes))):
        z, name = np.load(Path(runs[s]) / "zones.npz"), trip[s]
        hist = {"lo": bins[:-1], "hi": bins[1:]}
        for sensor, color, label in (("uavsar", UAVSAR, "UAVSAR"), ("s1", S1, "Sentinel-1")):
            h = np.degrees(z[f"{name}__h_{sensor}"])
            h = h[np.isfinite(h)]
            hist[sensor] = np.histogram(np.clip(h, bins[0], bins[-1]), bins=bins)[0] * 100 / len(h)
            ymax_all = max(ymax_all, hist[sensor].max())
            medians[f"xwmed{'ab'[i]}{'u' if sensor == 'uavsar' else 's'}"] = f"{np.median(h):.1f}"
            ax.hist(np.clip(h, bins[0], bins[-1]), bins=bins, histtype="stepfilled", alpha=0.18,
                    color=color, weights=np.full(len(h), 100 / len(h)))
            med = np.median(h)
            ax.hist(np.clip(h, bins[0], bins[-1]), bins=bins, histtype="step", color=color,
                    lw=1.1, weights=np.full(len(h), 100 / len(h)),
                    label=f"{label}, median {med:.1f}°")
            ax.axvline(med, color=color, lw=0.8, ls="--")
        write_dat(args.data, f"hist_{s}", **hist)
        ax.set_xscale("log")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xticks([0.1, 1, 10, 180], ["0.1", "1", "10", "180"])
        ax.set_ylabel("zones (%)")
        ax.grid(color=GRID, lw=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"({'abcd'[i]}) segment {s}", loc="left", fontweight="bold")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
        ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=1)
    write_macros(args.data, "widths", xwymax=f"{1.45 * ymax_all:.1f}", **medians)
    np.atleast_1d(axes)[-1].set_xlabel(r"half-width $\Delta$ of the 95% arc (°)")
    fig.tight_layout()
    fig.savefig(args.out_widths)
    print("wrote", args.out_widths)


def main(args):
    style()
    runs, trip = pairs(args.run), pairs(args.triplet)
    if args.out_maps:
        maps(args, runs, trip)
    if args.out_widths:
        widths(args, runs, trip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", required=True, help="SEG=run directory")
    parser.add_argument("--triplet", nargs="+", required=True, help="SEG=triplet name")
    parser.add_argument("--pfa", type=float, default=0.01)
    parser.add_argument("--vmax", type=float, default=3.0, help="Color limit (deg).")
    parser.add_argument("--zone-m", type=float, default=100.0)
    parser.add_argument("--radar-keys", nargs="+", default=None,
                        help="SEG=radar_keys.npz of radar_zone_keys.py (needed for the maps).")
    parser.add_argument("--box", type=float, nargs=4, default=None,
                        help="UTM box (xmin xmax ymin ymax, m) outlined on the maps.")
    parser.add_argument("--out-maps", type=Path, default=None)
    parser.add_argument("--out-widths", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None,
                        help="Folder of the pgfplots tables and image layers (optional).")
    main(parser.parse_args())
