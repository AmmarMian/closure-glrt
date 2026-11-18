r"""UAVSAR and Sentinel-1 closures on the same zones (TGRS paper): two figures.

* ``--out-maps`` (page width): maps of the zone closure $\hat\mu$ of each sensor (columns) and
  segment (rows) for one matched triplet, on the 100 m zones coherent in both sensors, colored
  where the test detects a closure and gray where it does not. The maps are rotated so that the
  UAVSAR flight line is horizontal (angle of the principal axis of the zones of all segments);
  north arrow and scale bar give the orientation and the size.
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
from scipy import ndimage
from style import COLUMN, GRID, INK, MUTED, PAGE, S1, UAVSAR, style

NEG, MID, POS = "#c0392b", "#f4f3ee", "#1f5fae"
UNDETECTED = "#c9c8c2"                         # zone tested, no closure detected
CMAP = LinearSegmentedColormap.from_list("closure", [NEG, "#ef9a8a", MID, "#8db7e6", POS])
SENSORS = (("uavsar", "UAVSAR (L)"), ("s1", "Sentinel-1 (C)"))


def pairs(items):
    return dict(s.split("=", 1) for s in items)


def coords(keys):
    return (keys // 1_000_000).astype(float), (keys % 1_000_000).astype(float)


def strip_angle(all_keys):
    """Angle (rad) of the flight line: mean over the segments of the principal axis of their zone
    centres, each folded to (-90, 90] degrees."""
    out = []
    for keys in all_keys:
        e, n = coords(keys)
        v = np.linalg.eigh(np.cov(np.c_[e - e.mean(), n - n.mean()].T))[1][:, -1]
        out.append((np.arctan2(v[1], v[0]) + np.pi / 2) % np.pi - np.pi / 2)
    return float(np.mean(out))


def rotated_grid(keys, values, angle_deg):
    """Raster of ``values`` on the zone grid (north up), rotated by ``angle_deg`` (nearest)."""
    e, n = coords(keys)
    e, n = (e - e.min()).astype(int), (n - n.min()).astype(int)
    img = np.full((n.max() + 1, e.max() + 1), np.nan)
    img[n.max() - n, e] = values                  # row 0 = north
    out = ndimage.rotate(np.nan_to_num(img, nan=-999.0), -angle_deg, order=0, reshape=True,
                         cval=-999.0)
    out[out == -999.0] = np.nan
    return out


def crop(*imgs):
    ok = np.isfinite(imgs[0])
    for im in imgs[1:]:
        ok |= np.isfinite(im)
    r, c = np.flatnonzero(ok.any(1)), np.flatnonzero(ok.any(0))
    return [im[r[0]:r[-1] + 1, c[0]:c[-1] + 1] for im in imgs]


def decorate(ax, angle_deg, zone_m, km=10):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    w = km * 1000 / zone_m
    x0, x1 = ax.get_xlim()
    frac = w / abs(x1 - x0)
    ax.plot([0.03, 0.03 + frac], [0.07, 0.07], transform=ax.transAxes, color=INK, lw=1.3,
            solid_capstyle="butt")
    ax.text(0.03 + frac / 2, 0.1, f"{km} km", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7)
    a = np.radians(90 - angle_deg)                # north after the rotation of the map
    cx, cy, L = 0.95, 0.2, 0.09
    ax.annotate("", xy=(cx + L * np.cos(a) * 0.55, cy + L * np.sin(a)), xytext=(cx, cy),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=6))
    ax.text(cx + L * np.cos(a) * 0.75, cy + L * np.sin(a) * 1.3, "N", transform=ax.transAxes,
            ha="center", va="center", fontsize=7)


def maps(args, runs, trip):
    segs = list(runs)
    data = {s: np.load(Path(runs[s]) / "zones.npz") for s in segs}
    angle = np.degrees(strip_angle([data[s][f"{trip[s]}__keys"] for s in segs]))
    norm = BoundaryNorm(np.arange(-args.vmax, args.vmax + 1e-9, 0.5), CMAP.N, extend="both")
    panels = []
    for s in segs:
        z, name = data[s], trip[s]
        keys = z[f"{name}__keys"]
        row = []
        for sensor, _ in SENSORS:
            det = z[f"{name}__p_{sensor}"] < args.pfa
            mu = np.degrees(z[f"{name}__mu_{sensor}"])
            row += [rotated_grid(keys, np.where(det, np.nan, 0.0), angle),
                    rotated_grid(keys, np.where(det, mu, np.nan), angle), det.mean()]
        g = crop(row[0], row[1], row[3], row[4])
        panels.append((s, name, [(g[0], g[1], row[2]), (g[2], g[3], row[5])]))
    heights = [p[2][0][0].shape[0] / p[2][0][0].shape[1] for p in panels]
    fig = plt.figure(figsize=(PAGE, PAGE / 2.25 * sum(heights) + 0.5))
    gs = fig.add_gridspec(len(segs), 3, width_ratios=[1, 1, 0.035], height_ratios=heights,
                          wspace=0.04, hspace=0.14, left=0.01, right=0.93, top=0.95,
                          bottom=0.09)
    letters = iter("abcdefgh")
    for i, (s, name, row) in enumerate(panels):
        for j, ((base, col, share), (_, label)) in enumerate(zip(row, SENSORS)):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(np.where(np.isfinite(base), 0.0, np.nan),
                      cmap=LinearSegmentedColormap.from_list("u", [UNDETECTED, UNDETECTED]),
                      vmin=0, vmax=1, origin="upper")
            im = ax.imshow(col, cmap=CMAP, norm=norm, origin="upper")
            decorate(ax, angle, args.zone_m)
            ax.set_title(f"({next(letters)}) {label}, segment {s}",
                         loc="left", fontweight="bold", fontsize=7, pad=3)
            ax.set_title(f"detected: {100 * share:.0f}%", loc="right", fontsize=7, pad=3,
                         color=INK)
    cax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(im, cax=cax, orientation="vertical", extend="both",
                      ticks=np.arange(-args.vmax, args.vmax + 1e-9, 1.0))
    cb.set_label(r"detected closure $\hat\mu$ (°)", color=INK)
    cb.outline.set_edgecolor(MUTED)
    cb.ax.tick_params(length=2)
    fig.legend(handles=[Patch(fc=UNDETECTED, ec="none", label="tested, not detected")],
               loc="lower left", bbox_to_anchor=(0.005, 0.0), frameon=False, fontsize=7,
               handlelength=1.0)
    fig.savefig(args.out_maps)
    print("wrote", args.out_maps)


def widths(args, runs, trip):
    segs = list(runs)
    fig, axes = plt.subplots(len(segs), 1, figsize=(COLUMN, 1.35 * len(segs) + 0.3), sharex=True)
    bins = np.logspace(np.log10(0.1), np.log10(180), 40)
    for i, (s, ax) in enumerate(zip(segs, np.atleast_1d(axes))):
        z, name = np.load(Path(runs[s]) / "zones.npz"), trip[s]
        for sensor, color, label in (("uavsar", UAVSAR, "UAVSAR"), ("s1", S1, "Sentinel-1")):
            h = np.degrees(z[f"{name}__h_{sensor}"])
            h = h[np.isfinite(h)]
            ax.hist(np.clip(h, bins[0], bins[-1]), bins=bins, histtype="stepfilled", alpha=0.18,
                    color=color, weights=np.full(len(h), 100 / len(h)))
            med = np.median(h)
            ax.hist(np.clip(h, bins[0], bins[-1]), bins=bins, histtype="step", color=color,
                    lw=1.1, weights=np.full(len(h), 100 / len(h)),
                    label=f"{label}, median {med:.1f}°")
            ax.axvline(med, color=color, lw=0.8, ls="--")
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
    parser.add_argument("--out-maps", type=Path, default=None)
    parser.add_argument("--out-widths", type=Path, default=None)
    main(parser.parse_args())
