r"""Fig. 8 of the paper (page width): UAVSAR and Sentinel-1 closures on the same zones.

One row per segment, for one matched triplet each. Maps of the zone closure $\hat\mu$ (100 m zones
coherent in both sensors): colored where the test detects a closure, gray where it does not.
Right: distribution of the half-width $\Delta$ of the $1-\alpha$ confidence arc of each sensor on
these zones (log scale).

Input: the ``zones.npz`` of runs of ``experiments/sentinel1/cross_sensor.py`` (one per segment).

Example:
    uv run --group figures python experiments/figures/fig_xsensor.py \
        --run s1=<runs>/runs/<cross-sensor-s1 run> --run s2=<runs>/runs/<cross-sensor-s2 run> \
        --triplet s1=T035_20170928_20171022_20171115_t65 \
        --triplet s2=T035_20170928_20171022_20171115_t65 --out xsensor.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from style import INK, MUTED, PAGE, S1, UAVSAR, panel_label, style

POS, NEG, MID = "#2a78d6", "#e34948", "#f2f1ec"   # diverging poles (as Fig. 7b), light midpoint
UNDETECTED = "#cfcec8"                            # zone tested, no closure detected
CMAP = LinearSegmentedColormap.from_list("closure", [NEG, MID, POS])


def pairs(items):
    return dict(s.split("=", 1) for s in items)


def grid(keys, values):
    e, n = keys // 1_000_000, keys % 1_000_000
    img = np.full((n.max() - n.min() + 1, e.max() - e.min() + 1), np.nan)
    img[n - n.min(), e - e.min()] = values
    return img


def scale_bar(ax, zone_m, km=10):
    x0, y0 = 0.04, 0.05
    w = km * 1000 / zone_m
    xlim = ax.get_xlim()
    frac = w / (xlim[1] - xlim[0])
    ax.plot([x0, x0 + frac], [y0, y0], transform=ax.transAxes, color=INK, lw=1.2,
            solid_capstyle="butt")
    ax.text(x0 + frac / 2, y0 + 0.02, f"{km} km", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=6.5)


def map_panel(ax, keys, mu, detected, vmax, zone_m):
    deg = np.degrees(mu)
    base = grid(keys, np.where(detected, np.nan, 0.0))
    col = grid(keys, np.where(detected, deg, np.nan))
    ax.imshow(base, origin="lower", cmap=ListedColormap([UNDETECTED]), vmin=0, vmax=1)
    im = ax.imshow(col, origin="lower", cmap=CMAP, vmin=-vmax, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    scale_bar(ax, zone_m)
    return im


def main(args):
    style()
    runs, trip = pairs(args.run), pairs(args.triplet)
    segs = list(runs)
    fig = plt.figure(figsize=(PAGE, 2.25 * len(segs) + 0.45))
    gs = fig.add_gridspec(len(segs) + 1, 3, width_ratios=[1, 1, 0.9],
                          height_ratios=[1] * len(segs) + [0.06], wspace=0.08, hspace=0.3,
                          left=0.06, right=0.98, top=0.95, bottom=0.07)
    letters = iter("abcdefghij")
    bins = np.logspace(np.log10(0.1), np.log10(180), 40)
    for i, seg in enumerate(segs):
        z = np.load(Path(runs[seg]) / "zones.npz")
        name = trip[seg]
        keys = z[f"{name}__keys"]
        title = f"{seg}, $t={name.rsplit('_t', 1)[1]}$, {name.split('_')[0]}"
        for j, (sensor, label) in enumerate((("uavsar", "UAVSAR (L)"), ("s1", "Sentinel-1 (C)"))):
            ax = fig.add_subplot(gs[i, j])
            det = z[f"{name}__p_{sensor}"] < args.pfa
            im = map_panel(ax, keys, z[f"{name}__mu_{sensor}"], det, args.vmax, args.zone_m)
            panel_label(ax, f"({next(letters)}) {label}, {title}", bg="white")
            ax.text(0.99, 0.05, f"detected {100 * det.mean():.0f}%", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6.5, color=INK)
        ax = fig.add_subplot(gs[i, 2])
        for sensor, color, label in (("uavsar", UAVSAR, "UAVSAR"), ("s1", S1, "Sentinel-1")):
            h = np.degrees(z[f"{name}__h_{sensor}"])
            h = h[np.isfinite(h)]
            ax.hist(np.clip(h, bins[0], bins[-1]), bins=bins, histtype="step", color=color,
                    lw=1.2, weights=np.full(len(h), 100 / len(h)))
            med = np.median(h)
            ax.axvline(med, color=color, lw=0.8, ls="--")
            left = sensor == "uavsar"
            ax.text(0.02 if left else 0.98, 0.97, f"{label}\nmedian {med:.1f}°",
                    transform=ax.transAxes, ha="left" if left else "right", va="top",
                    color=INK, fontsize=6.5)
        ax.set_xscale("log")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xticks([0.1, 1, 10, 180], ["0.1", "1", "10", "180"])
        ax.set_ylabel("zones (%)")
        if i == len(segs) - 1:
            ax.set_xlabel(r"half-width $\Delta$ of the 95% arc (°)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"({next(letters)}) arc half-width, {seg}", fontweight="bold", loc="left")
    cax = fig.add_subplot(gs[-1, :2])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label(r"detected closure $\hat\mu$ (°); gray: not detected", color=INK)
    cb.outline.set_edgecolor(MUTED)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", required=True, help="SEG=run directory")
    parser.add_argument("--triplet", nargs="+", required=True, help="SEG=triplet name")
    parser.add_argument("--pfa", type=float, default=0.01)
    parser.add_argument("--vmax", type=float, default=3.0, help="Color limit (deg).")
    parser.add_argument("--zone-m", type=float, default=100.0)
    parser.add_argument("--out", type=Path, default=Path("xsensor.pdf"))
    main(parser.parse_args())
