r"""Fig. 7 of the paper (one column): closures along the series and by crop.

(a) Detected share $\rho_t$ per triplet for the coherent zone classes (one blue step per class,
light to dark with the coherence), with the nominal rate; the same-day triplets are shaded.
(b) By crop (CDL), for two triplets: share of the zones detected with a negative closure (left,
red) and with a positive closure (right, blue).

Inputs: a run of ``experiments/series/closure_weather.py`` (``triplets.csv``: shares per triplet
and class of the zone coherence from independent pairs) and a run of
``experiments/series/closure_landcover.py`` (``by_class.csv``) of the same segment.

Example:
    uv run --group figures python experiments/figures/fig_series.py \
        --weather <runs>/runs/<closure-weather run> --landcover <runs>/runs/<landcover run> \
        --out series.pdf
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLUMN, GRID, INK, MUTED, panel_label, style

CLASS_COLORS = ("#86b6ef", "#2a78d6", "#104281")   # ordinal blue ramp, validated on white
POS, NEG = "#2a78d6", "#e34948"                     # diverging poles: closure > 0, < 0
SHORT = {"other tree crops": "tree crops", "grass/pasture": "grass, pasture",
         "fallow/idle": "fallow", "developed open": "developed"}


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def num(s):
    return float(s) if s not in ("", None) else np.nan


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(args):
    style()
    rows = read(Path(args.weather) / "triplets.csv")
    t = np.array([int(r["triplet"]) for r in rows])
    sameday = np.array([r["date1"] == r["date2"] == r["date3"] for r in rows])
    classes = [k[len("share_"):] for k in rows[0] if k.startswith("share_")]

    fig = plt.figure(figsize=(COLUMN, 3.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.75], hspace=0.38,
                             left=0.2, right=0.98, top=0.91, bottom=0.09)
    gs = outer[1].subgridspec(1, 2, wspace=0.08)
    ax_s = fig.add_subplot(outer[0])
    for k in t[sameday]:
        ax_s.axvspan(k - 0.5, k + 0.5, color=GRID, lw=0, zorder=0)
    despine(ax_s)

    ax_s.axhline(args.pfa, color=INK, lw=0.6, ls="--", zorder=1)
    for c, col in zip(classes, CLASS_COLORS):
        y = np.array([num(r[f"share_{c}"]) if num(r[f"zones_{c}"]) >= args.min_zones_series
                      else np.nan for r in rows])
        lo, hi = c.split("-")
        label = rf"$\bar\gamma^\perp\in[{lo},{min(float(hi), 1):g}]$"
        ax_s.plot(t, y, "-o", color=col, lw=0.9, ms=1.8, label=label, zorder=2)
    ax_s.set_ylim(0, None)
    ax_s.set_ylabel(r"Detected share $\rho_t$", labelpad=1)
    ax_s.legend(fontsize=5.5, frameon=False, loc="lower left", ncol=3, handlelength=1.5,
                columnspacing=1.0, borderaxespad=0.1, bbox_to_anchor=(0.0, 1.0))
    panel_label(ax_s, "(a)", corner="right")
    ax_s.set_xlabel(r"Triplet $t$", labelpad=1)
    ax_s.set_xlim(t[0] - 0.7, t[-1] + 0.7)

    lc = [r for r in read(Path(args.landcover) / "by_class.csv")
          if int(r["triplet"]) in args.triplets]
    by = {(int(r["triplet"]), r["class"]): r for r in lc}
    crops = [c for c in sorted({r["class"] for r in lc})
             if all((k, c) in by and int(by[k, c]["zones"]) >= args.min_zones
                    for k in args.triplets)]
    first = args.triplets[0]
    crops.sort(key=lambda c: float(by[first, c]["frac_detected_pos"])
               - float(by[first, c]["frac_detected_neg"]))
    y = np.arange(len(crops))
    lim = 1.08 * max(max(float(by[k, c]["frac_detected_pos"]), float(by[k, c]["frac_detected_neg"]))
                     for k in args.triplets for c in crops)
    for j, k in enumerate(args.triplets):
        ax = fig.add_subplot(gs[j])
        pos = [float(by[k, c]["frac_detected_pos"]) for c in crops]
        neg = [float(by[k, c]["frac_detected_neg"]) for c in crops]
        ax.barh(y, pos, height=0.72, color=POS, lw=0)
        ax.barh(y, [-v for v in neg], height=0.72, color=NEG, lw=0)
        ax.axvline(0, color=MUTED, lw=0.5)
        ax.set_xlim(-lim, lim)
        ticks = [v for v in (-0.5, -0.25, 0, 0.25, 0.5) if abs(v) <= lim]
        ax.set_xticks(ticks, [f"{abs(v):g}" for v in ticks])
        ax.set_ylim(-0.6, len(crops) - 0.4)
        ax.set_yticks(y, [SHORT.get(c, c) for c in crops] if j == 0 else [])
        ax.tick_params(axis="y", length=0)
        d = by[k, crops[0]]["dates"].split("/")
        ax.set_title(f"$t={k}$: {d[0][2:]}, {d[1][5:]}, {d[2][5:]}", fontsize=6, pad=2)
        despine(ax)
        if j == 0:
            fig.text(0.01, ax.get_position().y1 + 0.012, "(b)", fontweight="bold", fontsize=7)
    fig.text(0.59, 0.012, r"Share of zones detected with $\hat\mu<0$ (red) and $\hat\mu>0$ (blue)",
             ha="center", fontsize=6.5, color=INK)
    fig.savefig(args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weather", required=True, help="Run of closure_weather.py.")
    p.add_argument("--landcover", required=True, help="Run of closure_landcover.py.")
    p.add_argument("--triplets", type=int, nargs=2, default=[12, 13])
    p.add_argument("--min-zones", type=int, default=1000, help="Crops kept in (d).")
    p.add_argument("--min-zones-series", type=int, default=1000,
                   help="Triplets of a class with fewer zones are left out of (a).")
    p.add_argument("--pfa", type=float, default=0.01)
    p.add_argument("--out", default="series.pdf")
    main(p.parse_args())
