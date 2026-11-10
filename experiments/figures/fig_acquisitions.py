r"""Fig. 2 of the paper (one column): acquisitions and weather of segment 1.

(a) Dates of the 68 UAVSAR acquisitions and of the Sentinel-1 acquisitions of the matched
triplets, the same-day triplets labelled; (b) daily rain (mean of the GHCN stations); (c)
volumetric soil moisture of the top 7 cm (ERA5).

Input for now: the context store of the legacy run ``context-maps-s1`` (dates, weather series),
until that experiment is ported. Output: a PDF.

Example:
    python experiments/figures/fig_acquisitions.py --context <run>/context.zarr \
        --s1-triplets data/s1_triplets.csv --out acquisitions.pdf
"""

import argparse
import csv
from datetime import date, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import zarr

from style import COLUMN, GRID, INK, MUTED, S1, SAMEDAY, UAVSAR, panel_label, style


def main(args):
    style()
    g = zarr.open_group(args.context, mode="r")
    a = dict(g.attrs)
    acq = [date.fromisoformat(x["date"]) for x in a["acquisitions"]]
    with open(args.s1_triplets) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    s1 = sorted({date(int(d[:4]), int(d[4:6]), int(d[6:])) for r in rows
                 for d in (r["d1"], r["d2"], r["d3"]) if int(r["segment"]) == 1})
    d0 = date.fromisoformat(a["weather_days"][0])
    rain = g["weather/rain_ghcn"][:]
    sm = g["weather/sm0_7"][:]
    days = np.array([d0 + timedelta(days=int(i)) for i in range(len(rain))])

    fig = plt.figure(figsize=(COLUMN, 2.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1, 1], hspace=0.2,
                          left=0.15, right=0.97, top=0.95, bottom=0.1)
    ax1 = fig.add_subplot(gs[0])
    ax1.vlines(acq, 0.55, 0.95, color=UAVSAR, lw=0.7)
    ax1.vlines(s1, 0.05, 0.45, color=S1, lw=0.7)
    for label, idx in (("$t=10$", 10), ("$t=23$", 23)):
        ax1.vlines(acq[idx], 0.55, 0.95, color=SAMEDAY, lw=1.6)
        ax1.text(acq[idx], 1.0, label, fontsize=6, ha="center", va="bottom", color=INK)
    ax1.set_ylim(0, 1.35)
    ax1.set_yticks([0.75, 0.25], ["UAVSAR", "S1"])
    ax1.tick_params(axis="y", length=0)
    panel_label(ax1, "(a)", corner="right")

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.bar(days, rain, width=1.0, color=MUTED, lw=0)
    ax2.set_ylabel("Rain\n(mm/day)", labelpad=1)
    panel_label(ax2, "(b)", corner="right")

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(days, sm, color=INK, lw=0.8)
    ax3.set_ylim(0.05, 0.5)
    ax3.set_ylabel("Soil moist.\n(m$^3$/m$^3$)", labelpad=1)
    panel_label(ax3, "(c)", corner="right")
    for axx in (ax1, ax2, ax3):
        for s in ("top", "right"):
            axx.spines[s].set_visible(False)
        axx.grid(axis="x", color=GRID, lw=0.4)
        axx.set_xlim(date(2009, 6, 1), date(2018, 1, 1))
    for axx in (ax1, ax2):
        plt.setp(axx.get_xticklabels(), visible=False)
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(args.out)
    print("UAVSAR dates", len(acq), "S1 dates", len(s1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--context", required=True)
    p.add_argument("--s1-triplets", required=True)
    p.add_argument("--out", default="acquisitions.pdf")
    main(p.parse_args())
