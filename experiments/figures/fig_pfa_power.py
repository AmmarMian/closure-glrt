r"""Fig. 5 of the paper (one column): false alarm and power of the closure tests.

(a) False-alarm rate per class of the zone coherence for the four tests of ``pfa_power.py``,
measured speckle correlations (the scale-free test also with independent pixels), log scale, with
the nominal rate and the Monte Carlo band. (b) Detection probability of the tests
with a calibrated threshold (scale-free, plug-in ENL, equal weights) against the closure shared by
the zone.

Input: the run directory of ``experiments/detection/pfa_power.py``. Output: a PDF.

Example:
    python experiments/figures/fig_pfa_power.py --run <runs>/runs/<run_id> --out pfa_power.pdf
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLUMN, GRID, INK, MUTED, S1, SAMEDAY, UAVSAR, style

TESTS = {"scale_free": ("scale-free", UAVSAR, "o"), "plugin": ("plug-in ENL", SAMEDAY, "D"),
         "nominal": ("nominal looks", S1, "s"), "equal": ("equal weights", MUTED, "^")}


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main(args):
    style()
    pfa, power = read(Path(args.run) / "pfa.csv"), read(Path(args.run) / "power.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLUMN, 2.05),
                                   gridspec_kw=dict(wspace=0.42, left=0.12, right=0.98,
                                                    top=0.83, bottom=0.17))
    rows = [r for r in pfa if r["window"] == args.window]
    classes = sorted({r["class"] for r in rows}, key=lambda c: float(c.split("-")[0]))
    x = np.arange(len(classes))
    n = int(rows[0]["zones"])
    band = 2 * np.sqrt(args.pfa * (1 - args.pfa) / n)
    ax1.fill_between([-0.4, len(classes) - 0.6], args.pfa - band, args.pfa + band, color=GRID,
                     lw=0)
    ax1.axhline(args.pfa, color=INK, lw=0.6, ls="--")
    for test, (label, col, m) in TESTS.items():
        for speckle, filled in (("measured", True), ("white", False)):
            if speckle == "white" and test != "scale_free":
                continue
            y = [next(float(r["rate"]) for r in rows if r["class"] == c and r["test"] == test
                      and r["speckle"] == speckle) for c in classes]
            ax1.plot(x, y, m, ms=3.0, color=col, mfc=col if filled else "white", mew=0.7,
                     ls="-" if filled else ":", lw=0.6,
                     label=label if filled else f"{label}, indep. pixels")
    ax1.set_yscale("log")
    ax1.set_ylim(2e-3, 0.5)
    ax1.set_xticks(x, [c.split("-")[0].replace("0.", ".") for c in classes])
    ax1.set_xlabel(r"Zone coherence $\bar\gamma$ (lower edge)", labelpad=1)
    ax1.set_ylabel("False-alarm rate", labelpad=1)
    ax1.set_title("(a)", loc="left", fontweight="bold", pad=2)

    for win, ls in zip(args.windows, ("-", "--")):
        for test in ("scale_free", "plugin", "equal"):
            label, col, m = TESTS[test]
            pts = sorted((float(r["closure_deg"]), float(r["rate"])) for r in power
                         if r["window"] == win and r["test"] == test)
            if not pts:
                continue
            mu, rate = np.array(pts).T
            ax2.plot(mu, rate, m, ms=2.6, color=col, ls=ls, lw=0.8)
        ax2.set_xlabel(r"Closure $\mu$ (deg)", labelpad=1)
    ax2.set_ylabel("Detection probability", labelpad=1)
    ax2.set_ylim(0, 1.02)
    ax2.set_title("(b)", loc="left", fontweight="bold", pad=2)
    h, lab = ax1.get_legend_handles_labels()
    fig.legend(h, lab, fontsize=5.2, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.0)
    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.savefig(args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--window", default="15x5", help="Cell size of (a).")
    p.add_argument("--windows", nargs="+", default=["15x5"], help="Cell sizes of (b).")
    p.add_argument("--pfa", type=float, default=0.01)
    p.add_argument("--out", default="pfa_power.pdf")
    main(p.parse_args())
