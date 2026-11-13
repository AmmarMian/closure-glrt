r"""Coverage figure of the TGRS paper (two columns): coverage of the 95% intervals (top) and bias
of the estimate of the closure (bottom) against the true closure, per class of zone coherence.

Input: run directories of ``experiments/detection/coverage.py`` (scenario touching; the run with
the Yuan table adds the two tests of the literature). Output: a PDF.

Example:
    python experiments/figures/fig_coverage.py --runs <run1> <run2> --window 15x5 --out cov.pdf
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import GRID, INK, MUTED, PAGE, S1, SAMEDAY, UAVSAR, style

METHODS = {
    "scale_free": ("scale-free, own precisions", UAVSAR, "o", "--"),
    "scale_free_nb": ("scale-free, neighbor precisions", UAVSAR, "o", "-"),
    "plugin": ("plug-in ENL", SAMEDAY, "D", "-"),
    "equal": ("equal weights", MUTED, "^", "-"),
    "bootstrap": ("bootstrap", "#8a5cc2", "v", "-"),
    "zwieback": ("Zwieback (zone)", S1, "s", "-"),
    "yuan": ("Yuan (zone)", "#b0325a", "P", "-"),
}
CLASSES = ("0.5-0.7", "0.7-0.9", "0.9-1.01")
LABELS = {"0.5-0.7": r"$0.5\leq\bar\gamma<0.7$", "0.7-0.9": r"$0.7\leq\bar\gamma<0.9$",
          "0.9-1.01": r"$\bar\gamma\geq0.9$"}


def read(runs, window):
    rows = {}
    for r in runs:
        with open(Path(r) / "coverage.csv") as f:
            for x in csv.DictReader(f):
                if x["window"] == window and x["scenario"] == "touching":
                    rows[(x["method"], x["class"], float(x["closure_deg"]))] = x
    return rows


def main(args):
    style()
    rows = read(args.runs, args.window)
    closures = sorted({k[2] for k in rows})
    xs = np.arange(len(closures))
    fig, axes = plt.subplots(2, 3, figsize=(PAGE, 3.3), sharex=True, sharey="row")
    for j, c in enumerate(CLASSES):
        for m, (lab, col, mk, ls) in METHODS.items():
            pts = [(i, rows[(m, c, mu)]) for i, mu in enumerate(closures) if (m, c, mu) in rows]
            if not pts:
                continue
            i = np.array([p[0] for p in pts])
            cov = np.array([float(p[1]["coverage"]) for p in pts])
            bias = np.array([float(p[1]["mean_error_deg"]) for p in pts])
            mu = np.array([closures[k] for k in i])
            rel = np.where(mu >= 1, bias / np.where(mu > 0, mu, 1), np.nan)   # noisy below 1 deg
            kw = dict(color=col, marker=mk, ls=ls, lw=0.9, ms=2.8, label=lab)
            axes[0, j].plot(i, cov, **kw)
            axes[1, j].plot(i, 100 * rel, **kw)
        axes[0, j].axhline(0.95, color=INK, lw=0.6, ls=":")
        axes[1, j].axhline(0.0, color=INK, lw=0.6, ls=":")
        axes[0, j].set_title(f"({'abc'[j]}) " + LABELS[c])
        axes[1, j].set_title(f"({'def'[j]}) " + LABELS[c])
        for a in axes[:, j]:
            a.grid(color=GRID, lw=0.4)
            a.set_xticks(xs, [f"{v:g}" for v in closures])
        axes[1, j].set_xlabel(r"true closure $\mu$ (deg)")
    axes[0, 0].set_ylabel("coverage of the 95% interval")
    axes[0, 0].yaxis.set_label_coords(-0.14, 0.5)
    axes[1, 0].yaxis.set_label_coords(-0.14, 0.5)
    axes[1, 0].set_ylabel(r"bias of $\hat\mu$ (% of $\mu$)")
    axes[0, 0].set_ylim(0, 1.02)
    axes[1, 0].set_ylim(-30, 10)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(args.out, bbox_inches="tight")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--window", default="15x5")
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
