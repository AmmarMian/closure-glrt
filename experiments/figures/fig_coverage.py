r"""Coverage figure of the TGRS paper (two columns): coverage of the 95% arc (top) and bias of the
estimate of the closure in percent of the closure (bottom) against the true closure, per class of
zone coherence, for the arc with own and with neighbor precisions. The bias is shown at every
nonzero closure with its 2-sigma Monte Carlo error (rmse / sqrt(zones)).

Input: run directories of ``experiments/detection/coverage.py`` (scenario touching). Output: a PDF.

Example:
    python experiments/figures/fig_coverage.py --runs <run1> <run2> --window 15x5 --out cov.pdf
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pgfdata import write_dat
from style import GRID, INK, PAGE, S1, UAVSAR, style

METHODS = {
    "scale_free": (r"own precisions $\kappa_{0,\ell}$", S1, "s", "--"),
    "scale_free_nb": (r"neighbor precisions $\tilde\kappa_{0,\ell}$", UAVSAR, "o", "-"),
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
    tables = {}
    fig, axes = plt.subplots(2, 3, figsize=(PAGE, 3.3), sharex=True, sharey="row")
    for j, c in enumerate(CLASSES):
        for m, (lab, col, mk, ls) in METHODS.items():
            pts = [(i, rows[(m, c, mu)]) for i, mu in enumerate(closures) if (m, c, mu) in rows]
            if not pts:
                continue
            i = np.array([p[0] for p in pts])
            cov = np.array([float(p[1]["coverage"]) for p in pts])
            bias = np.array([float(p[1]["mean_error_deg"]) for p in pts])
            se = np.array([float(p[1]["rmse_deg"]) / np.sqrt(float(p[1]["zones"])) for p in pts])
            mu = np.array([closures[k] for k in i])
            pos = mu > 0
            tables[(m, c)] = dict(idx=i, closure=mu, coverage=cov,
                                  bias=np.where(pos, 100 * bias / np.where(pos, mu, 1), np.nan),
                                  err=np.where(pos, 200 * se / np.where(pos, mu, 1), np.nan))
            kw = dict(color=col, marker=mk, ls=ls, lw=0.9, ms=3, label=lab)
            axes[0, j].plot(i, cov, **kw)
            off = -0.08 if m == "scale_free" else 0.08            # side by side error bars
            axes[1, j].errorbar(i[pos] + off, 100 * bias[pos] / mu[pos],
                                yerr=200 * se[pos] / mu[pos], capsize=1.5, elinewidth=0.6, **kw)
        axes[0, j].axhline(0.95, color=INK, lw=0.6, ls=":")
        if c == "0.9-1.01":
            axes[0, j].text(len(closures) - 1, 0.5, "not\nfeasible", ha="center", va="center",
                            fontsize=7, color="#6b6a64")
        axes[1, j].axhline(0.0, color=INK, lw=0.6, ls=":")
        axes[0, j].set_title(f"({'abc'[j]}) " + LABELS[c])
        axes[1, j].set_title(f"({'def'[j]}) " + LABELS[c])
        for a in axes[:, j]:
            a.grid(color=GRID, lw=0.4)
            a.set_xticks(xs, [f"{v:g}" for v in closures])
        axes[1, j].set_xlabel(r"true closure $\mu$ (deg)")
    axes[0, 0].set_ylabel("coverage of the 95% arc")
    axes[0, 0].yaxis.set_label_coords(-0.14, 0.5)
    axes[1, 0].yaxis.set_label_coords(-0.14, 0.5)
    axes[1, 0].set_ylabel(r"bias of $\hat\mu$ (% of $\mu$)")
    axes[0, 0].set_ylim(0, 1.02)
    axes[1, 0].set_ylim(-45, 45)
    for a in axes.ravel():
        a.set_xlim(-0.5, len(closures) - 0.5)
    for (m, c), cols in tables.items():
        write_dat(args.data, f"{m}_{c.replace('-1.01', '-1')}", **cols)
    write_dat(args.data, "closures", idx=xs, closure=closures)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(args.out, bbox_inches="tight")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--window", default="15x5")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path, default=None,
                   help="Folder of the pgfplots tables <method>_<class> (optional).")
    main(p.parse_args())
