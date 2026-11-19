r"""False alarm and power of the tests (TGRS paper, one column), from the runs of ``coverage.py``.

(a) Rejection rate under $H_0$ per class of zone coherence: scale-free test with neighbor
precisions (touching cells, and independent pixels), with own precisions, equal weights, plug-in
ENL, nominal looks, Zwieback and Yuan, log scale, nominal rate and Monte Carlo band.
(b) Detection probability against the closure for $0.7\le\bar\gamma<0.9$ of the calibrated tests.

Example:
    python experiments/figures/fig_pfa_power_cov.py --touching <run> --white <run> \
        --competitors <run> --window 15x5 --out pfa_power.pdf
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pgfdata import write_dat
from style import COLUMN, GRID, INK, MUTED, S1, SAMEDAY, UAVSAR, style

TESTS = {"scale_free": ("scale-free (proposed, own)", UAVSAR, "o", "-"),
         "scale_free_nb": ("scale-free (neighbors)", UAVSAR, "o", "--"),
         "equal": ("equal weights", MUTED, "^", "-"),
         "plugin": ("plug-in ENL", SAMEDAY, "D", "-"),
         "nominal": ("nominal looks", S1, "s", ":"),
         "zwieback": ("Zwieback", S1, "s", "-"),
         "yuan": ("Yuan", "#b0325a", "P", "-")}
CLASSES = ("0-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.01")


def read(run, window):
    with open(Path(run) / "coverage.csv") as f:
        return [r for r in csv.DictReader(f) if r["window"] == window and r["reject"] != ""]


def main(args):
    style()
    comp = read(args.competitors, args.window)
    white = read(args.white, args.window) if args.white else []
    fig, (a, b) = plt.subplots(2, 1, figsize=(COLUMN, 4.1),
                               gridspec_kw={"hspace": 0.55, "top": 0.79, "bottom": 0.1,
                                            "left": 0.15, "right": 0.98})
    x = np.arange(len(CLASSES))
    pfa_cols = {"cls": x, "label": [c.replace("-1.01", "-1") for c in CLASSES]}
    for m, (lab, col, mk, ls) in TESTS.items():
        y = [next((float(r["reject"]) for r in comp if r["method"] == m and r["class"] == c
                   and float(r["closure_deg"]) == 0), np.nan) for c in CLASSES]
        pfa_cols[m] = y
        a.plot(x, y, color=col, marker=mk, ls=ls, lw=0.9, ms=3, label=lab)
    y = [next((float(r["reject"]) for r in white if r["method"] == "scale_free"
               and r["class"] == c and float(r["closure_deg"]) == 0), np.nan) for c in CLASSES]
    if white:
        a.plot(x, y, color=UAVSAR, marker="o", mfc="white", ls="none", ms=3.5,
               label="scale-free (own), indep. pixels")
    n = 2000
    band = 2 * np.sqrt(args.pfa * (1 - args.pfa) / n)
    a.axhspan(args.pfa - band, args.pfa + band, color=GRID, lw=0)
    write_dat(args.data, "pfa", **pfa_cols)
    write_dat(args.data, "band", x=[-0.5, len(CLASSES) - 0.5], pfa=[args.pfa] * 2,
              lo=[args.pfa - band] * 2, hi=[args.pfa + band] * 2)
    a.axhline(args.pfa, color=INK, lw=0.6, ls=":")
    a.set_yscale("log")
    a.set_xticks(x, [c.replace("-1.01", "-1") for c in CLASSES])
    a.set_xlabel(r"zone coherence $\bar\gamma$")
    a.set_ylabel("rejection rate under $H_0$")
    a.grid(color=GRID, lw=0.4)
    fig.legend(*a.get_legend_handles_labels(), fontsize=7, ncol=2, frameon=False,
               loc="upper center", bbox_to_anchor=(0.55, 1.0), handlelength=2.2,
               columnspacing=1.0)
    a.set_title("(a) false alarms", loc="left", fontweight="bold")
    closures = sorted({float(r["closure_deg"]) for r in comp})
    power_cols = {"idx": np.arange(len(closures)), "closure": closures}
    for m in ("scale_free", "scale_free_nb", "equal", "plugin"):
        lab, col, mk, ls = TESTS[m]
        pts = [(mu, float(r["reject"])) for mu in closures for r in comp
               if r["method"] == m and r["class"] == args.power_class
               and float(r["closure_deg"]) == mu]
        power_cols[m] = [next((p[1] for p in pts if p[0] == mu), np.nan) for mu in closures]
        if pts:                       # closures equally spaced, as in the coverage figure
            b.plot([closures.index(p[0]) for p in pts], [p[1] for p in pts], color=col, marker=mk,
                   ls=ls, lw=0.9, ms=3, label=lab)
    write_dat(args.data, "power", **power_cols)
    b.set_xticks(range(len(closures)), [f"{v:g}" for v in closures])
    b.axhline(args.pfa, color=INK, lw=0.6, ls=":")
    b.set_xlabel(r"closure $\mu$ (deg)")
    b.set_ylabel("detection probability")
    b.grid(color=GRID, lw=0.4)
    b.set_title(f"(b) power, {args.power_class.replace('-', '$\\leq\\bar\\gamma<$')}",
                loc="left", fontweight="bold")
    fig.savefig(args.out, bbox_inches="tight")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--competitors", required=True, help="coverage run with all the methods")
    p.add_argument("--white", default=None, help="coverage run with independent pixels (optional)")
    p.add_argument("--window", default="15x5")
    p.add_argument("--power-class", default="0.7-0.9")
    p.add_argument("--pfa", type=float, default=0.01)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path, default=None,
                   help="Folder of the pgfplots tables pfa, band, power (optional).")
    main(p.parse_args())
