r"""LaTeX table of the fit of three laws of the closure of a cell, from one ``closure_fit.py`` run.

Rows: sources of the run (real and simulated, UAVSAR and Sentinel-1). Columns, for the model
``vm_c`` $\mathcal{VM}(0,\hat c\kappa_{0,\ell})$, the free von Mises law ``vm_free`` (one
concentration per zone, equal weights) and the nominal number of looks ``vm_1`` ($c=1$): the
largest Kolmogorov-Smirnov distance of the normal scores to $\mathcal{N}(0,1)$ over the classes of
the weakest coherence of the cell, and the share of the normal scores beyond 3 in absolute value
over all the cells (0.27 % under $\mathcal{N}(0,1)$).

Example:
    python experiments/supplement/fit_laws_table.py --run <run> --out fit_laws.tex
"""

import argparse
import csv
from pathlib import Path

SOURCES = (("uavsar", "UAVSAR, real"), ("sim_h0", r"UAVSAR, sim.\ $\hypothesis_0$"),
           ("sentinel1", "Sentinel-1, real"), ("sim_h0_s1", r"Sentinel-1, sim.\ $\hypothesis_0$"))
LAWS = ("vm_c", "vm_free", "vm_1")


def main(args):
    with open(Path(args.run) / "fit.csv") as f:
        rows = list(csv.DictReader(f))
    lines = [r"\begin{tabular}{@{}l" + "rr" * len(LAWS) + "@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Model $\hat c\kappazerocell{\icell}$}"
             r" & \multicolumn{2}{c}{Free $\hat\concentration$}"
             r" & \multicolumn{2}{c}{Nominal ($\concscale=1$)}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
             r"Data" + r" & KS & $\abs{s}>3$" * len(LAWS) + r"\\", r"\midrule"]
    for src, label in SOURCES:
        sel = [r for r in rows if r["source"] == src and int(r["cells"]) >= args.min_cells]
        if not sel:
            continue
        cells = []
        for law in LAWS:
            per = [r for r in sel if r["law"] == law and r["class"] != "all"]
            pooled = next(r for r in sel if r["law"] == law and r["class"] == "all")
            cells += [f"{max(float(r['ks']) for r in per):.3f}",
                      f"{100 * float(pooled['p_abs_gt3']):.2f}\\%"]
        lines.append(label + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    args.out.write_text("\n".join(lines))
    print("wrote", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--min-cells", type=int, default=20000,
                   help="Classes with fewer cells are left out of the largest KS distance.")
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
