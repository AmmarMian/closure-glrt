r"""LaTeX table of the model fit with own and neighbor precisions, from ``closure_fit.py`` runs.

Per coherence class: Kolmogorov-Smirnov distance of the normal scores to N(0, 1), their mean
square and the share with |s| > 3, on real and simulated H0 zones, for the scaled von Mises law
(``vm_c``). One run with own precisions (``--own``), one with ``--weights nb`` (``--nb``).

Example:
    python experiments/supplement/fit_table.py --own <run> --nb <run> --window 15x5 --out fit.tex
"""

import argparse
import csv
from pathlib import Path

CLASSES = ["0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1", "all"]


def read(run, window):
    with open(Path(run) / "fit.csv") as f:
        return {(r["source"], r["class"]): r for r in csv.DictReader(f)
                if r["window"] == window and r["law"] == "vm_c"}


def label(cls):
    if cls == "all":
        return "All"
    lo, hi = cls.split("-")
    return rf"${lo}$--${hi}$" if hi != "1" else rf"$\ge{lo}$"


def main(args):
    runs = [read(args.own, args.window), read(args.nb, args.window)]
    cols = [(r, s) for r in runs for s in ("uavsar", "sim_h0")]
    lines = [r"\begin{tabular}{l" + "rrrr" * 3 + "}", r"\hline",
             r" & \multicolumn{4}{c}{KS distance} & \multicolumn{4}{c}{Mean $s^2$}"
             r" & \multicolumn{4}{c}{$|s|>3$ (\%)}\\",
             r"Precisions"
             + r" & \multicolumn{2}{c}{own} & \multicolumn{2}{c}{neighbor}" * 3 + r"\\",
             r"Class $\bar\gamma$" + " & real & sim." * 6 + r"\\", r"\hline"]
    for cls in CLASSES:
        ks = [f"{float(r[(s, cls)]['ks']):.3f}" for r, s in cols]
        ms = [f"{float(r[(s, cls)]['mean_s2']):.2f}" for r, s in cols]
        p3 = [f"{100 * float(r[(s, cls)]['p_abs_gt3']):.2f}" for r, s in cols]
        lines.append(label(cls) + " & " + " & ".join(ks + ms + p3) + r"\\")
    lines += [r"\hline", r"\end{tabular}", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--own", required=True, help="closure_fit run with own precisions")
    p.add_argument("--nb", required=True, help="closure_fit run with --weights nb")
    p.add_argument("--window", default="15x5")
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
