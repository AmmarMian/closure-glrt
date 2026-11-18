r"""LaTeX table of the detected share on incoherent zones per test, from ``tests_summary.py`` runs.

Incoherent zones: class 0-0.2 of the zone coherence, cells of ``--window``. Same-day triplets
(``--sameday``) use the class of the tested triplet (``gbar_own``) and are given as the range over
the triplets; the other triplets use the class from the independent pairs (``gbar_indep``) and are
given as the median over the triplets. One column pair per segment.

Example:
    python experiments/supplement/calibration_table.py --run s1=<run> --run s2=<run> \
        --out calibration.tex
"""

import argparse
import csv
import statistics
from pathlib import Path

TESTS = (("scale_free", "Proposed"), ("equal", "Equal weights"), ("nominal", "Nominal looks"),
         ("zwieback", "Zwieback"), ("yuan", "Yuan"))


def main(args):
    runs = dict(s.split("=", 1) for s in args.run)
    data = {}
    for seg, run in runs.items():
        with open(Path(run) / "tests.csv") as f:
            rows = [r for r in csv.DictReader(f) if r["window"] == args.window
                    and r["class"] == "0-0.2"]
        same = [r for r in rows if r["class_def"] == "gbar_own"
                and int(r["triplet"]) in args.sameday]
        series = [r for r in rows if r["class_def"] == "gbar_indep"
                  and int(r["triplet"]) not in args.sameday]
        data[seg] = (same, series)
    segs = list(runs)
    n = len(segs)
    out = [r"\begin{tabular}{@{}l" + "r" * (2 * n) + "@{}}", r"\toprule",
           r" & \multicolumn{" + str(n) + r"}{c}{Same-day} & \multicolumn{" + str(n)
           + r"}{c}{Series (median)}\\",
           r"\cmidrule(lr){2-" + str(n + 1) + r"}\cmidrule(l){" + str(n + 2) + "-"
           + str(2 * n + 1) + "}",
           "Test" + "".join(f" & {s}" for s in segs) * 2 + r"\\", r"\midrule"]
    for key, name in TESTS:
        cells = []
        for seg in segs:
            v = [100 * float(r[f"share_{key}"]) for r in data[seg][0]]
            cells.append(f"{min(v):.1f}--{max(v):.1f}")
        for seg in segs:
            v = [100 * float(r[f"share_{key}"]) for r in data[seg][1]]
            cells.append(f"{statistics.median(v):.1f}")
        out.append(f"{name} & " + " & ".join(cells) + r"\\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    args.out.write_text("\n".join(out))
    print("wrote", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="+", required=True, help="SEG=tests_summary run")
    p.add_argument("--sameday", type=int, nargs="+", default=[10, 23])
    p.add_argument("--window", default="15x5")
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
