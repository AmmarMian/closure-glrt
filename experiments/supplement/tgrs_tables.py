r"""LaTeX tables of the TGRS simulations (E1, E2) from the runs of ``coverage.py``.

* ``coverage.tex``: for the arc with own and with neighbor precisions, coverage of the 95%
  interval, median half-width and bias of the estimate at closures 0, 2 and 5 deg, for the
  classes 0.5-0.7, 0.7-0.9 and >= 0.9 (touching cells, one cell size);
* ``robustness.tex``: per scenario, false-alarm rate of the scale-free test (range over the
  classes >= 0.2 and class < 0.2), its power at 2 deg (class 0.7-0.9), and coverage and bias
  of the neighbor-precision interval at 2 deg (class 0.7-0.9).

Example:
    python experiments/supplement/tgrs_tables.py --runs <run> ... --window 15x5 --out <dir>
"""

import argparse
import csv
from pathlib import Path

NAMES = {"scale_free": "Scale-free, own precisions",
         "scale_free_nb": "Scale-free, neighbor prec.",
         "nominal": "Nominal looks", "plugin": "Plug-in ENL", "equal": "Equal weights",
         "bootstrap": "Bootstrap", "zwieback": "Zwieback (zone)", "yuan": "Yuan (zone)"}
ARC_METHODS = ("scale_free", "scale_free_nb")   # intervals reported: the arc of the paper only
SCEN = [("touching", "5x5", "Reference"),
        ("texture1", "5x5", r"Texture $\nu=1$"),
        ("texture4", "5x5", r"Texture $\nu=4$"), ("texture20", "5x5", r"Texture $\nu=20$"),
        ("mixed", "5x5", "Mixed coh."), ("ramp", "5x5", r"Ramp $\pm1^\circ$"),
        ("split", "5x5", r"Split $\pm1^\circ$"), ("split3", "5x5", r"Split $\pm3^\circ$"),
        ("zone3", "3x3", r"$3\times3$ zones"), ("zone7", "7x7", r"$7\times7$ zones")]


def read(run):
    with open(Path(run) / "coverage.csv") as f:
        return list(csv.DictReader(f))


def key_of(run):
    name = Path(run).name
    for k, _, _ in SCEN:
        if f"coverage-{k}-" in name:
            return k
    return "competitors" if "competitors" in name else name


def pct(x):
    return f"{100 * float(x):.1f}"


def coverage_table(rows, window):
    sel = {(r["method"], r["class"], float(r["closure_deg"])): r for r in rows
           if r["window"] == window}
    classes = ("0.5-0.7", "0.7-0.9", "0.9-1.01")
    out = [r"\begin{tabular}{@{}l" + "rrr" * 3 + r"@{}}", r"\toprule",
           r" & \multicolumn{3}{c}{$0.5\le\bar\gamma<0.7$}"
           r" & \multicolumn{3}{c}{$0.7\le\bar\gamma<0.9$}"
           r" & \multicolumn{3}{c}{$\bar\gamma\ge0.9$} \\",
           r"$\mu$ (deg) & 0 & 2 & 5 & 0 & 2 & 5 & 0 & 2 & 5 \\", r"\midrule"]
    for what, fmt, title in (("coverage", pct, "Coverage (\\%)"),
                             ("median_halfwidth_deg", lambda v: f"{float(v):.2f}",
                              "Median half-width (deg)"),
                             ("mean_error_deg", lambda v: f"{float(v):+.2f}", "Bias (deg)")):
        out.append(r"\multicolumn{10}{@{}l}{\textit{" + title + r"}} \\")
        for m, name in NAMES.items():
            if m not in ARC_METHODS:
                continue
            cells = []
            for c in classes:
                for mu in (0.0, 2.0, 5.0):
                    r = sel.get((m, c, mu))
                    cells.append(fmt(r[what]) if r else "--")
            if any(x != "--" for x in cells):
                out.append(f"{name} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


# Groups of Table IV (step of the simulation diagram that each variant modifies) and the limits
# beyond which a value departs from the nominal behaviour and is set in bold.
GROUPS = [(None, ["touching"]),
          ("Speckle texture, step (a)", ["texture1", "texture4", "texture20"]),
          ("Coherence within the cell, step (b)", ["mixed"]),
          (r"Closure varying within the zone (cells at $\mu\pm\delta$), step (b)",
           ["ramp", "split", "split3"]),
          ("Zone size, step (c)", ["zone3", "zone7"])]
LIMITS = dict(reject=0.02, power=0.50, coverage=0.92)


def robustness_table(by_key, window):
    labels = {k: label for k, _, label in SCEN}

    def bold(text, flag):
        return rf"\textbf{{{text}}}" if flag else text

    out = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
           r" & \multicolumn{2}{c}{Rej. at $\mu=0$ (\%)} & Pow. & Cov. & Bias \\",
           r" & $\bar\gamma\ge0.2$ & $<0.2$ & (\%) & (\%) & (deg) \\",
           r"\midrule"]
    for title, keys in GROUPS:
        if title:
            out.append(r"\multicolumn{6}{@{}l}{\textit{" + title + r"}} \\")
        for k in keys:
            rows = [r for r in by_key.get(k, []) if r["window"] == window]
            if not rows:
                continue
            pf = [float(r["reject"]) for r in rows if r["method"] == "scale_free"
                  and float(r["closure_deg"]) == 0 and r["class"] not in ("0-0.2",)]
            pf0 = [float(r["reject"]) for r in rows if r["method"] == "scale_free"
                   and float(r["closure_deg"]) == 0 and r["class"] == "0-0.2"]
            h1 = {r["method"]: r for r in rows if float(r["closure_deg"]) == 2
                  and r["class"] == "0.7-0.9"}
            pw = float(h1["scale_free"]["reject"]) if "scale_free" in h1 else None
            cv = float(h1["scale_free_nb"]["coverage"]) if "scale_free_nb" in h1 else None
            cells = [bold(f"{100 * min(pf):.1f}--{100 * max(pf):.1f}", max(pf) > LIMITS["reject"])
                     if pf else "--",
                     bold(pct(pf0[0]), pf0[0] > LIMITS["reject"]) if pf0 else "--",
                     bold(pct(pw), pw < LIMITS["power"]) if pw is not None else "--",
                     bold(pct(cv), cv < LIMITS["coverage"]) if cv is not None else "--",
                     f"{float(h1['scale_free_nb']['mean_error_deg']):+.2f}"
                     if "scale_free_nb" in h1 else "--"]
            label = r"\textit{Reference (model)}" if k == "touching" else r"\quad " + labels[k]
            out.append(f"{label} & " + " & ".join(cells) + r" \\")
            if k == "touching":
                out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def compact_table(rows, window, mu=2.0):
    """Coverage and median half-width at one closure, per method and class (one column)."""
    sel = {(r["method"], r["class"]): r for r in rows
           if r["window"] == window and float(r["closure_deg"]) == mu}
    classes = ("0.5-0.7", "0.7-0.9", "0.9-1.01")
    out = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
           r" & \multicolumn{2}{c}{$0.5\le\bar\gamma<0.7$}"
           r" & \multicolumn{2}{c}{$0.7\le\bar\gamma<0.9$}"
           r" & \multicolumn{2}{c}{$\bar\gamma\ge0.9$} \\",
           r"Method & cov. & $h$ & cov. & $h$ & cov. & $h$ \\", r"\midrule"]
    short = {"scale_free": "Scale-free, own", "scale_free_nb": "Scale-free, neighbors",
             "nominal": "Nominal looks", "plugin": "Plug-in ENL", "equal": "Equal weights",
             "bootstrap": "Bootstrap", "zwieback": "Zwieback", "yuan": "Yuan"}
    for m, name in short.items():
        if m not in ARC_METHODS:
            continue
        cells = []
        for c in classes:
            r = sel.get((m, c))
            cells += [pct(r["coverage"]), f"{float(r['median_halfwidth_deg']):.2f}"] if r \
                else ["--", "--"]
        out.append(f"{name} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main(args):
    by_key = {key_of(r): read(r) for r in args.runs}
    args.out.mkdir(parents=True, exist_ok=True)
    cov_rows = by_key.get("touching", []) + by_key.get("competitors", [])
    (args.out / f"coverage_{args.window}.tex").write_text(coverage_table(cov_rows, args.window))
    (args.out / f"coverage_compact_{args.window}.tex").write_text(compact_table(cov_rows,
                                                                                args.window))
    (args.out / f"robustness_{args.window}.tex").write_text(robustness_table(by_key, args.window))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--window", default="15x5")
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
