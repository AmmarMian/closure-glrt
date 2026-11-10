r"""Tables of the supplementary material, written from the run directories (Section V).

Every table is read from the small directory of a run and its caption names that run (id and
commit), so that each number of Section V can be traced to the run that produced it:

* ``runs.tex``: the runs used (id, task, commit, command);
* ``sameday.tex``: same-day triplets, per segment, cell size, triplet and class of the zone
  coherence of the triplet (``gbar_own``): zones, detected share of the scale-free test and of the
  equal-weights test, median $|\hat\mu|$ and median $\hat c$;
* ``samesign.tex``: same-day triplets, sign agreement of detected zones $d$ zones apart;
* ``series_summary.tex``: along the series, per segment, cell size, class definition and class:
  minimum, median and maximum over the triplets of the detected share, and range of median $\hat c$;
* ``series_<seg>.tex``: per triplet (cells of 15x5): dates, detected shares of the classes of the
  independent pairs (``gbar_indep``), share of the zones incoherent on the triplet
  (``gbar_own`` < 0.2) for both cell sizes, soil moisture and 7-day rain at the last date;
* ``landcover.tex``: by crop, for the triplets of the land-cover runs;
* ``sentinel1.tex``: per matched Sentinel-1 triplet, on the common coherent zones of both sensors
  (``experiments/sentinel1/cross_sensor.py``).

Shares are given in percent at ``--pfa``; a class with fewer than ``--min-zones`` zones on a
triplet is left blank in the per-triplet tables, as in Fig. 7.

Example:
    python experiments/supplement/tables.py --runs-root <runs>/runs \
        --scan s1=<zone-scan s1> s2=<zone-scan s2> --sameday s1=<run> s2=<run> \
        --weather s1=<run> s2=<run> --landcover s1=<run> s2=<run> --out <paper>/supplement/tables
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

CLASSES = ["0-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.01"]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def manifest(run):
    with open(Path(run) / "manifest.json") as f:
        return json.load(f)


def tt(s):
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def cite(run):
    m = manifest(run)
    return f"run {tt(m['run_id'])}"


def cls(c):
    lo, hi = c.split("-")
    return rf"$\ge{lo}$" if float(hi) > 1 else f"{lo}--{hi}"


def pct(x, nd=1):
    return "" if x is None else f"{100 * float(x):.{nd}f}"


def pairs(items):
    return dict(x.split("=", 1) for x in items)


def longtable(spec, head, rows, caption, label):
    out = [rf"\begin{{longtable}}{{{spec}}}",
           rf"\caption{{{caption}}}\label{{{label}}}\\", r"\toprule", head + r"\\", r"\midrule",
           r"\endfirsthead", r"\toprule", head + r"\\", r"\midrule", r"\endhead",
           r"\bottomrule", r"\endfoot"]
    out += [" & ".join(r) + r"\\" if isinstance(r, list) else r for r in rows]
    out.append(r"\end{longtable}")
    return "\n".join(out) + "\n"


def runs_table(runs):
    rows = []
    for run in runs:
        m = manifest(run)
        cmd = " ".join(Path(c).name if c.startswith("/") else c for c in m["command"])
        breakable = tt(cmd).replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")
        rows.append([tt(m["run_id"]), m["git"]["commit"][:7]])
        rows.append(r"\multicolumn{2}{p{0.9\linewidth}}{\raggedright\quad " + breakable
                    + r"}\\[3pt]")
    return longtable("ll", "Run & Commit (command below, paths shortened)", rows,
                     "Runs used in Section~V. Each run directory holds its manifest (inputs, "
                     "versions, git state) and its outputs; the ledger of all runs is "
                     r"\texttt{ledger.jsonl}.", "tab:runs")


def sameday_tables(sameday, scan, pfa):
    rows, rows_sign = [], []
    for seg in sorted(sameday):
        diag = read(Path(sameday[seg]) / "diagnostics.csv")
        summ = {(r["window"], r["triplet"], r["class"]): r for r in read(Path(scan[seg]) /
                "summary.csv") if r["class_def"] == "gbar_own"}
        for r in diag:
            if r["quantity"] == "share":
                s = summ[r["window"], r["triplet"], r["key"]]
                rows.append([seg, r["window"], r["triplet"], cls(r["key"]), r["zones"],
                             pct(r["scale_free"]), pct(r["equal_weights"]),
                             f"{float(s['median_abs_mu_deg']):.2f}",
                             f"{float(s['median_c1']):.2f}"])
            else:
                rows_sign.append([seg, r["window"], r["triplet"], r["key"].replace("d=", ""),
                                  r["zones"], pct(r["scale_free"])])
    cap = (rf"Same-day triplets: detected share (\%, $\Pfa={pfa:g}$) per class of the zone "
           r"coherence of the tested triplet \eqref{eq:zonecoh}, for the scale-free test and the "
           r"equal-weights test, and median $\abs{\hat\meandir}$ (deg) and $\hat c$ of all the "
           r"zones. " + "; ".join(f"{seg}: {cite(sameday[seg])} and {cite(scan[seg])}"
                                  for seg in sorted(sameday)) + ".")
    head = (r"Seg. & Cells & $t$ & $\zonecoh$ & Zones & Scale-free & Equal w. & "
            r"$\abs{\hat\meandir}$ & $\hat c$")
    t1 = longtable("lllrrrrrr", head, rows, cap, "tab:sameday")
    cap2 = (r"Same-day triplets: share (\%) of the pairs of detected zones of $\zonecoh\ge0.5$, "
            r"$d$ zones apart along azimuth and range, with the same sign of $\hat\meandir$ "
            r"(noise: 50\%). "
            + "; ".join(f"{seg}: {cite(sameday[seg])}" for seg in sorted(sameday)) + ".")
    t2 = longtable("lllrrr", r"Seg. & Cells & $t$ & $d$ & Pairs & Same sign", rows_sign, cap2,
                   "tab:samesign")
    return t1, t2


def series_summary(scan, pfa, min_zones):
    rows = []
    for seg in sorted(scan):
        summ = read(Path(scan[seg]) / "summary.csv")
        for cdef in ("gbar_own", "gbar_indep"):
            for win in ("15x5", "30x10"):
                for c in CLASSES:
                    sel = [r for r in summ if r["class_def"] == cdef and r["window"] == win
                           and r["class"] == c and int(r["zones"]) >= min_zones]
                    if not sel:
                        continue
                    v = [float(r[f"rate_{pfa:g}"]) for r in sel]
                    cc = [float(r["median_c1"]) for r in sel]
                    name = r"$\zonecoh$" if cdef == "gbar_own" else r"$\zonecohind$"
                    rows.append([seg, win, name, cls(c), str(len(sel)), pct(min(v)),
                                 pct(statistics.median(v)), pct(max(v)),
                                 f"{min(cc):.2f}--{max(cc):.2f}"])
    cap = (rf"Along the series: detected share (\%, $\Pfa={pfa:g}$) over the triplets with at "
           rf"least {min_zones} zones in the class (minimum, median, maximum) and range of the "
           r"median $\hat c$, per class definition. "
           + "; ".join(f"{seg}: {cite(scan[seg])}" for seg in sorted(scan)) + ".")
    head = r"Seg. & Cells & Class & Range & Triplets & Min & Median & Max & Median $\hat c$"
    return longtable("lllrrrrrr", head, rows, cap, "tab:series-summary")


def series_triplets(seg, scan, weather, pfa, min_zones):
    summ = read(Path(scan) / "summary.csv")
    get = {(r["window"], r["class_def"], r["class"], int(r["triplet"])): r for r in summ}
    wx = read(Path(weather) / "triplets.csv")
    ind = ["0.5-0.7", "0.7-0.9", "0.9-1.01"]
    rows = []
    for w in wx:
        t = int(w["triplet"])
        cells = [str(t), "/".join(w[f"date{i}"][2:] for i in (1, 2, 3))]
        for c in ind:
            r = get["15x5", "gbar_indep", c, t]
            cells.append(pct(r[f"rate_{pfa:g}"]) if int(r["zones"]) >= min_zones else "")
        for win in ("15x5", "30x10"):
            r = get[win, "gbar_own", "0-0.2", t]
            cells.append(pct(r[f"rate_{pfa:g}"]))
        r = get["30x10", "gbar_own", "0-0.2", t]
        cells.append(f"{float(r[f'median_abs_mu_det_{pfa:g}_deg']):.1f}")
        cells += [f"{float(w['sm0_7_last']):.2f}", f"{float(w['rain7d_last']):.1f}"]
        rows.append(cells)
    cap = (rf"Segment {seg}, per triplet: detected share (\%, $\Pfa={pfa:g}$) of the classes of "
           rf"$\zonecohind$ (cells of $15\times5$; blank: fewer than {min_zones} zones), of the "
           r"zones incoherent on the triplet ($\zonecoh<0.2$) for both cell sizes, median "
           r"$\abs{\hat\meandir}$ (deg) of the detected incoherent zones ($30\times10$), and ERA5 "
           r"soil moisture (m$^3$/m$^3$) and GHCN 7-day rain (mm) at the last date. "
           f"Shares: {cite(scan)}; weather: {cite(weather)}.")
    head = (r"$t$ & Dates & \multicolumn{3}{c}{$\zonecohind$: 0.5--0.7, 0.7--0.9, $\ge0.9$} & "
            r"\multicolumn{2}{c}{$\zonecoh<0.2$: $15{\times}5$, $30{\times}10$} & "
            r"$\abs{\hat\meandir}$ & SM & Rain")
    return longtable("rlrrrrrrrr", head, rows, cap, f"tab:series-{seg}")


def landcover_table(landcover, pfa):
    rows = []
    for seg in sorted(landcover):
        for r in read(Path(landcover[seg]) / "by_class.csv"):
            rows.append([seg, r["triplet"], r["class"], r["zones"], pct(r["share_detected"]),
                         pct(r["frac_detected_neg"]), pct(r["frac_detected_pos"]),
                         f"{float(r['mu_circmean_deg']):+.1f}"])
    cap = (r"By crop (CDL of the year of the middle date), zones with $\zonecohind\ge0.5$ and CDL "
           rf"purity $\ge7/9$: zones, detected share (\%, $\Pfa={pfa:g}$), share detected with "
           r"$\hat\meandir<0$ and with $\hat\meandir>0$ (\%), and circular mean of $\hat\meandir$ "
           r"over all the zones of the crop (deg). "
           + "; ".join(f"{seg}: {cite(landcover[seg])}" for seg in sorted(landcover)) + ".")
    head = r"Seg. & $t$ & Crop & Zones & Detected & $\hat\meandir<0$ & $\hat\meandir>0$ & Mean"
    return longtable("llp{0.22\\linewidth}rrrrr", head, rows, cap, "tab:landcover")


def sentinel1_table(xsensor):
    rows = []
    for seg in sorted(xsensor):
        for r in read(Path(xsensor[seg]) / "summary.csv"):
            if not r.get("zones"):
                continue
            f2 = lambda k: f"{float(r[k]):.2f}"  # noqa: E731
            rows.append([seg, r["track"], r["t"], r["zones"], pct(r["det_uavsar"]),
                         pct(r["det_s1"]), f"{float(r['h_uavsar_med_deg']):.1f}",
                         f"{float(r['h_s1_med_deg']):.1f}", f2("rel_robust_uavsar"),
                         f2("rel_robust_s1"), f2("r_max_robust"),
                         f"{float(r['spearman_mu']):+.3f} [{float(r['spearman_lo']):+.3f}, "
                         f"{float(r['spearman_hi']):+.3f}]",
                         f"{pct(r['sign_agree_both'], 0)} ({r['n_both']})"])
    cap = (r"Sentinel-1 and UAVSAR on the common coherent zones of 100~m of each matched triplet "
           r"(VV, UAVSAR cells of $30\times10$ pixels, Sentinel-1 cells of 20~m): zones, detected "
           r"shares (\%, $\Pfa=0.01$), median half-width of the $95\%$ confidence arc (deg), "
           r"robust share of the spread of $\hat\meandir$ over the zones that exceeds the "
           r"estimation error ($\varrho$, medians and MAD), rank correlation $r_{\max}$ expected "
           r"if both sensors measured the same closures, observed rank correlation of the two "
           r"$\hat\meandir$ with its spatial block bootstrap $95\%$ interval, and sign agreement "
           r"(\%) on the zones detected by both (their number). "
           + "; ".join(f"{seg}: {cite(xsensor[seg])}" for seg in sorted(xsensor)) + ".")
    head = (r"Seg. & Track & $t$ & Zones & Det.~U & Det.~S1 & $h$~U & $h$~S1 & $\varrho$~U & "
            r"$\varrho$~S1 & $r_{\max}$ & Rank corr. & Sign")
    return longtable("lllrrrrrrrrlr", head, rows, cap, "tab:sentinel1")


def main(args):
    root = Path(args.runs_root)
    scan, sameday, weather, landcover, xsensor = (
        {k: root / v for k, v in pairs(x).items()}
        for x in (args.scan, args.sameday, args.weather, args.landcover, args.xsensor))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    every = [*scan.values(), *sameday.values(), *weather.values(), *landcover.values(),
             *xsensor.values()]
    (out / "runs.tex").write_text(runs_table(every))
    t1, t2 = sameday_tables(sameday, scan, args.pfa)
    (out / "sameday.tex").write_text(t1)
    (out / "samesign.tex").write_text(t2)
    (out / "series_summary.tex").write_text(series_summary(scan, args.pfa, args.min_zones))
    for seg in sorted(scan):
        (out / f"series_{seg}.tex").write_text(
            series_triplets(seg, scan[seg], weather[seg], args.pfa, args.min_zones))
    (out / "landcover.tex").write_text(landcover_table(landcover, args.pfa))
    (out / "sentinel1.tex").write_text(sentinel1_table(xsensor))
    print(f"tables written to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", required=True, help="Directory holding the run directories.")
    for name in ("scan", "sameday", "weather", "landcover", "xsensor"):
        p.add_argument(f"--{name}", nargs="+", required=True, metavar="SEG=RUN_ID")
    p.add_argument("--pfa", type=float, default=0.01)
    p.add_argument("--min-zones", type=int, default=1000)
    p.add_argument("--out", required=True)
    main(p.parse_args())
