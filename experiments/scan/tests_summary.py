r"""Real-data comparison of the closure tests and intervals from a ``zone_scan.py --extra`` run
(TGRS plan, E3).

Per window, triplet and class of zone coherence (``gbar_own`` for the same-day triplets,
``gbar_indep`` otherwise, as in the paper): detected share at ``--pfa`` of the scale-free test
with own and neighbor precisions, nominal looks, equal weights, Zwieback and Yuan (zone
multilooked as one cell); median $|\hat\mu|$ of the three estimates; median half-width of the
scale-free arcs (own and neighbor precisions) and of the Yuan interval.

Output: ``tests.csv`` (one row per window, triplet, class definition and class).

Example:
    python experiments/scan/tests_summary.py --scan-run <runs>/runs/<zone-scan-extra run> \
        --task tests-summary-s1
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import zarr
from scipy import stats

from closure_glrt import cli
from closure_glrt.provenance import finish_run, start_run

logger = logging.getLogger(__name__)
SAMEDAY = (10, 23)


def main(args):
    small = Path(args.scan_run)
    zones = zarr.open_group(small.parent.parent / "large" / "runs" / small.name / "zones.zarr",
                            mode="r")
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), [small / "manifest.json"],
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    chi = stats.chi2.ppf(1 - args.pfa, 1)
    z = stats.norm.ppf(1 - args.pfa / 2)
    triplets = list(zones.attrs["triplets"])
    rows = []
    for wname in zones.group_keys():
        g = zones[wname]
        if "lam_nb" not in g:
            continue
        for k, t in enumerate(triplets):
            a = {f: np.asarray(g[f][k]) for f in g.array_keys()}
            dec = {"scale_free": a["p"] < args.pfa, "scale_free_nb": a["p_nb"] < args.pfa,
                   "nominal": a["lam_nominal"] > chi, "equal": a["p_equal"] < args.pfa,
                   "zwieback": a["zw"] > chi, "yuan": a["yuan"] > z}
            cdef = "gbar_own" if t in SAMEDAY else "gbar_indep"
            for lo, hi in zip(args.classes[:-1], args.classes[1:]):
                sel = np.isfinite(a["lam"]) & np.isfinite(a["lam_nb"]) & (a[cdef] >= lo) & \
                    (a[cdef] < hi)
                n = int(sel.sum())
                if n == 0:
                    continue
                row = {"window": wname, "triplet": t, "class_def": cdef, "class": f"{lo:g}-{hi:g}",
                       "zones": n}
                for m, d in dec.items():
                    row[f"share_{m}"] = float((d & sel).sum() / n)
                for m in ("mu", "mu_nb", "mu_zone"):
                    row[f"median_abs_{m}_deg"] = float(np.degrees(np.median(np.abs(a[m][sel]))))
                for m in ("hw", "hw_nb", "hw_yuan"):
                    row[f"median_{m}_deg"] = float(np.degrees(np.nanmedian(a[m][sel])))
                rows.append(row)
            logger.info("%s t=%d done", wname, t)
    with open(run.small_dir / "tests.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    finish_run(run, {"tests": run.small_dir / "tests.csv"})
    logger.info("done: %s", run.small_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan-run", required=True)
    parser.add_argument("--classes", type=float, nargs="+",
                        default=[0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01])
    parser.add_argument("--pfa", type=float, default=0.01)
    cli.add_run_arguments(parser, task="tests-summary")
    args = parser.parse_args()
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
