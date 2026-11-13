r"""Detected shares per triplet and class from a ``zone_scan.py --extra`` run, for a chosen test
(``--p-field``: ``p`` own precisions, ``p_nb`` neighbor precisions, ``p_equal`` equal weights).

Output: ``summary.csv`` with the columns of ``zone_scan.py`` (window, triplet, class_def,
class, zones, median_abs_mu_deg, rate_<pfa>), so that ``experiments/series/closure_weather.py``
can read it as a scan run.

Example:
    python experiments/scan/share_summary.py --scan-run <zone-scan-extra run> --p-field p_nb \
        --task share-summary-nb-s1
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import zarr

from closure_glrt import cli
from closure_glrt.provenance import finish_run, start_run

logger = logging.getLogger(__name__)


def main(args):
    small = Path(args.scan_run)
    zones = zarr.open_group(small.parent.parent / "large" / "runs" / small.name / "zones.zarr",
                            mode="r")
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), [small / "manifest.json"],
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    mu_field = {"p": "mu", "p_nb": "mu_nb", "p_equal": "mu_equal"}[args.p_field]
    rows = []
    for wname in zones.group_keys():
        g = zones[wname]
        for k, t in enumerate(zones.attrs["triplets"]):
            p, mu = np.asarray(g[args.p_field][k]), np.asarray(g[mu_field][k])
            for cdef in ("gbar_own", "gbar_indep"):
                gb = np.asarray(g[cdef][k])
                for lo, hi in zip(args.classes[:-1], args.classes[1:]):
                    sel = np.isfinite(p) & (gb >= lo) & (gb < hi)
                    n = int(sel.sum())
                    if n == 0:
                        continue
                    row = {"window": wname, "triplet": t, "class_def": cdef,
                           "class": f"{lo:g}-{hi:g}", "zones": n,
                           "median_abs_mu_deg": float(np.degrees(np.median(np.abs(mu[sel]))))}
                    for pfa in args.pfa:
                        row[f"rate_{pfa:g}"] = float((p[sel] < pfa).mean())
                    rows.append(row)
    with open(run.small_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    finish_run(run, {"summary": run.small_dir / "summary.csv"})
    logger.info("done: %s", run.small_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan-run", required=True)
    parser.add_argument("--p-field", default="p_nb", choices=["p", "p_nb", "p_equal"])
    parser.add_argument("--classes", type=float, nargs="+",
                        default=[0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01])
    parser.add_argument("--pfa", type=float, nargs="+", default=[0.01, 0.1, 0.001])
    cli.add_run_arguments(parser, task="share-summary")
    args = parser.parse_args()
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
