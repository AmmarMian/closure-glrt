r"""UAVSAR / Sentinel-1 comparison by class of coherence in both sensors (Section V-C, supplement).

Reads the ``zones.npz`` of runs of ``cross_sensor.py`` (common zones of each matched triplet,
already coherent in both sensors at 0.5) and recomputes, on the zones whose selection coherence
is at least ``q`` in BOTH sensors, for each ``q`` of ``--thresholds``: number of zones, detected
shares at ``--pfa``, median arc half-width and median $|\hat\mu|$ of each sensor, Spearman
correlation of the two $\hat\mu$ with a spatial block bootstrap interval, and sign agreement on
the zones detected by both. Rows with fewer than ``--min-zones`` zones are written without
statistics.

Output: ``classes.csv`` (one row per segment, matched triplet and threshold).

Example:
    uv run python experiments/sentinel1/coherence_classes.py \
        --xsensor s1=<runs>/runs/<cross-sensor-s1> s2=<runs>/runs/<cross-sensor-s2> \
        --task cross-sensor-coherence-classes
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cross_sensor import block_bootstrap, spearman  # noqa: E402

from closure_glrt import cli  # noqa: E402
from closure_glrt.provenance import finish_run, start_run  # noqa: E402

logger = logging.getLogger(__name__)


def main(args):
    runs = dict(s.split("=", 1) for s in args.xsensor)
    inputs = [Path(r) / "zones.npz" for r in runs.values()]
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), inputs,
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    rng = np.random.default_rng(args.seed)
    rows = []
    for seg, path in zip(runs, inputs):
        z = np.load(path)
        for name in sorted({k.split("__")[0] for k in z.files}):
            g = {f: z[f"{name}__{f}"] for f in ("keys", "coh_uavsar", "coh_s1", "p_uavsar",
                                                 "p_s1", "mu_uavsar", "mu_s1", "h_uavsar", "h_s1")}
            for q in args.thresholds:
                m = (g["coh_uavsar"] >= q) & (g["coh_s1"] >= q)
                row = {"segment": seg, "s1_triplet": name.rsplit("_t", 1)[0],
                       "track": name.split("_")[0], "t": int(name.rsplit("_t", 1)[1]),
                       "coh_min": q, "zones": int(m.sum())}
                if m.sum() >= args.min_zones:
                    du, ds = g["p_uavsar"][m] < args.pfa, g["p_s1"][m] < args.pfa
                    mu, ms = g["mu_uavsar"][m], g["mu_s1"][m]
                    lo, hi = block_bootstrap(g["keys"][m], mu, ms, args.block_zones, args.boot,
                                             rng, spearman)
                    both = du & ds
                    row.update({
                        "det_uavsar": du.mean(), "det_s1": ds.mean(),
                        "h_uavsar_med_deg": np.degrees(np.median(g["h_uavsar"][m])),
                        "h_s1_med_deg": np.degrees(np.median(g["h_s1"][m])),
                        "absmu_uavsar_med_deg": np.degrees(np.median(np.abs(mu))),
                        "absmu_s1_med_deg": np.degrees(np.median(np.abs(ms))),
                        "spearman_mu": spearman(mu, ms), "spearman_lo": lo, "spearman_hi": hi,
                        "n_both": int(both.sum()),
                        "sign_agree_both": float((np.sign(mu[both]) == np.sign(ms[both])).mean())
                        if both.any() else np.nan})
                rows.append(row)
                logger.info("%s %s q=%.2f: %d zones", seg, name, q, m.sum())
    out = run.small_dir / "classes.csv"
    fields = list(dict.fromkeys(f for r in rows for f in r))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    finish_run(run, {"classes": out})
    logger.info("done: %s", run.small_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xsensor", nargs="+", required=True, help="SEG=cross-sensor run dir")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--pfa", type=float, default=0.01)
    parser.add_argument("--min-zones", type=int, default=50)
    parser.add_argument("--block-zones", type=int, default=10)
    parser.add_argument("--boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    cli.add_run_arguments(parser, task="cross-sensor-coherence-classes")
    args = parser.parse_args()
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
