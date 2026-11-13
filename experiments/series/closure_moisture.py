r"""Closure by crop against the change of soil moisture over all the triplets (TGRS plan, E6).

Pre-specified analysis (docs/tgrs-plan.md, E6, committed before any result): the unit is a
(crop, triplet) pair of a ``closure_landcover.py`` run made on all the triplets; the response is
the median signed closure of the zones of the crop (``--response``); the predictors are computed
from the ERA5 soil moisture of the top 7 cm at the three acquisition times, $m_1, m_2, m_3$:

* primary: the net change $m_3-m_1$;
* secondary: the second difference $m_1-2m_2+m_3$.

Per crop with at least ``--min-triplets`` triplets: Spearman correlation over the triplets and a
two-sided permutation p-value (``--perms`` permutations of the predictor). Pooled over the crops:
the slope of the predictor in a least-squares model with one intercept per crop, with a
permutation p-value (the same permutation of the triplets for all crops). Each analysis is done
on all the triplets and on the non-overlapping ones ($t\equiv0\bmod3$), which share no image.

Outputs: ``units.csv`` (crop, triplet, response, predictors) and ``moisture.csv`` (one row per
subset, predictor and crop or ``pooled``: n, statistic, p-value).

Example:
    uv run python experiments/series/closure_moisture.py --landcover-run <runs>/runs/<id> \
        --task closure-moisture-s1
"""

import argparse
import csv
import datetime as dt
import logging
from pathlib import Path

import numpy as np
from closure_weather import acquisition_weather, era5_scene, ghcn_rain  # same directory
from scipy import stats

from closure_glrt import cli
from closure_glrt.provenance import finish_run, start_run

logger = logging.getLogger(__name__)
PREDICTORS = ("dm_net", "dm_second")


def pooled_slope(y, x, crop):
    """Slope of x in y ~ crop + x (least squares, one intercept per crop)."""
    yc, xc = y.copy(), x.copy()
    for c in np.unique(crop):
        m = crop == c
        yc[m] -= yc[m].mean()
        xc[m] -= xc[m].mean()
    return float((xc * yc).sum() / (xc * xc).sum())


def main(args):
    by_class = Path(args.landcover_run) / "by_class.csv"
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), [by_class],
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    rng = np.random.default_rng(args.seed)
    anc = args.ancillary or cli.data_root() / "ancillary"
    with open(anc / "uavsar" / "acquisition_times.csv") as fh:
        when = [dt.datetime.fromisoformat(r["start_utc"]) for r in csv.DictReader(fh)]
    times, era = era5_scene(anc)
    rain = ghcn_rain(anc)
    sm = np.array([acquisition_weather(w, times, era, rain)["sm0_7"] for w in when])
    units = []
    with open(by_class) as fh:
        for r in csv.DictReader(fh):
            t = int(r["triplet"])
            m1, m2, m3 = sm[t:t + 3]
            units.append({"crop": r["class"], "triplet": t, "zones": int(r["zones"]),
                          "response_deg": float(r[args.response]),
                          "dm_net": m3 - m1, "dm_second": m1 - 2 * m2 + m3})
    rows = []
    for subset in ("all", "non_overlapping"):
        U = [u for u in units if subset == "all" or u["triplet"] % 3 == 0]
        crop = np.array([u["crop"] for u in U])
        y = np.array([u["response_deg"] for u in U])
        tri = np.array([u["triplet"] for u in U])
        for k in PREDICTORS:
            x = np.array([u[k] for u in U])
            ok = np.isfinite(x) & np.isfinite(y)
            kept = []
            for c in np.unique(crop[ok]):
                m = ok & (crop == c)
                if m.sum() < args.min_triplets:
                    continue
                kept.append(c)
                rho = stats.spearmanr(y[m], x[m])[0]
                null = np.array([stats.spearmanr(y[m], rng.permutation(x[m]))[0]
                                 for _ in range(args.perms)])
                rows.append({"subset": subset, "predictor": k, "crop": c, "n": int(m.sum()),
                             "statistic": "spearman", "value": float(rho),
                             "p_perm": float((np.abs(null) >= abs(rho)).mean())})
            m = ok & np.isin(crop, kept)
            if m.sum() < 3:
                continue
            b = pooled_slope(y[m], x[m], crop[m])
            ut = np.unique(tri[m])
            null = []
            for _ in range(args.perms):            # same permutation of triplets for all crops
                perm = dict(zip(ut, rng.permutation(ut)))
                xp = {t: x[m][tri[m] == t][0] for t in ut}
                null.append(pooled_slope(y[m], np.array([xp[perm[t]] for t in tri[m]]), crop[m]))
            null = np.array(null)
            rows.append({"subset": subset, "predictor": k, "crop": "pooled", "n": int(m.sum()),
                         "statistic": "slope_deg_per_m3m3", "value": b,
                         "p_perm": float((np.abs(null) >= abs(b)).mean())})
            logger.info("%s %s: pooled slope %.2f (p %.3f, %d units, %d crops)", subset, k, b,
                        rows[-1]["p_perm"], int(m.sum()), len(kept))
    for name, data in (("units", units), ("moisture", rows)):
        with open(run.small_dir / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    finish_run(run, {"units": run.small_dir / "units.csv",
                     "moisture": run.small_dir / "moisture.csv"}, extra={"ancillary": str(anc)})
    logger.info("done: %s", run.small_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--landcover-run", required=True, help="Run of closure_landcover.py.")
    parser.add_argument("--ancillary", type=Path, default=None)
    parser.add_argument("--response", default="mu_median_deg",
                        choices=["mu_median_deg", "mu_circmean_deg"])
    parser.add_argument("--min-triplets", type=int, default=10)
    parser.add_argument("--perms", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    cli.add_run_arguments(parser, task="closure-moisture")
    args = parser.parse_args()
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
