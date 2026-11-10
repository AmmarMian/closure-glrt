r"""Detected share of every triplet against the weather around its dates (Section V-B).

From a ``zone_scan.py`` run: the share $\rho_t$ of zones with a detected closure (exact test,
``--pfa``) per triplet, for the zone-coherence classes ``--classes`` of the definition
``--class-def``. From the ancillary data (``$DATA_ROOT/ancillary``):

* acquisition times (UTC) of the scan dates, ``uavsar/acquisition_times.csv`` (scan order);
* daily rain: mean over the GHCN-Daily stations reporting precipitation, ``ghcn/daily.csv``;
* hourly ERA5 (Open-Meteo archive) at a grid of points, averaged: ``era5/point_*.json``.

Per acquisition: soil moisture of the top 7 cm at the acquisition hour and GHCN rain of the 7
days ending on the acquisition date. Per triplet: the soil moisture at its last date, and the
largest difference of the 7-day rain between its three dates. Outputs: ``triplets.csv`` (one row
per triplet: dates, shares, weather) and ``correlations.csv`` (Spearman correlation across
triplets of each class's share with each weather variable; consecutive triplets share two dates,
so the samples are dependent).

Example:
    uv run python experiments/series/closure_weather.py --scan-run <runs>/runs/<zone-scan run> \
        --task closure-weather-s1
"""

import argparse
import csv
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats

from closure_glrt import cli
from closure_glrt.provenance import finish_run, start_run

logger = logging.getLogger(__name__)
WEATHER = ("sm0_7_last", "rain7d_maxdiff", "rain7d_last", "sm0_7_maxdiff")


def era5_scene(ancillary):
    """Hourly UTC times and the mean over the grid points of the ERA5 series."""
    pts = [json.loads(p.read_text()) for p in sorted((ancillary / "era5").glob("point_*.json"))]
    times = np.array(pts[0]["hourly"]["time"], dtype="datetime64[h]")
    out = {}
    for k in pts[0]["hourly"]:
        if k != "time":
            v = np.array([[np.nan if x is None else x for x in p["hourly"][k]] for p in pts], float)
            out[k] = np.nanmean(v, 0)
    return times, out


def ghcn_rain(ancillary):
    """Daily mean precipitation (mm) over the GHCN stations that report it."""
    acc = {}
    with open(ancillary / "ghcn" / "daily.csv") as fh:
        for r in csv.DictReader(fh):
            if r.get("PRCP"):
                acc.setdefault(r["DATE"], []).append(float(r["PRCP"]))
    return {dt.date.fromisoformat(k): float(np.mean(v)) for k, v in acc.items()}


def acquisition_weather(when, times, era, rain):
    i = int(np.searchsorted(times, np.datetime64(when.replace(tzinfo=None), "h")))
    days = [when.date() - dt.timedelta(days=k) for k in range(7)]
    r = [rain[d] for d in days if d in rain]
    return {"sm0_7": float(era["soil_moisture_0_to_7cm"][i]),
            "rain7d": float(np.sum(r)) if r else np.nan}


def main(args):
    summary = Path(args.scan_run) / "summary.csv"
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), [summary],
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    anc = args.ancillary or cli.data_root() / "ancillary"
    with open(anc / "uavsar" / "acquisition_times.csv") as fh:
        when = [dt.datetime.fromisoformat(r["start_utc"]) for r in csv.DictReader(fh)]
    times, era = era5_scene(anc)
    rain = ghcn_rain(anc)
    acq = [acquisition_weather(w, times, era, rain) for w in when]
    share = {}
    with open(summary) as fh:
        for r in csv.DictReader(fh):
            if r["window"] == args.window and r["class_def"] == args.class_def:
                share[(int(r["triplet"]), r["class"])] = (float(r[f"rate_{args.pfa:g}"]),
                                                          int(r["zones"]))
    table = []
    for t in sorted({k[0] for k in share}):
        w3, a3 = when[t:t + 3], acq[t:t + 3]
        row = {"triplet": t, **{f"date{k + 1}": w3[k].date().isoformat() for k in range(3)}}
        for c in args.classes:
            rate, n = share.get((t, c), (np.nan, 0))
            row[f"share_{c}"], row[f"zones_{c}"] = rate, n
        sm, r7 = [a["sm0_7"] for a in a3], [a["rain7d"] for a in a3]
        row.update({"sm0_7_last": sm[2], "sm0_7_maxdiff": float(np.max(sm) - np.min(sm)),
                    "rain7d_last": r7[2], "rain7d_maxdiff": float(np.nanmax(r7) - np.nanmin(r7))})
        table.append(row)
    cors = []
    for c in args.classes:
        y = np.array([r[f"share_{c}"] for r in table], float)
        for k in WEATHER:
            x = np.array([r[k] for r in table], float)
            m = np.isfinite(x) & np.isfinite(y) & (np.array([r[f"zones_{c}"] for r in table])
                                                   >= args.min_zones)
            if m.sum() > 8:
                rho, p = stats.spearmanr(y[m], x[m])
                cors.append({"class": c, "variable": k, "triplets": int(m.sum()),
                             "spearman": float(rho), "p": float(p)})
                logger.info("%s %s: rho %.2f (%d triplets)", c, k, rho, m.sum())
    for name, data in (("triplets", table), ("correlations", cors)):
        with open(run.small_dir / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    finish_run(run, {"triplets": run.small_dir / "triplets.csv",
                     "correlations": run.small_dir / "correlations.csv"},
               extra={"ancillary": str(anc)})
    logger.info("done: %s", run.small_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan-run", type=Path, required=True, help="Run of zone_scan.py.")
    parser.add_argument("--ancillary", type=Path, default=None,
                        help="Default: $DATA_ROOT/ancillary.")
    parser.add_argument("--window", default="15x5")
    parser.add_argument("--class-def", default="gbar_indep", choices=["gbar_indep", "gbar_own"])
    parser.add_argument("--classes", nargs="+", default=["0.5-0.7", "0.7-0.9", "0.9-1.01"])
    parser.add_argument("--pfa", type=float, default=0.01)
    parser.add_argument("--min-zones", type=int, default=1000)
    cli.add_run_arguments(parser, task="closure-weather")
    args = parser.parse_args()
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
