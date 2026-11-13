r"""Coverage of the confidence arcs, bias of $\hat\mu$ and false-alarm rate, under $H_0$ and $H_1$,
in UAVSAR conditions and outside the model (TGRS plan, E1 and E2).

Zones are simulated as in :mod:`pfa_power` (coherences of the real cells of triplet ``--triplet``,
speckle filtered by the measured spectrum, one field per zone cut into touching cells), with one
perturbation per ``--scenario``:

* ``touching``: the reference (one field per zone, as on the images);
* ``independent``: one field per cell (the model of the test);
* ``white``: independent pixels;
* ``texture``: Gamma texture of shape ``--nu`` per pixel, common to the dates (K amplitude);
* ``mixed``: coherence varying within the cell, the right half of its pixels has the three
  coherences multiplied by ``--mix-factor``;
* ``ramp``: closure varying linearly in range across the zone, from $\mu-$``--spread`` to
  $\mu+$``--spread``;
* ``split``: the first rows of the zone at $\mu+$``--spread``, the last rows at
  $\mu-$``--spread``, the middle row (odd zones) at $\mu$.

The target of the arcs is the mean closure $\mu$ of the zone. Closures ``--closures`` (degrees)
x classes of the zone coherence x cell sizes ``--factors``. Methods (centre, half-width):

* ``scale_free``: weighted circular mean, arc (eq. arc) of the exact statistic; test at ``--pfa``
  with the $F(1, L-1)$ threshold;
* ``nominal``, ``plugin``: same centre, arc of $2c(R-C)$ with $c=1$ and $c=1/A$ ($\chi^2_1$);
* ``equal``: unweighted circular mean, arc of the scale-free statistic with unit weights;
* ``bootstrap``: weighted circular mean, percentile half-width of ``--boot`` resamplings of the
  cells of the zone (no test).

Outputs (run directory): ``coverage.csv`` (scenario, window, zone, class, closure_deg, zones,
method, reject, coverage, median_halfwidth_deg, mean_error_deg, median_error_deg, rmse_deg)
and ``summary.json`` (correlation areas, thresholds, valid zones).

Example:
    uv run python experiments/detection/coverage.py --store <run>/scan.zarr \
        --spectrum <run>/spectrum.npz --scenario texture --nu 1 --task coverage-texture1-s1
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import zarr
from pfa_power import real_zones  # same directory
from scipy import stats

from closure_glrt import cli
from closure_glrt.closure import neighbor_concentration, triplet_quantities
from closure_glrt.glrt import (
    confidence_halfwidth,
    glrt,
    glrt_known_scale,
    known_scale_halfwidth,
    threshold,
)
from closure_glrt.provenance import finish_run, start_run
from closure_glrt.simulation import (
    coherence_matrix,
    correlation_area,
    kernel_from_spectrum,
    simulate_zones,
)

logger = logging.getLogger(__name__)
METHODS = ("scale_free", "scale_free_nb", "nominal", "plugin", "equal", "bootstrap")
SCENARIOS = ("touching", "independent", "white", "texture", "mixed", "ramp", "split")


def wrap(x):
    return np.angle(np.exp(1j * x))


def cell_closures(mu, zone, scenario, spread):
    """Closure of every cell (L,) for a zone of mean closure mu (rad)."""
    za, zb = zone
    if scenario == "ramp":
        offs = np.tile(np.linspace(-spread, spread, zb), za)
    elif scenario == "split":
        rows = np.repeat(np.arange(za), zb)
        offs = np.where(rows < za // 2, spread, np.where(rows >= za - za // 2, -spread, 0.0))
    else:
        offs = np.zeros(za * zb)
    return mu + offs


def valid(g, closures, factor=None):
    """Zones whose cell matrices stay positive definite with the closures (and population 2)."""
    ok = (np.linalg.eigvalsh(coherence_matrix(g[..., 0], g[..., 1], g[..., 2], closures))
          > 1e-6).all(-1).all(-1)
    if factor is not None:
        ga = g * factor
        ok &= (np.linalg.eigvalsh(coherence_matrix(ga[..., 0], ga[..., 1], ga[..., 2], closures))
               > 1e-6).all(-1).all(-1)
    return ok


def simulate(g, window, zone, rng, h_az, h_rg, closures, batch, args):
    R = coherence_matrix(g[..., 0], g[..., 1], g[..., 2])
    R_alt = None
    if args.scenario == "mixed":
        ga = g * args.mix_factor
        R_alt = coherence_matrix(ga[..., 0], ga[..., 1], ga[..., 2])
    texture = args.nu if args.scenario == "texture" else None
    B, L = R.shape[:2]
    cl = np.broadcast_to(closures, (B, L))
    if args.scenario == "independent":
        R = R.reshape(B * L, 1, 3, 3)
        cl = cl.reshape(B * L, 1)
        zone, batch = (1, 1), batch * L
    out = []
    for s in range(0, len(R), batch):
        out.append(simulate_zones(R[s:s + batch], window, zone, rng, h_az, h_rg, cl[s:s + batch],
                                  texture=texture,
                                  R_alt=None if R_alt is None else R_alt[s:s + batch]))
    return np.concatenate(out).reshape(B, L, 3, 3)


def bootstrap_halfwidth(phi, w, mu_hat, rng, n_boot, alpha):
    """Percentile half-width of the weighted circular mean over resamplings of the cells."""
    Z, L = phi.shape
    idx = rng.integers(0, L, size=(n_boot, L))
    z = w * np.exp(1j * phi)                                    # (Z, L)
    hw = np.empty(Z)
    for s in range(0, Z, 500):
        mb = np.angle(z[s:s + 500][:, idx].sum(-1))             # (z, n_boot)
        hw[s:s + 500] = np.quantile(np.abs(wrap(mb - mu_hat[s:s + 500, None])), 1 - alpha, axis=1)
    return hw


def evaluate(lz, n_looks, c_eff, mu_true, args, rng, L, zone):
    phi, kappa, _ = (x[..., 0] for x in triplet_quantities(lz, n_looks))
    knb = neighbor_concentration(lz, n_looks, zone)
    ok = np.isfinite(phi).all(-1) & np.isfinite(knb).all(-1) & (knb > 0).all(-1)
    phi, kappa, knb = phi[ok], kappa[ok], knb[ok]
    eta, chi = threshold(L, args.pfa), stats.chi2.ppf(1 - args.pfa, 1)
    r = glrt(phi, kappa, details=True)
    ones = np.ones_like(kappa)
    re = glrt(phi, ones, details=True)
    rn = glrt(phi, knb, details=True)
    mu_w, mu_e = r["mu"], re["mu"]
    res = {
        "scale_free": (r["lam"] > eta, mu_w, confidence_halfwidth(phi, kappa, args.alpha)),
        "scale_free_nb": (rn["lam"] > eta, rn["mu"], confidence_halfwidth(phi, knb, args.alpha)),
        "nominal": (glrt_known_scale(phi, kappa, 1.0) > chi, mu_w,
                    known_scale_halfwidth(phi, kappa, 1.0, args.alpha)),
        "plugin": (glrt_known_scale(phi, kappa, c_eff) > chi, mu_w,
                   known_scale_halfwidth(phi, kappa, c_eff, args.alpha)),
        "equal": (re["lam"] > eta, mu_e, confidence_halfwidth(phi, ones, args.alpha)),
        "bootstrap": (np.full(len(phi), np.nan), mu_w,
                      bootstrap_halfwidth(phi, kappa, mu_w, rng, args.boot, args.alpha)),
    }
    out = {}
    for m, (rej, mu, hw) in res.items():
        err = wrap(mu - mu_true)
        rej_rate = float(np.nanmean(rej)) if m != "bootstrap" else ""
        out[m] = {"zones": int(len(phi)), "reject": rej_rate,
                  "coverage": float(np.mean(np.abs(err) <= hw)),
                  "median_halfwidth_deg": float(np.degrees(np.median(hw))),
                  "mean_error_deg": float(np.degrees(np.mean(err))),
                  "median_error_deg": float(np.degrees(np.median(err))),
                  "rmse_deg": float(np.degrees(np.sqrt(np.mean(err ** 2))))}
    return out


def main(args):
    rng = np.random.default_rng(args.seed)
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), [args.store, args.spectrum],
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    t0 = time.time()
    root = zarr.open_group(args.store, mode="r")
    br, bc = root.attrs["window"]
    lags = root[args.pol]["lags"]
    sp = np.load(args.spectrum)
    if args.scenario == "white":
        h_az, h_rg = np.ones(1), np.ones(1)
    else:
        h_az, h_rg = (kernel_from_spectrum(sp["s_az"], args.taps),
                      kernel_from_spectrum(sp["s_rg"], args.taps))
    zone, L = tuple(args.zone), args.zone[0] * args.zone[1]
    rows, summary = [], {"thresholds": {"scale_free": float(threshold(L, args.pfa)),
                                        "chi2_1": float(stats.chi2.ppf(1 - args.pfa, 1))}}
    spread = np.radians(args.spread)
    for a, b in args.factors:
        window, n_looks = (a * br, b * bc), a * b * br * bc
        wname = f"{window[0]}x{window[1]}"
        g_all, gbar = real_zones(lags, (a, b), zone, args.triplet)
        A = correlation_area(np.outer(h_az, h_rg), window)
        c_eff = 1.0 / A
        summary[f"{wname}|correlation_area"] = A
        for lo, hi in zip(args.classes[:-1], args.classes[1:]):
            idx0 = np.flatnonzero((gbar >= lo) & (gbar < hi))
            if len(idx0) < 100:
                continue
            for mu_deg in args.closures:
                cl = cell_closures(np.radians(mu_deg), zone, args.scenario, spread)
                idx = idx0[valid(g_all[idx0], cl,
                                 args.mix_factor if args.scenario == "mixed" else None)]
                cname = f"{lo:g}-{hi:g}"
                summary[f"{wname}|{cname}|{mu_deg:g}|valid_zones"] = [int(len(idx)), int(len(idx0))]
                if len(idx) < 100:
                    continue
                pick = rng.choice(idx, args.zones, replace=len(idx) < args.zones)
                lz = simulate(g_all[pick], window, zone, rng, h_az, h_rg, cl, args.batch, args)
                out = evaluate(lz, n_looks, c_eff, np.radians(mu_deg), args, rng, L, zone)
                for m in METHODS:
                    rows.append({"scenario": args.scenario, "window": wname,
                                 "zone": f"{zone[0]}x{zone[1]}", "class": cname,
                                 "closure_deg": mu_deg, "method": m, **out[m]})
                logger.info("%s class %s mu=%g: cov %s rej %s, %.1f min", wname, cname, mu_deg,
                            {m: round(out[m]["coverage"], 3) for m in METHODS},
                            round(out["scale_free"]["reject"], 4), (time.time() - t0) / 60)
    with open(run.small_dir / "coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (run.small_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    finish_run(run, {"coverage": run.small_dir / "coverage.csv",
                     "summary": run.small_dir / "summary.json"},
               extra={"elapsed_min": (time.time() - t0) / 60})
    logger.info("done: %s", run.small_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True, help="UAVSAR scan store.")
    parser.add_argument("--spectrum", type=Path, required=True)
    parser.add_argument("--scenario", default="touching", choices=SCENARIOS)
    parser.add_argument("--nu", type=float, default=1.0, help="Texture shape (scenario texture).")
    parser.add_argument("--mix-factor", type=float, default=0.5,
                        help="Coherence factor of the second population (scenario mixed).")
    parser.add_argument("--spread", type=float, default=1.0,
                        help="Half-range of the closure within the zone, degrees (ramp, split).")
    parser.add_argument("--pol", default="VV")
    parser.add_argument("--triplet", type=int, default=10)
    parser.add_argument("--factors", type=int, nargs=2, action="append", default=None)
    parser.add_argument("--zone", type=int, nargs=2, default=[5, 5])
    parser.add_argument("--classes", type=float, nargs="+",
                        default=[0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01])
    parser.add_argument("--closures", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--zones", type=int, default=4000,
                        help="Simulated zones per cell of the grid.")
    parser.add_argument("--pfa", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.05, help="Level of the arcs.")
    parser.add_argument("--boot", type=int, default=200)
    parser.add_argument("--taps", type=int, default=8)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    cli.add_run_arguments(parser, task="coverage")
    args = parser.parse_args()
    args.factors = args.factors or [[1, 1], [2, 2]]
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
