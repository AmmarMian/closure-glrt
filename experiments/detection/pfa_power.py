r"""False-alarm rate and power of the closure tests in UAVSAR conditions (Section IV-E, Fig. 5).

Zones of $L$ adjacent cells are simulated with :func:`closure_glrt.simulation.simulate_zones`: the
three coherences of every cell are the sample coherences of a real zone of the scan (triplet
``--triplet``), drawn per class of the zone coherence $\bar\gamma$ (median over the cells of the
weakest coherence of the cell, ``--classes``); speckle is filtered with the measured spectrum
(``--spectrum``) or white (independent pixels). On the same zones, four tests at ``--pfa``:

* ``scale_free``: $\Lambda$ of :func:`closure_glrt.glrt.glrt`, threshold of the $F(1, L-1)$ law;
* ``nominal``: $2(R-C)$ with the scale fixed to 1, $\chi^2_1$ threshold (nominal looks);
* ``plugin``: $2c_{\mathrm{eff}}(R-C)$ with $c_{\mathrm{eff}}=1/A$, $A$ the correlation area of the
  simulation kernel on the cell (white: 1), $\chi^2_1$ threshold;
* ``equal``: $\Lambda$ with all the precisions set to 1, threshold of the $F(1, L-1)$ law.

False alarm: closure 0, every class, measured and white speckle. Power: closures ``--closures``
(degrees, common to the zone) in the zones of $\bar\gamma\ge$ ``--power-min-coh``, measured speckle.
With ``--independent-cells``, every cell is drawn from its own speckle field (the model of the
test: independent cells, correlated pixels within a cell); by default one field per zone is cut
into adjacent cells, as on the images, so that neighbouring cells are slightly dependent.

Outputs (run directory): ``pfa.csv`` (window, speckle, class, zones, test, rate), ``power.csv``
(window, closure_deg, zones, test, rate, median $|\hat\mu|$ of the scale-free fit),
``summary.json`` (correlation areas, thresholds).

Example:
    uv run python experiments/detection/pfa_power.py --store <run>/scan.zarr \
        --spectrum <run>/spectrum.npz --task pfa-power-s1
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import zarr
from scipy import stats

from closure_glrt import cli
from closure_glrt.closure import aggregate_cells, tile_zones, triplet_quantities
from closure_glrt.glrt import glrt, glrt_known_scale, threshold
from closure_glrt.provenance import finish_run, start_run
from closure_glrt.simulation import (
    coherence_matrix,
    correlation_area,
    kernel_from_spectrum,
    simulate_zones,
)

logger = logging.getLogger(__name__)
TESTS = ("scale_free", "nominal", "plugin", "equal")


def real_zones(lags, factor, zone, t):
    """Sample coherences (Z, L, 3) and zone coherence (Z,) of the complete zones of triplet t."""
    La = aggregate_cells(np.asarray(lags[:, :, t:t + 3], dtype=np.complex128), factor)
    Lz = tile_zones(La, zone).reshape(-1, zone[0] * zone[1], 3, La.shape[-1])
    P = Lz[..., 0].real
    g = np.stack([np.abs(Lz[..., 0, 1]) / np.sqrt(P[..., 0] * P[..., 1]),
                  np.abs(Lz[..., 1, 1]) / np.sqrt(P[..., 1] * P[..., 2]),
                  np.abs(Lz[..., 0, 2]) / np.sqrt(P[..., 0] * P[..., 2])], -1)
    ok = np.isfinite(g).all((-1, -2))
    g = g[ok]
    R = coherence_matrix(g[..., 0], g[..., 1], g[..., 2])
    pd = (np.linalg.eigvalsh(R) > 1e-6).all(-1).all(-1)
    g = g[pd]
    return g, np.median(g.min(-1), axis=1)


def run_tests(lags, n_looks, c_eff, pfa, L):
    """Decisions (Z,) of the four tests and mu-hat of the scale-free fit on simulated zones."""
    phi, kappa, _ = (x[..., 0] for x in triplet_quantities(lags, n_looks))
    ok = np.isfinite(phi).all(-1)
    phi, kappa = phi[ok], kappa[ok]
    eta, chi = threshold(L, pfa), stats.chi2.ppf(1 - pfa, 1)
    r = glrt(phi, kappa, details=True)
    dec = {"scale_free": r["lam"] > eta,
           "nominal": glrt_known_scale(phi, kappa, 1.0) > chi,
           "plugin": glrt_known_scale(phi, kappa, c_eff) > chi,
           "equal": glrt(phi, np.ones_like(kappa)) > eta}
    return dec, r["mu"]


def valid_with_closure(g, closure):
    """Zones whose cell coherence matrices stay positive definite with the closure added."""
    R = coherence_matrix(g[..., 0], g[..., 1], g[..., 2], closure)
    return (np.linalg.eigvalsh(R) > 1e-6).all(-1).all(-1)


def simulate(g, window, zone, rng, h_az, h_rg, closure, batch, independent=False):
    """Zone lags (B, L, 3, 3). ``independent``: each cell is drawn from its own speckle field,
    so that the cells of a zone are independent (the model of the test); otherwise one field
    per zone is cut into adjacent cells, as on the images."""
    R = coherence_matrix(g[..., 0], g[..., 1], g[..., 2])
    B, L = R.shape[:2]
    if independent:
        R = R.reshape(B * L, 1, 3, 3)
        zone, batch = (1, 1), batch * L
    out = np.concatenate([simulate_zones(R[s:s + batch], window, zone, rng, h_az, h_rg, closure)
                          for s in range(0, len(R), batch)])
    return out.reshape(B, L, 3, 3)


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
    kernels = {"measured": (kernel_from_spectrum(sp["s_az"], args.taps),
                            kernel_from_spectrum(sp["s_rg"], args.taps)),
               "white": (np.ones(1), np.ones(1))}
    zone, L = tuple(args.zone), args.zone[0] * args.zone[1]
    pfa_rows, power_rows, summary = [], [], {"thresholds": {
        "scale_free": float(threshold(L, args.pfa)),
        "chi2_1": float(stats.chi2.ppf(1 - args.pfa, 1))}}
    for a, b in args.factors:
        window, n_looks = (a * br, b * bc), a * b * br * bc
        wname = f"{window[0]}x{window[1]}"
        g_all, gbar = real_zones(lags, (a, b), zone, args.triplet)
        logger.info("%s: %d real zones", wname, len(g_all))
        for speckle, (h_az, h_rg) in kernels.items():
            A = correlation_area(np.outer(h_az, h_rg), window)
            c_eff = 1.0 / A
            summary[f"{wname}|{speckle}|correlation_area"] = A
            for lo, hi in zip(args.classes[:-1], args.classes[1:]):
                idx = np.flatnonzero((gbar >= lo) & (gbar < hi))
                if len(idx) < 100:
                    continue
                pick = rng.choice(idx, args.zones, replace=len(idx) < args.zones)
                lz = simulate(g_all[pick], window, zone, rng, h_az, h_rg, 0.0, args.batch,
                              args.independent_cells)
                dec, _ = run_tests(lz, n_looks, c_eff, args.pfa, L)
                for test in TESTS:
                    pfa_rows.append({"window": wname, "speckle": speckle, "class": f"{lo:g}-{hi:g}",
                                     "zones": int(len(dec[test])), "test": test,
                                     "rate": float(dec[test].mean())})
                logger.info("%s %s class %g-%g: %s, %.1f min", wname, speckle, lo, hi,
                            {k: round(float(v.mean()), 4) for k, v in dec.items()},
                            (time.time() - t0) / 60)
        h_az, h_rg = kernels["measured"]
        c_eff = 1.0 / summary[f"{wname}|measured|correlation_area"]
        idx0 = np.flatnonzero(gbar >= args.power_min_coh)
        for mu in args.closures:
            idx = idx0[valid_with_closure(g_all[idx0], np.radians(mu))]
            summary[f"{wname}|power|{mu:g}|valid_zones"] = [int(len(idx)), int(len(idx0))]
            pick = rng.choice(idx, args.power_zones, replace=len(idx) < args.power_zones)
            lz = simulate(g_all[pick], window, zone, rng, h_az, h_rg, np.radians(mu), args.batch,
                          args.independent_cells)
            dec, mu_hat = run_tests(lz, n_looks, c_eff, args.pfa, L)
            for test in TESTS:
                power_rows.append({"window": wname, "closure_deg": mu, "zones": int(len(dec[test])),
                                   "test": test, "rate": float(dec[test].mean()),
                                   "median_mu_hat_deg": float(np.degrees(np.median(mu_hat)))})
            logger.info("%s power mu=%g deg: %s, median mu-hat %.2f deg", wname, mu,
                        {k: round(float(v.mean()), 3) for k, v in dec.items()},
                        float(np.degrees(np.median(mu_hat))))
    for name, rows in (("pfa", pfa_rows), ("power", power_rows)):
        with open(run.small_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (run.small_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    finish_run(run, {"pfa": run.small_dir / "pfa.csv", "power": run.small_dir / "power.csv",
                     "summary": run.small_dir / "summary.json"},
               extra={"elapsed_min": (time.time() - t0) / 60})
    logger.info("done: %s", run.small_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True, help="UAVSAR scan store.")
    parser.add_argument("--spectrum", type=Path, required=True,
                        help="spectrum.npz of slc_spectrum.py")
    parser.add_argument("--pol", default="VV")
    parser.add_argument("--triplet", type=int, default=10)
    parser.add_argument("--factors", type=int, nargs=2, action="append", default=None,
                        help="Cell as a multiple of the base cell (repeatable). Default: 1 1, 2 2.")
    parser.add_argument("--zone", type=int, nargs=2, default=[5, 5])
    parser.add_argument("--classes", type=float, nargs="+",
                        default=[0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01],
                        help="Edges of the zone-coherence classes (median weakest coherence).")
    parser.add_argument("--zones", type=int, default=20_000, help="Simulated zones per class.")
    parser.add_argument("--closures", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    parser.add_argument("--power-min-coh", type=float, default=0.7)
    parser.add_argument("--power-zones", type=int, default=5000)
    parser.add_argument("--pfa", type=float, default=0.01)
    parser.add_argument("--taps", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--independent-cells", action="store_true",
                        help="One speckle field per cell (cells of a zone independent, as in the "
                             "model); default: one field per zone cut into adjacent cells.")
    parser.add_argument("--seed", type=int, default=0)
    cli.add_run_arguments(parser, task="pfa-power")
    args = parser.parse_args()
    args.factors = args.factors or [[1, 1], [2, 2]]
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
