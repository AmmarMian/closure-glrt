r"""Fit of the closure model $\hat\Phi_\ell\sim\mathcal{VM}(\mu,c\,\kappa_{0,\ell})$ on UAVSAR.

Section III-C of the paper.

Each zone of $L$ adjacent cells is fitted as in the test: one direction $\hat\mu$ and one scale
$\hat c$ per zone (maximum likelihood, :func:`closure_glrt.glrt.glrt` with ``details=True``),
and the precision $\kappa_{0,\ell}$ of each cell from its own coherences. The residuals
$d_\ell=\operatorname{wrap}(\hat\Phi_\ell-\hat\mu)$ of all zones are then on a common scale:

* standardized residuals $z_\ell=d_\ell\sqrt{\hat c\,\kappa_{0,\ell}}$ and probability integral
  transforms $u_\ell=F(d_\ell)$ under three laws: ``vm_c`` $\mathcal{VM}(0,\hat c\kappa_{0,\ell})$
  (the model), ``vm_1`` $\mathcal{VM}(0,\kappa_{0,\ell})$ (scale one, nominal looks), ``wn_c``
  wrapped normal of variance $1/(\hat c\kappa_{0,\ell})$ (another shape, same scale). Per class
  of the cell's weakest coherence: KS distance of $u$ to the uniform law, and with
  $s=\Phi^{-1}(u)$ the mean of $s^2$ (1), $P(|s|>2)$ (0.0455) and $P(|s|>3)$ (0.0027);
* precision against $\kappa_{0,\ell}$: cells binned by $\kappa_{0,\ell}$; in each bin the
  concentration $\hat\kappa$ of the residuals ($A(\hat\kappa)=$ mean of $\cos d$), divided by the
  bin's median $\kappa_{0}$ (the model predicts a flat profile at the level $c$; $c=1$ is the
  nominal number of looks), and the 95 % quantile of $|d|$, empirical and predicted by
  $\mathcal{VM}(0,\hat c\kappa_{0,\ell})$ and by $\mathcal{VM}(0,\kappa_{0,\ell})$.

The class ``all`` pools the cells of every class; ``scores.npz`` also holds the histograms of the
normal scores ($\ldots$``|<law>|hist`` on ``z_edges``).

$\hat\mu$ and $\hat c$ are fitted on the same cells, which pulls the residuals in slightly
($\approx 1/L$). The same pipeline is applied to zones simulated under the model in UAVSAR
conditions (``--spectrum``: measured speckle spectrum, coherences of the real cells of the same
triplet, adjacent cells of one filtered field, closure 0), which is the reference: real data fit
the model as well as the model fits itself if the two agree.

Outputs (run directory): ``fit.csv`` (one row per source, window, triplet, class and law),
``profiles.npz`` (precision profiles), ``scores.npz`` (normal-score quantiles at ``probs`` and
histograms of $z$, per source, window, triplet and class), ``summary.json``.

Example:
    uv run python experiments/fit/closure_fit.py --store <run>/scan.zarr \
        --spectrum <run>/spectrum.npz --task closure-fit-s1
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import zarr
from scipy import special, stats

from closure_glrt import cli
from closure_glrt.closure import (
    aggregate_cells,
    neighbor_concentration,
    tile_zones,
    triplet_quantities,
)
from closure_glrt.directional import pit, wrap
from closure_glrt.glrt import glrt
from closure_glrt.provenance import finish_run, start_run
from closure_glrt.simulation import coherence_matrix, kernel_from_spectrum, simulate_zones

logger = logging.getLogger(__name__)
LAWS = ("vm_c", "vm_1", "wn_c")
PROBS = np.concatenate([[1e-4, 3e-4, 1e-3, 3e-3], np.linspace(0.01, 0.99, 99),
                        [0.997, 0.999, 0.9997, 0.9999]])
Z_EDGES = np.linspace(-6, 6, 121)


def coherences(lags):
    """Sample coherences (g12, g23, g13) of the first triplet of ``lags`` ``(..., 3, 3)``."""
    P = lags[..., 0].real
    return (np.abs(lags[..., 0, 1]) / np.sqrt(P[..., 0] * P[..., 1]),
            np.abs(lags[..., 1, 1]) / np.sqrt(P[..., 1] * P[..., 2]),
            np.abs(lags[..., 0, 2]) / np.sqrt(P[..., 0] * P[..., 2]))


def zone_cells(lags, n_looks):
    """Zone lags ``(B, L, 3, 3)`` -> phi, kappa0, gmin ``(B, L)``; zones with a bad cell dropped."""
    phi, kappa, gmin = (x[..., 0] for x in triplet_quantities(lags, n_looks))
    ok = np.isfinite(phi).all(-1)
    return phi[ok], kappa[ok], gmin[ok], ok


def fit_zones(phi, kappa):
    """Per-zone ML fit: residuals d, scale c (broadcast to cells), standardized z."""
    r = glrt(phi, kappa, details=True)
    c = np.broadcast_to(r["c1"][:, None], phi.shape)
    d = wrap(phi - r["mu"][:, None])
    return d, c, d * np.sqrt(c * kappa)


def a_inverse(r):
    """Concentration with mean resultant length ``r`` (A(kappa) = I1/I0), by bisection in log."""
    r = np.asarray(r, dtype=np.float64)
    lo, hi = np.full(r.shape, -10.0), np.full(r.shape, 15.0)
    for _ in range(80):
        mid = (lo + hi) / 2
        k = np.exp(mid)
        up = special.i1e(k) / special.i0e(k) < r
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return np.exp((lo + hi) / 2)


def abs_quantile(kappa, level=0.95):
    """Quantile ``level`` of |d| for a mixture of VM(0, kappa_i) with equal weights (bisection)."""
    lo, hi = 0.0, np.pi
    for _ in range(40):
        q = (lo + hi) / 2
        inside = np.mean(2 * stats.vonmises.cdf(q, kappa) - 1)
        lo, hi = (q, hi) if inside < level else (lo, q)
    return (lo + hi) / 2


def precision_profile(d, kappa, c, n_bins=24, min_cells=2000, model_cells=3000, rng=None):
    """Bins of log kappa0: median kappa0, n, concentration of d / median kappa0, and the 95 %
    quantile of |d|: empirical, under the model VM(0, c kappa0) and under VM(0, kappa0)."""
    lk = np.log10(kappa)
    edges = np.quantile(lk, np.linspace(0.0005, 0.9995, n_bins + 1))
    idx = np.digitize(lk, edges) - 1
    rng = rng or np.random.default_rng(0)
    out = {k: [] for k in ("kappa0", "scale", "n", "q95", "q95_model", "q95_nominal")}
    for b in range(n_bins):
        sel = np.flatnonzero(idx == b)
        if len(sel) < min_cells:
            continue
        km = float(np.median(kappa[sel]))
        sub = rng.choice(sel, min(model_cells, len(sel)), replace=False)
        out["kappa0"].append(km)
        out["scale"].append(float(a_inverse(np.mean(np.cos(d[sel])))) / km)
        out["n"].append(len(sel))
        out["q95"].append(float(np.quantile(np.abs(d[sel]), 0.95)))
        out["q95_model"].append(abs_quantile(c[sub] * kappa[sub]))
        out["q95_nominal"].append(abs_quantile(kappa[sub]))
    return {k: np.array(v) for k, v in out.items()}


def evaluate(writer, scores, key, d, kappa, c, z, gmin, classes, max_cells, rng):
    base = "|".join(map(str, key))
    bounds = list(zip(classes[:-1], classes[1:])) + [(classes[0], classes[-1] + 1e-9)]
    for lo, hi in bounds:
        sel = np.flatnonzero((gmin >= lo) & (gmin < hi))
        if len(sel) < 100:
            continue
        if len(sel) > max_cells:
            sel = rng.choice(sel, max_cells, replace=False)
        cl = f"{lo:g}-{hi:g}" if hi <= classes[-1] else "all"
        scores[f"{base}|{cl}|z_hist"] = np.histogram(z[sel], Z_EDGES, density=True)[0]
        for law in LAWS:
            u = np.clip(pit(d[sel], kappa[sel], c[sel], law), 1e-12, 1 - 1e-12)
            s = stats.norm.ppf(u)
            writer.writerow({
                "source": key[0], "window": key[1], "triplet": key[2], "class": cl,
                "cells": len(sel), "law": law, "median_c": float(np.median(c[sel])),
                "ks": float(stats.kstest(u, "uniform").statistic),
                "mean_s2": float((s**2).mean()), "p_abs_gt2": float((np.abs(s) > 2).mean()),
                "p_abs_gt3": float((np.abs(s) > 3).mean())})
            scores[f"{base}|{cl}|{law}"] = np.quantile(s, PROBS)
            scores[f"{base}|{cl}|{law}|hist"] = np.histogram(s, Z_EDGES, density=True)[0]


def simulated_lags(lags_real, window, zone, n_zones, h_az, h_rg, rng, batch=1000):
    """Zones simulated under the model with the coherences of random real zones (closure 0)."""
    g12, g23, g13 = coherences(lags_real)
    R = coherence_matrix(g12, g23, g13)
    pd = (np.linalg.eigvalsh(R) > 1e-6).all(-1).all(-1)
    R = R[pd]
    R = R[rng.choice(len(R), size=min(n_zones, len(R)), replace=False)]
    return np.concatenate([simulate_zones(R[s:s + batch], window, zone, rng, h_az, h_rg)
                           for s in range(0, len(R), batch)])


def main(args):
    rng = np.random.default_rng(args.seed)
    inputs = [args.store] + ([args.spectrum] if args.spectrum else [])
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), inputs,
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    fh = open(run.small_dir / "fit.csv", "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=["source", "window", "triplet", "class", "cells", "law",
                                            "median_c", "ks", "mean_s2", "p_abs_gt2", "p_abs_gt3"])
    writer.writeheader()
    profiles, scores, summary, t0 = {}, {"probs": PROBS, "z_edges": Z_EDGES}, [], time.time()
    zone = tuple(args.zone)
    root = zarr.open_group(args.store, mode="r")
    br, bc = root.attrs["window"]
    lags = root[args.pol]["lags"]
    if args.spectrum:
        sp = np.load(args.spectrum)
        h_az = kernel_from_spectrum(sp["s_az"], args.taps)
        h_rg = kernel_from_spectrum(sp["s_rg"], args.taps)

    def process(source, wname, t, Lz, n_looks):
        phi, kappa, gmin, ok = zone_cells(Lz, n_looks)
        if args.weights == "nb":    # precision of each cell from its neighbors in the zone
            with np.errstate(all="ignore"):
                kappa = neighbor_concentration(Lz[ok], n_looks, zone)
            good = np.isfinite(kappa).all(-1) & (kappa > 0).all(-1)
            phi, kappa, gmin = phi[good], kappa[good], gmin[good]
        d, c, z = fit_zones(phi, kappa)
        key = (source, wname, t)
        evaluate(writer, scores, key, d.ravel(), kappa.ravel(), c.ravel(), z.ravel(), gmin.ravel(),
                 args.classes, args.max_cells, rng)
        base = "|".join(map(str, key))
        prof = precision_profile(d.ravel(), kappa.ravel(), c.ravel(), rng=rng)
        profiles.update({f"{base}|{k}": v for k, v in prof.items()})
        cz = c[:, 0]
        summary.append({"source": source, "window": wname, "triplet": t, "zones": int(len(phi)),
                        "c_quartiles": np.quantile(cz[np.isfinite(cz)],
                                                   [0.25, 0.5, 0.75]).tolist()})
        fh.flush()
        logger.info("%s %s t=%d: %d zones, median c %.3f, %.1f min", source, wname, t, len(phi),
                    float(np.nanmedian(cz)), (time.time() - t0) / 60)

    for a, b in args.factors:
        window, n_looks = (a * br, b * bc), a * b * br * bc
        wname = f"{window[0]}x{window[1]}"
        for t in args.triplets:
            La = aggregate_cells(np.asarray(lags[:, :, t:t + 3], dtype=np.complex128), (a, b))
            Lz = tile_zones(La, zone).reshape(-1, zone[0] * zone[1], 3, La.shape[-1])
            process("uavsar", wname, t, Lz, n_looks)
            if args.spectrum and t in args.sim_triplets:
                _, _, _, ok = zone_cells(Lz, n_looks)
                Ls = simulated_lags(Lz[ok], window, zone, args.sim_zones, h_az, h_rg, rng)
                process("sim_h0", wname, t, Ls, n_looks)
    fh.close()
    np.savez_compressed(run.small_dir / "profiles.npz", **profiles)
    np.savez_compressed(run.small_dir / "scores.npz", **scores)
    (run.small_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    finish_run(run, {"fit": run.small_dir / "fit.csv", "profiles": run.small_dir / "profiles.npz",
                     "scores": run.small_dir / "scores.npz",
                     "summary": run.small_dir / "summary.json"},
               extra={"elapsed_min": (time.time() - t0) / 60})
    logger.info("done: %s", run.small_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True, help="UAVSAR scan store.")
    parser.add_argument("--pol", default="VV")
    parser.add_argument("--factors", type=int, nargs=2, action="append", default=None,
                        help="Cell as a multiple of the base cell (repeatable). Default: 1 1, 2 2.")
    parser.add_argument("--triplets", type=int, nargs="+", default=[10, 40, 12])
    parser.add_argument("--zone", type=int, nargs=2, default=[5, 5])
    parser.add_argument("--classes", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
                        help="Edges of the classes of the cells' weakest coherence.")
    parser.add_argument("--max-cells", type=int, default=300_000, help="Cells per class (random).")
    parser.add_argument("--spectrum", type=Path, default=None,
                        help="spectrum.npz of slc_spectrum.py: adds the simulated reference.")
    parser.add_argument("--sim-triplets", type=int, nargs="+", default=[10])
    parser.add_argument("--sim-zones", type=int, default=20_000)
    parser.add_argument("--taps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", default="own", choices=["own", "nb"],
                        help="Precisions of the cells: own coherences, or those of the neighbors.")
    cli.add_run_arguments(parser, task="closure-fit")
    args = parser.parse_args()
    args.factors = args.factors or [[1, 1], [2, 2]]
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
