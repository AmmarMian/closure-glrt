r"""Split-zone check of the confidence arcs on real data (TGRS, real-data check of the interval).

Each zone of $z_a\times z_b$ cells is cut into its two checkerboard halves (cells with $i+j$ even
and odd). With the neighbor precisions of the zone
(:func:`closure_glrt.closure.neighbor_concentration`, computed on the whole zone as in the
paper), each half gives an estimate $\hat\mu_h$ of the closure and the half-width $\Delta_h$ of its
$1-\alpha$ arc (:func:`confidence_halfwidth`). Both
halves estimate the closure of the same zone from disjoint cells, so that, if the arcs are right,
$z=\operatorname{wrap}(\hat\mu_A-\hat\mu_B)/\sqrt{\sigma_A^2+\sigma_B^2}$, with
$\sigma_h=\Delta_h/z_{1-\alpha/2}$, is close to standard normal. Reported per class of the zone
coherence $\bar\gamma$ (weakest coherence of the cells of the tested triplet, median over the zone):
the share of zones with $|z|\le z_{1-\alpha/2}$ (expected $1-\alpha$), with $|z|\le1$ (expected
$68\%$), and the median $|z|/0.6745$ (expected 1). Zones where an arc is the whole circle are left
out.

Real data: the triplets ``--triplets`` of the scan store. Reference: zones simulated as in
:mod:`experiments.detection.coverage` (coherences of real zones of ``--sim-triplet``, measured
speckle spectrum, touching cells) with closures ``--sim-closures``. Halves of touching cells share
border pixels, which correlates their estimates in the same way in both.

Output: ``split.csv`` (source, window, triplet or closure, class, zones, cover95, cover68, scale).

Example:
    python experiments/scan/split_check.py --store <scan.zarr> --spectrum <spectrum.npz> \
        --backend torch-cuda --task split-check-s1
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import zarr
from scipy import stats

from closure_glrt import cli
from closure_glrt.backend import get_data_on_device, to_numpy
from closure_glrt.closure import (
    aggregate_cells,
    neighbor_concentration,
    tile_zones,
    triplet_quantities,
)
from closure_glrt.glrt import confidence_halfwidth, glrt
from closure_glrt.provenance import finish_run, start_run
from closure_glrt.simulation import kernel_from_spectrum

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detection"))
from coverage import simulate, valid  # noqa: E402
from pfa_power import real_zones  # noqa: E402

logger = logging.getLogger(__name__)


def halves(zone):
    za, zb = zone
    ij = np.add.outer(np.arange(za), np.arange(zb)).ravel()
    return np.flatnonzero(ij % 2 == 0), np.flatnonzero(ij % 2 == 1)


def split_stats(lz, n_looks, zone, alpha, backend):
    """z statistic per zone (nan where an arc is the whole circle) and the zone coherence."""
    phi, _, gmin = (x[..., 0] for x in triplet_quantities(lz, n_looks))
    with np.errstate(all="ignore"):
        knb = neighbor_concentration(lz, n_looks, zone)
    ok = np.isfinite(phi).all(-1) & np.isfinite(knb).all(-1) & (knb > 0).all(-1)
    phi, knb, gbar = phi[ok], knb[ok], np.median(gmin[ok], -1)
    q = stats.norm.ppf(1 - alpha / 2)
    mus, sig = [], []
    for idx in halves(zone):
        p = get_data_on_device(np.ascontiguousarray(phi[:, idx]), backend)
        w = get_data_on_device(np.ascontiguousarray(knb[:, idx]), backend)
        mus.append(to_numpy(glrt(p, w, backend, details=True)["mu"]).astype(np.float64))
        h = to_numpy(confidence_halfwidth(p, w, alpha, backend)).astype(np.float64)
        sig.append(np.where(h < np.pi - 1e-6, h / q, np.nan))
    d = np.angle(np.exp(1j * (mus[0] - mus[1])))
    return d / np.sqrt(sig[0] ** 2 + sig[1] ** 2), gbar


def summarize(z, gbar, classes, alpha, **key):
    q = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for lo, hi in zip(classes[:-1], classes[1:]):
        s = np.isfinite(z) & (gbar >= lo) & (gbar < hi)
        n = int(s.sum())
        if n < 50:
            continue
        a = np.abs(z[s])
        rows.append({**key, "class": f"{lo:g}-{hi:g}", "zones": n,
                     "cover95": float((a <= q).mean()), "cover68": float((a <= 1).mean()),
                     "scale": float(np.median(a) / stats.norm.ppf(0.75))})
    return rows


def main(args):
    rng = np.random.default_rng(args.seed)
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), [args.store, args.spectrum],
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    t0 = time.time()
    root = zarr.open_group(args.store, mode="r")
    br, bc = root.attrs["window"]
    lags = root[args.pol]["lags"]
    zone = tuple(args.zone)
    sp = np.load(args.spectrum)
    h_az, h_rg = (kernel_from_spectrum(sp["s_az"], args.taps),
                  kernel_from_spectrum(sp["s_rg"], args.taps))
    rows = []
    for a, b in args.factors:
        window, n_looks = (a * br, b * bc), a * b * br * bc
        wname = f"{window[0]}x{window[1]}"
        for t in args.triplets:
            La = aggregate_cells(np.asarray(lags[:, :, t:t + 3], dtype=np.complex128), (a, b))
            lz = tile_zones(La, zone)
            lz = lz.reshape(-1, *lz.shape[2:])
            z, gbar = split_stats(lz, n_looks, zone, args.alpha, args.backend)
            rows += summarize(z, gbar, args.classes, args.alpha, source="real", window=wname,
                              key=f"t={t}")
            logger.info("%s t=%d: %s, %.1f min", wname, t,
                        [(r["class"], round(r["cover95"], 3)) for r in rows[-6:]],
                        (time.time() - t0) / 60)
        g_all, gb_all = real_zones(lags, (a, b), zone, args.sim_triplet)
        for mu in args.sim_closures:
            for lo, hi in zip(args.classes[:-1], args.classes[1:]):
                idx = np.flatnonzero((gb_all >= lo) & (gb_all < hi))
                if len(idx) < 100:
                    continue
                cl = np.full(zone[0] * zone[1], np.radians(mu))
                idx = idx[valid(g_all[idx], cl)]
                if len(idx) < 100:
                    continue
                pick = rng.choice(idx, args.sim_zones, replace=len(idx) < args.sim_zones)
                lz = simulate(g_all[pick], window, zone, rng, h_az, h_rg, cl, 500,
                              argparse.Namespace(scenario="touching", nu=1.0, mix_factor=0.5))
                z, gbar = split_stats(lz, n_looks, zone, args.alpha, args.backend)
                rows += summarize(z, np.full(len(z), (lo + hi) / 2), [lo, hi], args.alpha,
                                  source="sim", window=wname, key=f"mu={mu:g}")
            logger.info("%s sim mu=%g done, %.1f min", wname, mu, (time.time() - t0) / 60)
    with open(run.small_dir / "split.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    finish_run(run, {"split": run.small_dir / "split.csv"},
               extra={"elapsed_min": (time.time() - t0) / 60})
    logger.info("done: %s", run.small_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--spectrum", type=Path, required=True)
    parser.add_argument("--pol", default="VV")
    parser.add_argument("--factors", type=int, nargs=2, action="append", default=None)
    parser.add_argument("--zone", type=int, nargs=2, default=[5, 5])
    parser.add_argument("--triplets", type=int, nargs="+",
                        default=[10, 23, 0, 5, 12, 13, 20, 30, 40, 50, 60, 65])
    parser.add_argument("--classes", type=float, nargs="+",
                        default=[0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sim-triplet", type=int, default=10)
    parser.add_argument("--sim-closures", type=float, nargs="+", default=[0.0, 2.0])
    parser.add_argument("--sim-zones", type=int, default=4000)
    parser.add_argument("--taps", type=int, default=8)
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--seed", type=int, default=0)
    cli.add_run_arguments(parser, task="split-check")
    args = parser.parse_args()
    args.factors = args.factors or [[1, 1], [2, 2]]
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
