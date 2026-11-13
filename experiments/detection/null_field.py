r"""Whole-scene null field: detections, clusters and sign agreement under $H_0$ against real data
(TGRS plan, E4).

A strip of ``--zones-az`` zones in azimuth and the full width of the scan (triplet ``--triplet``,
cells of the base size) is simulated under $H_0$: every cell keeps the three sample coherences of
the real cell at the same place, and one speckle field filtered by the measured spectrum covers
each block of ``--block`` zone rows, so that the cells and the zones touch as on the image
(blocks are independent; pairs that cross a block border are left out). The same statistics are
computed on the real strip:

* share of detected zones per class of $\bar\gamma$ (scale-free test at ``--pfa``);
* clusters of detected zones of class at least ``--coh`` (4-connectivity): share of the detected
  zones that lie in clusters of at least 2, 3, 5, 10 zones;
* sign agreement of $\hat\mu$ between detected zones of class at least ``--coh`` at a distance
  of 1, 2, 3 zones (azimuth and range).

Outputs: ``null_field.csv`` (source sim/real, replicate, quantity, key, value, n) and
``maps.npz`` (gbar, p and mu of the real strip and of the first replicate).

Example:
    uv run python experiments/detection/null_field.py --store <run>/scan.zarr \
        --spectrum <run>/spectrum.npz --task null-field-s1
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage

from closure_glrt import cli
from closure_glrt.closure import tile_zones, triplet_quantities
from closure_glrt.glrt import glrt, pvalue
from closure_glrt.provenance import finish_run, start_run
from closure_glrt.simulation import coherence_matrix, kernel_from_spectrum, simulate_zones

logger = logging.getLogger(__name__)


def cell_coherences(lags):
    """(I, J, 3) sample coherences |g12|, |g23|, |g13| of one triplet (lags (I, J, 3, K))."""
    P = lags[..., 0].real
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([np.abs(lags[..., 0, 1]) / np.sqrt(P[..., 0] * P[..., 1]),
                         np.abs(lags[..., 1, 1]) / np.sqrt(P[..., 1] * P[..., 2]),
                         np.abs(lags[..., 0, 2]) / np.sqrt(P[..., 0] * P[..., 2])], -1)


def zone_maps(lags, n_looks, zone):
    """p, mu-hat and gbar (median weakest coherence) of the zones of a cell grid of lags."""
    phi, kappa, gmin = (x[..., 0] for x in triplet_quantities(lags, n_looks))
    zp, zk, zg = (tile_zones(x, zone) for x in (phi, kappa, gmin))
    ok = np.isfinite(zp).all(-1) & np.isfinite(zk).all(-1)
    p = np.full(ok.shape, np.nan)
    mu = np.full(ok.shape, np.nan)
    r = glrt(zp[ok], zk[ok], details=True)
    p[ok] = pvalue(r["lam"], zp.shape[-1])
    mu[ok] = r["mu"]
    return p, mu, np.median(zg, -1)


def statistics(p, mu, gbar, classes, pfa, coh, block_rows):
    rows = []
    for lo, hi in zip(classes[:-1], classes[1:]):
        sel = np.isfinite(p) & (gbar >= lo) & (gbar < hi)
        if sel.any():
            rows.append(("share", f"{lo:g}-{hi:g}", float((p[sel] < pfa).mean()), int(sel.sum())))
    det = np.isfinite(p) & (p < pfa) & (gbar >= coh)
    lab, n = ndimage.label(det)
    sizes = ndimage.sum(det, lab, index=np.arange(1, n + 1)) if n else np.zeros(0)
    nd = int(det.sum())
    for k in (2, 3, 5, 10):
        rows.append(("cluster_share", f">={k}", float(sizes[sizes >= k].sum() / nd) if nd else
                     float("nan"), nd))
    s = np.sign(mu)
    blk = np.arange(det.shape[0]) // block_rows
    for d in (1, 2, 3):
        same_blk = (blk[:-d] == blk[d:])[:, None]
        pairs = [(det[:-d] & det[d:] & same_blk, s[:-d] == s[d:]),
                 (det[:, :-d] & det[:, d:], s[:, :-d] == s[:, d:])]
        npair = sum(int(m.sum()) for m, _ in pairs)
        same = sum(int((m & e).sum()) for m, e in pairs)
        rows.append(("same_sign", f"d={d}", same / npair if npair else float("nan"), npair))
    return rows


def main(args):
    rng = np.random.default_rng(args.seed)
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), [args.store, args.spectrum],
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    t0 = time.time()
    root = zarr.open_group(args.store, mode="r")
    br, bc = root.attrs["window"]
    n_looks = br * bc
    lags_all = root[args.pol]["lags"]
    za, zb = args.zone
    t = args.triplet
    sp = np.load(args.spectrum)
    h_az, h_rg = (kernel_from_spectrum(sp["s_az"], args.taps),
                  kernel_from_spectrum(sp["s_rg"], args.taps))
    # pick the strip: the azimuth window of zones with the most coherent zones
    n_rows, J = lags_all.shape[:2]
    gmin_rows = []
    step = za * args.zones_az
    for i0 in range(0, n_rows - step + 1, step):
        lg = np.asarray(lags_all[i0:i0 + step, :, t:t + 3], dtype=np.complex128)
        g = cell_coherences(lg).min(-1)
        gmin_rows.append((float(np.nanmean(g >= args.coh)), i0))
    frac, i0 = max(gmin_rows) if args.az_start is None else (None, args.az_start)
    logger.info("strip: cell rows %d-%d (coherent share %s)", i0, i0 + step, frac)
    lags = np.asarray(lags_all[i0:i0 + step, :, t:t + 3], dtype=np.complex128)
    Jz = (J // zb) * zb
    lags = lags[:, :Jz]
    p_r, mu_r, g_r = zone_maps(lags, n_looks, (za, zb))
    rows = [("real", 0, *r) for r in statistics(p_r, mu_r, g_r, args.classes, args.pfa,
                                                 args.coh, args.block)]
    g = cell_coherences(lags)                                     # (I, Jz, 3)
    bad = ~np.isfinite(g).all(-1)
    g = np.where(bad[..., None], 0.05, np.clip(g, 0.0, 0.999))
    R = coherence_matrix(g[..., 0], g[..., 1], g[..., 2])
    pd = (np.linalg.eigvalsh(R) > 1e-6).all(-1)
    R[~pd] = coherence_matrix(0.05, 0.05, 0.05)
    maps = {"real_p": p_r, "real_mu": mu_r, "gbar": g_r}
    cells_blk = args.block * za
    for rep in range(args.reps):
        sim = np.empty((step, Jz, 3, 3), np.complex128)
        for b0 in range(0, step, cells_blk):
            Rb = R[b0:b0 + cells_blk]                            # (rows, Jz, 3, 3)
            nr = Rb.shape[0]
            out = simulate_zones(Rb.reshape(1, nr * Jz, 3, 3), (br, bc), (nr, Jz), rng, h_az, h_rg)
            sim[b0:b0 + nr] = out.reshape(nr, Jz, 3, 3)
        p_s, mu_s, _ = zone_maps(sim, n_looks, (za, zb))
        p_s[bad.reshape(-1, za, Jz // zb, zb).any((1, 3))] = np.nan
        rows += [("sim", rep, *r) for r in statistics(p_s, mu_s, g_r, args.classes, args.pfa,
                                                       args.coh, args.block)]
        if rep == 0:
            maps.update(sim_p=p_s, sim_mu=mu_s)
        logger.info("replicate %d done, %.1f min", rep, (time.time() - t0) / 60)
    with open(run.small_dir / "null_field.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "replicate", "quantity", "key", "value", "n"])
        w.writerows(rows)
    np.savez_compressed(run.small_dir / "maps.npz", **maps)
    finish_run(run, {"null_field": run.small_dir / "null_field.csv",
                     "maps": run.small_dir / "maps.npz"},
               extra={"elapsed_min": (time.time() - t0) / 60, "cell_rows": [i0, i0 + step]})
    logger.info("done: %s", run.small_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--spectrum", type=Path, required=True)
    parser.add_argument("--pol", default="VV")
    parser.add_argument("--triplet", type=int, default=10)
    parser.add_argument("--zone", type=int, nargs=2, default=[5, 5])
    parser.add_argument("--zones-az", type=int, default=100, help="Strip length in zones.")
    parser.add_argument("--az-start", type=int, default=None,
                        help="First cell row of the strip (default: most coherent strip).")
    parser.add_argument("--block", type=int, default=20,
                        help="Zone rows per independently simulated block.")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--classes", type=float, nargs="+",
                        default=[0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01])
    parser.add_argument("--coh", type=float, default=0.5)
    parser.add_argument("--pfa", type=float, default=0.01)
    parser.add_argument("--taps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    cli.add_run_arguments(parser, task="null-field")
    args = parser.parse_args()
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
