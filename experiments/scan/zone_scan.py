r"""Scale-free closure test on every zone of every triplet of a UAVSAR scan (Section V).

For each cell size (a multiple of the base cell of the scan store) and each triplet of
consecutive dates $(t,t+1,t+2)$, every cell gives its closure $\hat\Phi_\ell$ and precision
$\kappa_{0,\ell}$ (:func:`closure_glrt.closure.triplet_quantities`, nominal $N$); the cell grid is
cut into zones of ``--zone`` cells (zones with an invalid cell are left out) and each zone gets
the exact statistic $\Lambda$, $\hat\mu$ and $\hat c_1$ (:func:`closure_glrt.glrt.glrt`) and the
p-value of the $F(1, L-1)$ threshold law.

Two zone-coherence classes are stored:

* ``gbar_own``: median over the cells of the weakest of the three coherences of the tested
  triplet (the class of the earlier runs; it depends on the tested data);
* ``gbar_indep``: median over the cells of the mean of the lag-1 coherences
  $|\hat\gamma_{t-2,t-1}|$ and $|\hat\gamma_{t+3,t+4}|$, pairs that share no date with the
  triplet (one of them at the ends of the series; nan if neither exists).

Outputs (run directory): ``zones.zarr`` (large dir), per window arrays ``(T-2, I, J)`` float32
``lam``, ``p``, ``mu``, ``c1``, ``gbar_own``, ``gbar_indep``; ``summary.csv`` (small dir), per
window, triplet, class definition and class: zones, detection rates at ``--pfa``, median
$|\hat\mu|$ (deg) and median $\hat c_1$ of all and of the detected zones.

Example:
    uv run python experiments/scan/zone_scan.py --store <run>/scan.zarr --backend torch-cuda \
        --task zone-scan-s1
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
import zarr

from closure_glrt import cli
from closure_glrt.backend import get_data_on_device, to_numpy
from closure_glrt.closure import (
    aggregate_cells,
    neighbor_concentration,
    tile_zones,
    triplet_quantities,
)
from closure_glrt.competitors import YuanLUT, yuan_zone, zone_multilook, zwieback_zone
from closure_glrt.glrt import confidence_halfwidth, glrt, glrt_known_scale, pvalue
from closure_glrt.provenance import finish_run, start_run
from closure_glrt.simulation import correlation_area, kernel_from_spectrum

logger = logging.getLogger(__name__)
FIELDS = ("lam", "p", "mu", "c1", "gbar_own", "gbar_indep",
          # TGRS additions: neighbor weights, arcs, competing tests
          "lam_nb", "p_nb", "mu_nb", "hw", "hw_nb", "lam_nominal", "lam_equal", "p_equal",
          "mu_equal", "zw", "yuan", "hw_yuan", "mu_zone")


def pair_coherence(lags, a):
    """|gamma_{a,a+1}| of every cell from the scan lags (I, J, T, K); nan if a is out of range."""
    T = lags.shape[2]
    if a < 0 or a + 1 >= T:
        return None
    P0, P1 = lags[:, :, a, 0].real, lags[:, :, a + 1, 0].real
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(lags[:, :, a, 1]) / np.sqrt(P0 * P1)


def triplet_zones(La, t, zone, n_looks, backend, extra=None):
    """Zone quantities of triplet t from the aggregated lags La (I, J, T, K)."""
    phi, kappa, gmin = (x[..., 0] for x in triplet_quantities(La[:, :, t:t + 3], n_looks))
    pairs = [g for g in (pair_coherence(La, t - 2), pair_coherence(La, t + 3)) if g is not None]
    gind = np.nanmean(np.stack(pairs), 0) if pairs else np.full(phi.shape, np.nan)
    zp, zk, zg, zi = (tile_zones(x, zone) for x in (phi, kappa, gmin, gind))
    ni, nj, L = zp.shape
    ok = np.isfinite(zp).all(-1)
    out = {k: np.full((ni, nj), np.nan, np.float32) for k in FIELDS}
    if ok.any():
        r = glrt(get_data_on_device(zp[ok], backend), get_data_on_device(zk[ok], backend),
                 backend, details=True)
        lam = to_numpy(r["lam"]).astype(np.float64)
        out["lam"][ok] = lam
        out["p"][ok] = pvalue(lam, L)
        out["mu"][ok] = to_numpy(r["mu"])
        out["c1"][ok] = to_numpy(r["c1"])
        out["gbar_own"][ok] = np.median(zg[ok], -1)
        with np.errstate(all="ignore"):
            out["gbar_indep"][ok] = np.nanmedian(zi[ok], -1)
        if extra is not None:
            _extra_fields(out, ok, zp[ok], zk[ok], tile_zones(La[:, :, t:t + 3], zone)[ok],
                          zone, n_looks, L, extra)
    return out


def _extra_fields(out, ok, phi, kappa, lz, zone, n_looks, L, extra):
    """Neighbor-weighted test and arc, arcs, nominal / equal-weights / Zwieback / Yuan statistics."""
    alpha = extra["alpha"]
    with np.errstate(all="ignore"):
        knb = neighbor_concentration(lz, n_looks, zone)
    good = np.isfinite(knb).all(-1) & (knb > 0).all(-1)
    sub = np.zeros(ok.shape, bool)
    sub[ok] = good
    rn = glrt(phi[good], knb[good], details=True)
    out["lam_nb"][sub] = rn["lam"]
    out["p_nb"][sub] = pvalue(rn["lam"], L)
    out["mu_nb"][sub] = rn["mu"]
    out["hw_nb"][sub] = confidence_halfwidth(phi[good], knb[good], alpha)
    out["hw"][ok] = confidence_halfwidth(phi, kappa, alpha)
    out["lam_nominal"][ok] = glrt_known_scale(phi, kappa, 1.0)
    re = glrt(phi, np.ones_like(kappa), details=True)
    out["lam_equal"][ok] = re["lam"]
    out["p_equal"][ok] = pvalue(re["lam"], L)
    out["mu_equal"][ok] = re["mu"]
    pz, g12, g23, g13 = zone_multilook(lz)
    out["mu_zone"][ok] = pz
    out["zw"][ok] = zwieback_zone(pz, g12, g23, g13, n_looks * L)
    if extra.get("lut") is not None:
        yr, yhw = yuan_zone(pz, g12, g23, g13, extra["enl_zone"], extra["lut"], alpha)
        out["yuan"][ok] = yr
        out["hw_yuan"][ok] = yhw


def summarize(writer, wname, t, z, classes, pfas):
    for cdef in ("gbar_own", "gbar_indep"):
        g = z[cdef]
        for lo, hi in zip(classes[:-1], classes[1:]):
            sel = np.isfinite(z["lam"]) & (g >= lo) & (g < hi)
            n = int(sel.sum())
            if n == 0:
                continue
            row = {"window": wname, "triplet": t, "class_def": cdef, "class": f"{lo:g}-{hi:g}",
                   "zones": n,
                   "median_abs_mu_deg": float(np.degrees(np.median(np.abs(z["mu"][sel])))),
                   "median_c1": float(np.median(z["c1"][sel]))}
            for pfa in pfas:
                det = sel & (z["p"] < pfa)
                row[f"rate_{pfa:g}"] = float(det.sum() / n)
                row[f"median_abs_mu_det_{pfa:g}_deg"] = (
                    float(np.degrees(np.median(np.abs(z["mu"][det])))) if det.any() else "")
            writer.writerow(row)


def main(args):
    inputs = [args.store] + [x for x in (args.spectrum, args.yuan_lut) if x is not None]
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), inputs,
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    t0 = time.time()
    root = zarr.open_group(args.store, mode="r")
    br, bc = root.attrs["window"]
    lags = root[args.pol]["lags"]
    T = lags.shape[2]
    triplets = args.triplets if args.triplets else list(range(T - 2))
    zone = tuple(args.zone)
    out = zarr.open_group(run.large_dir / "zones.zarr", mode="w")
    out.attrs.update({"store": str(args.store), "pol": args.pol, "zone": list(zone),
                      "triplets": triplets, "classes": args.classes, "run_id": run.run_id})
    fh = open(run.small_dir / "summary.csv", "w", newline="")
    fields = ["window", "triplet", "class_def", "class", "zones", "median_abs_mu_deg",
              "median_c1"]
    for pfa in args.pfa:
        fields += [f"rate_{pfa:g}", f"median_abs_mu_det_{pfa:g}_deg"]
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    lut = None
    if args.yuan_lut is not None:
        d = np.load(args.yuan_lut)
        lut = YuanLUT(d["grid"], d["table"])
    kern = None
    if args.spectrum is not None:
        sp = np.load(args.spectrum)
        kern = np.outer(kernel_from_spectrum(sp["s_az"], args.taps),
                        kernel_from_spectrum(sp["s_rg"], args.taps))
    for a, b in args.factors:
        n_looks, wname = a * b * br * bc, f"{a * br}x{b * bc}"
        g = out.create_group(wname)
        extra = None
        if args.extra:
            L = zone[0] * zone[1]
            enl = None
            if kern is not None:
                Az = correlation_area(kern, (zone[0] * a * br, zone[1] * b * bc))
                enl = L * n_looks / Az
                g.attrs["zone_enl"] = enl
            extra = {"alpha": args.alpha, "lut": lut if enl is not None else None,
                     "enl_zone": enl}
        arrays = None
        # read the store in blocks of dates: the triplet and the independent pairs (t-2 .. t+4)
        for k, t in enumerate(triplets):
            lo, hi = max(t - 2, 0), min(t + 5, T)
            La = aggregate_cells(np.asarray(lags[:, :, lo:hi], dtype=np.complex128), (a, b))
            z = triplet_zones(La, t - lo, zone, n_looks, args.backend, extra)
            if arrays is None:
                shape = (len(triplets),) + z["lam"].shape
                fields = FIELDS if extra is not None else FIELDS[:6]
                arrays = {f: g.create_array(f, shape=shape, dtype="float32",
                                            chunks=(1,) + shape[1:]) for f in fields}
            for f in arrays:
                arrays[f][k] = z[f]
            summarize(writer, wname, t, z, args.classes, args.pfa)
            fh.flush()
            n_ok = int(np.isfinite(z["lam"]).sum())
            logger.info("%s t=%d: %d zones, rate %.4f at %g, %.1f min", wname, t, n_ok,
                        float(np.nanmean(z["p"] < args.pfa[0])) if n_ok else float("nan"),
                        args.pfa[0], (time.time() - t0) / 60)
    fh.close()
    finish_run(run, {"summary": run.small_dir / "summary.csv",
                     "zones": run.large_dir / "zones.zarr"},
               extra={"elapsed_min": (time.time() - t0) / 60})
    logger.info("done: %s", run.small_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True, help="UAVSAR scan store.")
    parser.add_argument("--pol", default="VV")
    parser.add_argument("--factors", type=int, nargs=2, action="append", default=None,
                        help="Cell as a multiple of the base cell (repeatable). Default: 1 1, 2 2.")
    parser.add_argument("--triplets", type=int, nargs="+", default=None, help="Default: all.")
    parser.add_argument("--zone", type=int, nargs=2, default=[5, 5])
    parser.add_argument("--classes", type=float, nargs="+",
                        default=[0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01])
    parser.add_argument("--pfa", type=float, nargs="+", default=[0.01, 0.1, 0.001])
    parser.add_argument("--backend", default="numpy", help="numpy, torch-cpu, torch-cuda, ...")
    parser.add_argument("--extra", action="store_true",
                        help="Also store the neighbor-weighted test, the arcs and the competing "
                             "tests (nominal, equal weights, Zwieback, Yuan).")
    parser.add_argument("--spectrum", type=Path, default=None,
                        help="spectrum.npz (zone ENL of the Yuan test).")
    parser.add_argument("--yuan-lut", type=Path, default=None, help="yuan_lut.npz.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Level of the arcs.")
    parser.add_argument("--taps", type=int, default=8)
    cli.add_run_arguments(parser, task="zone-scan")
    args = parser.parse_args()
    args.factors = args.factors or [[1, 1], [2, 2]]
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
