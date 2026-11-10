r"""Raw UAVSAR and Sentinel-1 closures in one box, for one matched triplet (input of Fig. 9).

Same data, cells and zone test as ``cross_sensor.py``, without any selection: every cell of the
box with a finite closure and every 100 m zone with at least ``--min-cells`` valid cells.

Outputs ``box.npz``:
  ``{sensor}_cell_x``, ``_cell_y``, ``_cell_phi``, ``_cell_kappa``: cell centres (UTM 10N, m),
      closure phase (rad) and De Zan precision of every cell in the box;
  ``{sensor}_zone_keys``, ``_zone_{p,mu,c1,h,se}``, ``_zone_n``: zone test results (p-value,
      closure, scale, arc half-width at ``--alpha``, standard error) and valid cells per zone;
  ``box`` (x0, y0, x1, y1), ``uavsar_dates``, ``s1_dates``.

Example:
    PYTHONPATH=src python experiments/sentinel1/zoom_closures.py --uavsar-store <scan.zarr> \
        --llh SDelta_23518_02_BC_s1_2x8.llh --triplets-store <triplets.zarr> \
        --s1-triplet T042_20170302_20170407_20170606 --t 62 --center 617700 4218500 \
        --task zoom-closures-s1
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent))
from cross_sensor import Uavsar, closure_from_cov, group, zone_keys, zone_test  # noqa: E402

from closure_glrt import cli  # noqa: E402
from closure_glrt.provenance import finish_run, start_run  # noqa: E402

logger = logging.getLogger(__name__)
np.seterr(all="ignore")


def in_box(x, y, box):
    return (x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3])


def zone_results(keys, phi, kappa, args):
    uniq, _, (p, k) = group(keys, phi, kappa)
    res, n = zone_test(p, k, args.min_cells, args.alpha, args.backend)
    ok = np.isfinite(res["p"])
    return {"zone_keys": uniq[ok], "zone_n": n[ok], **{f"zone_{f}": v[ok] for f, v in res.items()}}


def s1_cells(group_tr, n_looks, box):
    """Cells of the box from the burst that covers most of it."""
    best = None
    for b in sorted(group_tr.group_keys()):
        g = group_tr[b]
        cov = np.asarray(g["cov"][:], dtype=np.complex128)
        phi, kappa = closure_from_cov(cov[..., 0].real, cov[..., 1].real, cov[..., 2].real,
                                      cov[..., 3], cov[..., 4], cov[..., 5], n_looks)
        ni, nj = cov.shape[:2]
        X, Y = np.meshgrid(g.attrs["x0"] + g.attrs["dx"] * np.arange(nj),
                           g.attrs["y0"] + g.attrs["dy"] * np.arange(ni))
        sel = in_box(X, Y, box) & np.isfinite(phi)
        logger.info("S1 burst %s: %d valid cells in the box", b, sel.sum())
        if sel.sum() and (best is None or sel.sum() > best[0].size):
            best = (X[sel], Y[sel], phi[sel], kappa[sel])
    return best


def main(args):
    half = args.size / 2
    box = (args.center[0] - half, args.center[1] - half, args.center[0] + half,
           args.center[1] + half)
    run = start_run(args.task, cli.params(args), *cli.run_roots(args),
                    [args.uavsar_store, args.llh, args.triplets_store],
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    t0 = time.time()
    out = {"box": np.array(box)}

    U = Uavsar(args.uavsar_store, args.llh, args.uavsar_factor, args.zone_m)
    L = U.lags(args.t, args.t + 3)
    phi, kappa = (a.ravel() for a in U.triplet(args.t, L, args.t))
    sel = in_box(U.x, U.y, box) & np.isfinite(phi)
    out.update(uavsar_cell_x=U.x[sel], uavsar_cell_y=U.y[sel], uavsar_cell_phi=phi[sel],
               uavsar_cell_kappa=kappa[sel], uavsar_dates=np.array(
                   [d.strftime("%Y-%m-%d") for d in U.dates[args.t:args.t + 3]]))
    out.update({f"uavsar_{k}": v for k, v in zone_results(
        zone_keys(U.x[sel], U.y[sel], args.zone_m), phi[sel], kappa[sel], args).items()})
    logger.info("UAVSAR t=%d: %d cells, %d zones in the box", args.t, sel.sum(),
                len(out["uavsar_zone_keys"]))

    trip = zarr.open_group(args.triplets_store, mode="r")
    r, c = trip.attrs["window"]
    g = trip[args.s1_triplet]
    x, y, phi, kappa = s1_cells(g, r * c, box)
    out.update(s1_cell_x=x, s1_cell_y=y, s1_cell_phi=phi, s1_cell_kappa=kappa,
               s1_dates=np.array(g[next(iter(g.group_keys()))].attrs["dates"]))
    out.update({f"s1_{k}": v for k, v in zone_results(
        zone_keys(x, y, args.zone_m), phi, kappa, args).items()})
    logger.info("S1 %s: %d cells, %d zones in the box", args.s1_triplet, len(x),
                len(out["s1_zone_keys"]))

    path = run.small_dir / "box.npz"
    np.savez_compressed(path, **out)
    finish_run(run, {"box": path}, extra={"elapsed_min": (time.time() - t0) / 60})
    logger.info("done: %s", run.small_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uavsar-store", type=Path, required=True)
    parser.add_argument("--llh", type=Path, required=True)
    parser.add_argument("--triplets-store", type=Path, required=True)
    parser.add_argument("--s1-triplet", required=True, help="Group of the triplet store.")
    parser.add_argument("--t", type=int, required=True, help="Matched UAVSAR triplet index.")
    parser.add_argument("--center", nargs=2, type=float, required=True,
                        help="UTM 10N easting northing of the box centre (m).")
    parser.add_argument("--size", type=float, default=3000.0)
    parser.add_argument("--uavsar-factor", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--zone-m", type=float, default=100.0)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05, help="Level of the confidence arcs.")
    parser.add_argument("--backend", default="numpy")
    cli.add_run_arguments(parser, task="zoom-closures")
    args = parser.parse_args()
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
