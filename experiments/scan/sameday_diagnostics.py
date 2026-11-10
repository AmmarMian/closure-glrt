r"""Diagnostics of the detections on the same-day triplets (Section V-A).

For the triplets ``--triplets`` of a ``zone_scan.py`` run (UAVSAR scan store of the run):

* equal weights: the test with all the precisions set to 1 (Watson--Williams), recomputed on the
  same zones; detected share per class, next to that of the scale-free test;
* spatial coherence of the sign: among pairs of detected zones of class at least ``--coh`` at a
  distance of ``d`` zones (``d`` = 1, 2, 3, along azimuth and range), the fraction with the same
  sign of $\hat\mu$ (noise alone gives 50 %);
* precision: median half-width (deg) of the $1-\alpha$ confidence arc of $\hat\mu$
  (:func:`closure_glrt.glrt.confidence_halfwidth`) per class, over all the zones and over the
  detected ones.

Output: ``diagnostics.csv`` (one row per window, triplet, quantity and class or distance).

Example:
    uv run python experiments/scan/sameday_diagnostics.py --scan-run <runs>/runs/<zone-scan run> \
        --backend torch-cuda --task sameday-diagnostics-s1
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import zarr

from closure_glrt import cli
from closure_glrt.backend import get_data_on_device, to_numpy
from closure_glrt.closure import aggregate_cells, tile_zones, triplet_quantities
from closure_glrt.glrt import confidence_halfwidth, glrt, pvalue
from closure_glrt.provenance import finish_run, start_run

logger = logging.getLogger(__name__)


def main(args):
    small = Path(args.scan_run)
    zones = zarr.open_group(small.parent.parent / "large" / "runs" / small.name / "zones.zarr",
                            mode="r")
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), [small / "summary.csv"],
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    store = zarr.open_group(zones.attrs["store"], mode="r")
    br, bc = store.attrs["window"]
    lags = store[zones.attrs["pol"]]["lags"]
    zone = tuple(zones.attrs["zone"])
    triplets = list(zones.attrs["triplets"])
    rows = []
    for a, b in args.factors:
        wname, n_looks = f"{a * br}x{b * bc}", a * b * br * bc
        g = zones[wname]
        for t in args.triplets:
            k = triplets.index(t)
            p, mu, gbar = (np.asarray(g[f][k]) for f in ("p", "mu", args.class_def))
            La = aggregate_cells(np.asarray(lags[:, :, t:t + 3], dtype=np.complex128), (a, b))
            phi, kappa, _ = (x[..., 0] for x in triplet_quantities(La, n_looks))
            zp, zk = tile_zones(phi, zone), tile_zones(kappa, zone)
            ok = np.isfinite(zp).all(-1)
            h = np.full(ok.shape, np.nan)
            h[ok] = np.degrees(to_numpy(confidence_halfwidth(
                get_data_on_device(zp[ok], args.backend), get_data_on_device(zk[ok], args.backend),
                args.alpha, args.backend)))
            p_eq = np.full(ok.shape, np.nan)
            lam = glrt(get_data_on_device(zp[ok], args.backend),
                       get_data_on_device(np.ones_like(zp[ok]), args.backend), args.backend)
            p_eq[ok] = pvalue(to_numpy(lam), zp.shape[-1])
            for lo, hi in zip(args.classes[:-1], args.classes[1:]):
                sel = np.isfinite(p) & (gbar >= lo) & (gbar < hi)
                if sel.sum() == 0:
                    continue
                rows.append({"window": wname, "triplet": t, "quantity": "share",
                             "key": f"{lo:g}-{hi:g}", "zones": int(sel.sum()),
                             "scale_free": float((p[sel] < args.pfa).mean()),
                             "equal_weights": float((p_eq[sel] < args.pfa).mean())})
                det_c = sel & (p < args.pfa)
                rows.append({"window": wname, "triplet": t, "quantity": "arc_halfwidth_deg",
                             "key": f"{lo:g}-{hi:g}", "zones": int(sel.sum()),
                             "median_all": float(np.nanmedian(h[sel])),
                             "median_detected": float(np.nanmedian(h[det_c]))
                             if det_c.any() else float("nan")})
            det = np.isfinite(p) & (p < args.pfa) & (gbar >= args.coh)
            s = np.sign(mu)
            for d in (1, 2, 3):
                pairs = [(det[:-d] & det[d:], s[:-d] == s[d:]),
                         (det[:, :-d] & det[:, d:], s[:, :-d] == s[:, d:])]
                n = sum(int(m.sum()) for m, _ in pairs)
                same = sum(int((m & e).sum()) for m, e in pairs)
                rows.append({"window": wname, "triplet": t, "quantity": "same_sign",
                             "key": f"d={d}", "zones": n,
                             "scale_free": same / n if n else float("nan"),
                             "equal_weights": ""})
            logger.info("%s t=%d done", wname, t)
    with open(run.small_dir / "diagnostics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    finish_run(run, {"diagnostics": run.small_dir / "diagnostics.csv"})
    logger.info("done: %s", run.small_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan-run", required=True, help="Run directory of zone_scan.py.")
    parser.add_argument("--factors", type=int, nargs=2, action="append", default=None)
    parser.add_argument("--triplets", type=int, nargs="+", default=[10, 23])
    parser.add_argument("--class-def", default="gbar_indep", choices=["gbar_indep", "gbar_own"])
    parser.add_argument("--classes", type=float, nargs="+",
                        default=[0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01])
    parser.add_argument("--coh", type=float, default=0.5)
    parser.add_argument("--pfa", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.05, help="Level of the confidence arcs.")
    parser.add_argument("--backend", default="numpy")
    cli.add_run_arguments(parser, task="sameday-diagnostics")
    args = parser.parse_args()
    args.factors = args.factors or [[1, 1], [2, 2]]
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
