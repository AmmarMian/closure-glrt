r"""Store 3: scan of the OPERA CSLC-S1 bursts into lagged cell covariances (VV).

Same products and conventions as ``uavsar_scan.py``, one group per burst:
``<burst>/VV/lags`` (H // r, W // c, T, K + 1) complex64, ``amp_mean``, ``amp_disp``.
Every date of a burst is on the same UTM grid (checked: identical shape and origin, else the
burst is refused); rows are northing, columns easting. Pixels outside the burst's valid area
are NaN, so cells touching them are NaN. The azimuth carrier is common to the dates of a burst
and cancels in $x_tx_{t+k}^*$. Used for the long-term Sentinel-1 coherence of the cross-sensor
comparison. Paper run: every burst with at least three dates in ``data/urls/opera_cslc*.txt``,
cells of 2 x 4 pixels (20 m x 20 m), $K=2$.

Example:
    uv run --extra io python experiments/stores/cslc_scan.py
"""

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import zarr

from closure_glrt import cli
from closure_glrt.io.opera import grid, list_bursts, read_rows
from closure_glrt.io.uavsar import strip_statistics
from closure_glrt.provenance import finish_run, sha256_file, start_run

logger = logging.getLogger(__name__)


def main(args):
    r, c = args.window
    if args.strip_rows % r:
        raise ValueError("--strip-rows must be a multiple of the window rows")
    bursts = list_bursts(args.cslc_dir or cli.cslc_dir())
    if args.bursts:
        bursts = {b: bursts[b] for b in args.bursts}
    bursts = {b: a for b, a in bursts.items() if len(a) >= 3}
    grids = {}
    for b, acq in bursts.items():
        g = [grid(a["path"]) for a in acq]
        if any(x != g[0] for x in g):
            raise ValueError(f"{b}: grids differ between dates")
        grids[b] = g[0]
    K = args.max_lag
    inputs = [a["path"] for acq in bursts.values() for a in acq]
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), inputs,
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    logger.info("run %s: %d bursts, %d files, window %s, K=%d", run.run_id, len(bursts),
                len(inputs), args.window, K)

    store_path = run.large_dir / "scan.zarr"
    root = zarr.open_group(store_path, mode="w")
    root.attrs.update({"kind": "opera-cslc", "bursts": sorted(bursts), "window": [r, c],
                       "max_lag": K, "run_id": run.run_id})
    total = sum(len(a) * grids[b]["shape"][0] * grids[b]["shape"][1] * 8 for b, a in bursts.items())
    done_bytes, t_start = 0, time.time()
    with (ProcessPoolExecutor(args.io_workers, mp_context=get_context("forkserver")) as procs,
          ThreadPoolExecutor(4) as hasher):
        futures = {str(p): hasher.submit(sha256_file, p) for p in inputs}
        for b in sorted(bursts):
            acq, gr = bursts[b], grids[b]
            (H, W), T = gr["shape"], len(acq)
            gb = root.create_group(b)
            gb.attrs.update({**gr, "acquisitions": [
                {"index": t, "date": a["date"], "time": a["time"], "platform": a["platform"],
                 "file": a["path"].name} for t, a in enumerate(acq)],
                "window": [r, c], "max_lag": K, "image_shape": [H, W], "pols": ["VV"],
                "segment": None, "run_id": run.run_id,
                "lags": "lags[i, j, t, k] = mean over cell (i, j) of x_t conj(x_{t+k}); "
                        "nan if t + k >= T or a pixel of the cell is invalid"})
            g = gb.create_group("VV")
            cell_chunk = (args.strip_rows // r, max(1, (W // c) // 8))
            lags = g.create_array("lags", shape=(H // r, W // c, T, K + 1), dtype="complex64",
                                  chunks=(*cell_chunk, T, K + 1), fill_value=np.nan + 0j)
            amp_mean = g.create_array("amp_mean", shape=(H, W), dtype="float32",
                                      chunks=(args.strip_rows, max(1, W // 8)), fill_value=np.nan)
            amp_disp = g.create_array("amp_disp", shape=(H, W), dtype="float32",
                                      chunks=(args.strip_rows, max(1, W // 8)), fill_value=np.nan)
            for row0 in range(0, H, args.strip_rows):
                row1 = min(H, row0 + args.strip_rows)
                X = np.stack(list(procs.map(read_rows, [str(a["path"]) for a in acq],
                                            [row0] * T, [row1] * T)))
                L, m, d = strip_statistics(X, (r, c), K, args.threads)
                if L.shape[0]:
                    lags[row0 // r: row0 // r + L.shape[0]] = L
                amp_mean[row0:row1] = m
                amp_disp[row0:row1] = d
                done_bytes += X.nbytes
                elapsed = time.time() - t_start
                logger.info("%s rows %d-%d (%.1f %%), %.0f MB/s decoded, ETA %.0f min", b, row0,
                            row1, 100 * done_bytes / total, done_bytes / 1e6 / elapsed,
                            elapsed * (total - done_bytes) / done_bytes / 60)
        hashes = {p: f.result() for p, f in futures.items()}
    finish_run(run, {"scan": store_path}, input_sha256=hashes,
               extra={"elapsed_min": (time.time() - t_start) / 60})
    logger.info("done: %s", run.manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cslc-dir", type=Path, default=None,
                        help="Default: $DATA_ROOT/sentinel1/OPERA_CSLC.")
    parser.add_argument("--bursts", nargs="+", default=None,
                        help="Default: every burst with >= 3 dates.")
    parser.add_argument("--window", type=int, nargs=2, default=[2, 4],
                        help="Base cell (rows, cols) = (northing, easting); 20 m x 20 m.")
    parser.add_argument("--max-lag", type=int, default=2)
    parser.add_argument("--strip-rows", type=int, default=256)
    parser.add_argument("--threads", type=int, default=12, help="Compute threads.")
    parser.add_argument("--io-workers", type=int, default=12,
                        help="Processes decoding HDF5 strips.")
    cli.add_run_arguments(parser, task="cslc-scan")
    args = parser.parse_args()
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
