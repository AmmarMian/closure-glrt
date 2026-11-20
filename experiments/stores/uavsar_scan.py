r"""Store 1: scan of the UAVSAR SLC stack into lagged cell covariances.

For one segment of SDelta_23518 and each polarisation (VV for the paper), every acquisition is
read strip by strip (one sequential pass per file, SHA-256 computed on the way) and written to
``scan.zarr``:

- ``<pol>/lags``, shape (H // r, W // c, T, K + 1), complex64:
  ``lags[i, j, t, k]`` $=\frac1N\sum_n x_{t,n}x_{t+k,n}^*$ over the $N=rc$ pixels of cell
  $(i,j)$, ``nan`` when $t+k\ge T$ (products in complex64, sums in complex128). Cells of any
  window that is a multiple of $(r,c)$ are averages of these, so every triplet of dates at most
  $K$ apart can be closed later without rereading the SLCs;
- ``<pol>/amp_mean``, ``<pol>/amp_disp``, shape (H, W), float32: mean of $|x_t|$ over time
  and amplitude dispersion $\mathrm{std}_t|x_t|/\mathrm{mean}_t|x_t|$.

Attributes: acquisitions (flight, date) in file order, window, K, image shape, segment.

Paper runs: base cell $15\times5$ pixels, $K=2$, strips of 600 rows, 12 threads; segment 1
(66 664 x 9 900 pixels, 68 acquisitions) took 92 min for three polarisations on our NAS and
gave a store whose SHA-256 (sorted file hashes, :func:`closure_glrt.provenance.sha256_path`) is
recorded in ``data/stores.json``.

Example:
    uv run python experiments/stores/uavsar_scan.py --segment 1
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import zarr

from closure_glrt import cli
from closure_glrt.io.uavsar import StripReader, list_acquisitions, slc_shape, strip_statistics
from closure_glrt.provenance import finish_run, sha256_file, start_run

logger = logging.getLogger(__name__)


def main(args):
    r, c = args.window
    if args.strip_rows % r:
        raise ValueError("--strip-rows must be a multiple of the window rows")
    stack_dir = args.stack_dir or cli.uavsar_dir()
    stacks = {pol: list_acquisitions(stack_dir, pol, args.segment) for pol in args.pols}
    acq = stacks[args.pols[0]]
    if not acq:
        raise FileNotFoundError(f"no {args.pols[0]} SLC of segment {args.segment} in {stack_dir}")
    for pol, a in stacks.items():
        if [(x.flight, x.date) for x in a] != [(x.flight, x.date) for x in acq]:
            raise ValueError(f"acquisitions of {pol} differ from {args.pols[0]}")
    shapes = {slc_shape(x.path, args.segment) for a in stacks.values() for x in a}
    if len(shapes) != 1:
        raise ValueError(f"SLC shapes differ: {shapes}")
    (H, W), T, K = shapes.pop(), len(acq), args.max_lag
    strip_gb = T * args.strip_rows * W * 8 / 1e9
    if 6 * strip_gb > args.mem_gb:
        raise ValueError(f"estimated memory {6 * strip_gb:.1f} GB > --mem-gb {args.mem_gb}")

    inputs = [x.path for a in stacks.values() for x in a]
    inputs += sorted({Path(str(p).replace(f"_s{args.segment}_1x1.slc", ".ann")) for p in inputs})
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), inputs,
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    logger.info("run %s: %d acquisitions, %s, image %dx%d, window %s, K=%d, strip %.2f GB",
                run.run_id, T, args.pols, H, W, args.window, K, strip_gb)

    store_path = run.large_dir / "scan.zarr"
    root = zarr.open_group(store_path, mode="w")
    root.attrs.update({
        "acquisitions": [{"index": t, "flight": x.flight, "date": x.date}
                         for t, x in enumerate(acq)],
        "window": [r, c], "max_lag": K, "image_shape": [H, W], "segment": args.segment,
        "pols": args.pols, "run_id": run.run_id,
        "lags": "lags[i, j, t, k] = mean over cell (i, j) of x_t conj(x_{t+k}); nan if t + k >= T",
    })
    cell_chunk = (args.strip_rows // r, max(1, (W // c) // 9))
    hashes = {}
    slc_bytes = len(args.pols) * T * H * W * 8
    t_start = time.time()
    for pol in args.pols:
        g = root.create_group(pol)
        lags = g.create_array("lags", shape=(H // r, W // c, T, K + 1), dtype="complex64",
                              chunks=(*cell_chunk, T, K + 1), fill_value=np.nan + 0j)
        amp_mean = g.create_array("amp_mean", shape=(H, W), dtype="float32",
                                  chunks=(args.strip_rows, max(1, W // 9)), fill_value=np.nan)
        amp_disp = g.create_array("amp_disp", shape=(H, W), dtype="float32",
                                  chunks=(args.strip_rows, max(1, W // 9)), fill_value=np.nan)
        reader = StripReader([x.path for x in stacks[pol]], (H, W), args.strip_rows,
                             args.io_threads)
        for row0, X in reader:
            L, m, d = strip_statistics(X, (r, c), K, args.threads)
            if L.shape[0]:
                lags[row0 // r: row0 // r + L.shape[0]] = L
            amp_mean[row0: row0 + X.shape[1]] = m
            amp_disp[row0: row0 + X.shape[1]] = d
            done = (args.pols.index(pol) * H + row0 + X.shape[1]) / (len(args.pols) * H)
            elapsed = time.time() - t_start
            logger.info("%s rows %d-%d (%.1f %%), %.0f MB/s read, ETA %.0f min", pol, row0,
                        row0 + X.shape[1], 100 * done, done * slc_bytes / 1e6 / elapsed,
                        elapsed * (1 - done) / done / 60)
        hashes.update(reader.hashes)
    hashes.update({str(p): sha256_file(p) for p in inputs if p.suffix == ".ann"})
    finish_run(run, {"scan": store_path}, input_sha256=hashes,
               extra={"elapsed_min": (time.time() - t_start) / 60})
    logger.info("done: %s", run.manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack-dir", type=Path, default=None,
                        help="Default: $DATA_ROOT/uavsar/SDelta_23518.")
    parser.add_argument("--segment", type=int, default=1)
    parser.add_argument("--pols", nargs="+", default=["VV"],
                        help="VV for the paper; the NAS scans also hold HH and HV.")
    parser.add_argument("--window", type=int, nargs=2, default=[15, 5],
                        help="Base cell (rows = azimuth, cols = range).")
    parser.add_argument("--max-lag", type=int, default=2)
    parser.add_argument("--strip-rows", type=int, default=600)
    parser.add_argument("--threads", type=int, default=12, help="Compute threads.")
    parser.add_argument("--io-threads", type=int, default=12)
    parser.add_argument("--mem-gb", type=float, default=24)
    cli.add_run_arguments(parser, task="uavsar-scan")
    args = parser.parse_args()
    args.task = f"{args.task}-s{args.segment}"
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
