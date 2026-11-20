r"""Store 4: 3 x 3 cell covariances of the Sentinel-1 triplets matched to UAVSAR triplets (VV).

The burst scan keeps lags 0 to 2 between consecutive dates only, while a Sentinel-1 triplet
matched to a UAVSAR triplet has dates further apart. For each triplet of ``data/s1_triplets.csv``
of the chosen UAVSAR segment and each burst of its track that intersects the segment's box
(``data/segments.json``), the three dates are read on the crop and, per cell of ``--window``
pixels, the six entries of the sample covariance are stored as
``<track>_<D1>_<D2>_<D3>/<burst>/cov`` (I, J, 6) complex64:
$[P_1,P_2,P_3,L_{12},L_{23},L_{13}]$, $L_{ab}=\frac1N\sum_nx_{an}x_{bn}^*$, $P_a=L_{aa}$.
Covariances are computed in complex128 and stored in complex64, as for the paper. NaN pixels
(outside the burst's valid area) make the cell NaN.

Group attributes: dates, files, crop (row0, col0), cell-centre grid (x0, y0, dx, dy in m),
EPSG, window.

Example:
    uv run --extra io python experiments/stores/cslc_triplets.py --segment 1
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import zarr

from closure_glrt import cli
from closure_glrt.closure import block_sample_covariance
from closure_glrt.io.opera import grid, list_bursts
from closure_glrt.provenance import finish_run, sha256_file, start_run

logger = logging.getLogger(__name__)


def crop(gr, bbox, window):
    """Row/column range of the burst grid inside bbox (xmin, ymin, xmax, ymax), cell-aligned."""
    xmin, ymin, xmax, ymax = bbox
    H, W = gr["shape"]
    x = gr["x0"] + gr["dx"] * np.arange(W)
    y = gr["y0"] + gr["dy"] * np.arange(H)
    cols = np.flatnonzero((x >= xmin) & (x <= xmax))
    rows = np.flatnonzero((y >= ymin) & (y <= ymax))
    if not len(cols) or not len(rows):
        return None
    r, c = window
    r0, c0 = rows[0] // r * r, cols[0] // c * c
    r1, c1 = (rows[-1] + 1) // r * r, (cols[-1] + 1) // c * c
    return (r0, r1, c0, c1) if r1 > r0 and c1 > c0 else None


def load_triplets(path, segment):
    with open(path) as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        return [(r["track"], r["d1"], r["d2"], r["d3"]) for r in rows
                if int(r["segment"]) == segment]


def main(args):
    r, c = args.window
    bbox = args.bbox or json.loads(args.segments.read_text())["segments"][str(args.segment)]["bbox"]
    triplets = args.triplet or load_triplets(args.triplets_csv, args.segment)
    bursts = list_bursts(args.cslc_dir or cli.cslc_dir())
    jobs, inputs = [], []
    for track, *dates in triplets:
        for b, acq in sorted(bursts.items()):
            if not b.startswith(track):
                continue
            by_date = {a["date"]: a for a in acq}
            if not all(d in by_date for d in dates):
                continue
            gr = grid(by_date[dates[0]]["path"])
            box = crop(gr, bbox, (r, c))
            if box is None:
                continue
            files = [by_date[d]["path"] for d in dates]
            jobs.append((track, tuple(dates), b, gr, box, files))
            inputs += files
    inputs = sorted(set(inputs))
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), inputs,
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    logger.info("run %s: %d (triplet, burst) jobs, %d files", run.run_id, len(jobs), len(inputs))

    store_path = run.large_dir / "triplets.zarr"
    root = zarr.open_group(store_path, mode="w")
    root.attrs.update({"window": [r, c], "bbox": list(bbox), "segment": args.segment,
                       "run_id": run.run_id,
                       "cov": "[P1, P2, P3, L12, L23, L13], "
                              "L_ab = mean x_a conj(x_b) over the cell"})
    t_start = time.time()
    for n, (track, dates, b, gr, (r0, r1, c0, c1), files) in enumerate(jobs):
        X = np.empty((r1 - r0, c1 - c0, 3), np.complex64)
        for k, f in enumerate(files):
            with h5py.File(f, "r") as h:
                X[..., k] = h["data/VV"][r0:r1, c0:c1]
        C = block_sample_covariance(X.astype(np.complex128), (r, c))
        cov = np.stack([C[..., 0, 0], C[..., 1, 1], C[..., 2, 2], C[..., 0, 1], C[..., 1, 2],
                        C[..., 0, 2]], axis=-1).astype(np.complex64)
        g = root.require_group(f"{track}_{'_'.join(dates)}").create_group(b)
        g.attrs.update({"dates": list(dates), "files": [f.name for f in files], "epsg": gr["epsg"],
                        "row0": int(r0), "col0": int(c0), "window": [r, c],
                        "x0": gr["x0"] + gr["dx"] * (c0 + (c - 1) / 2),
                        "y0": gr["y0"] + gr["dy"] * (r0 + (r - 1) / 2),
                        "dx": gr["dx"] * c, "dy": gr["dy"] * r})
        g.create_array("cov", data=cov,
                       chunks=(min(512, cov.shape[0]), min(512, cov.shape[1]), 6))
        logger.info("%d/%d %s %s %s: crop %dx%d px, %.1f min", n + 1, len(jobs), track,
                    "-".join(dates), b, r1 - r0, c1 - c0, (time.time() - t_start) / 60)
    hashes = {str(p): sha256_file(p) for p in inputs}
    finish_run(run, {"triplets": store_path}, input_sha256=hashes,
               extra={"elapsed_min": (time.time() - t_start) / 60})
    logger.info("done: %s", run.manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", type=int, default=1)
    parser.add_argument("--cslc-dir", type=Path, default=None,
                        help="Default: $DATA_ROOT/sentinel1/OPERA_CSLC.")
    parser.add_argument("--triplets-csv", type=Path, default=cli.REPO / "data" / "s1_triplets.csv")
    parser.add_argument("--segments", type=Path, default=cli.REPO / "data" / "segments.json")
    parser.add_argument("--bbox", type=float, nargs=4, default=None,
                        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        help="Overrides the segment box (UTM, burst EPSG).")
    parser.add_argument("--triplet", nargs=4, action="append", default=None,
                        metavar=("TRACK", "D1", "D2", "D3"), help="Overrides the CSV (repeatable).")
    parser.add_argument("--window", type=int, nargs=2, default=[2, 4],
                        help="Cell (rows, cols); 2 4 = 20 m x 20 m.")
    cli.add_run_arguments(parser, task="cslc-triplets")
    args = parser.parse_args()
    args.task = f"{args.task}-s{args.segment}"
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
