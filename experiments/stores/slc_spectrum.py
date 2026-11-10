r"""Store 2: speckle spectrum and autocorrelation of the SLCs (correlated looks, scale $c$).

Blocks of ``--block`` pixels are read from a few dates (memory map, no full read) at random
positions where no pixel is zero, and normalised by their mean power. Per block $x$ (UAVSAR:
rows azimuth, columns slant range; OPERA CSLC with ``--cslc``: rows northing, columns easting,
azimuth carrier and flattening phases removed):

- 2-D power spectrum $S=\operatorname{mean}|\mathcal F_2x|^2$ (``s_2d``);
- azimuth and range power spectra $S_{az}$, $S_{rg}$ (``s_az``, ``s_rg``), whose square roots
  filter white noise into simulated speckle;
- autocorrelation $|\rho(k,l)|=|\sum x_{i,j}x^*_{i+k,j+l}|/\sum|x_{i,j}|^2$, $0\le k,l\le$
  ``--max-lag``, to check separability ($|\rho(1,1)|$ against $|\rho(1,0)||\rho(0,1)|$).

Outputs: ``spectrum.npz`` (``f_az, s_az, f_rg, s_rg, s_2d, rho, blocks``) and ``summary.json``.
Paper run (UAVSAR, VV, segment 1, dates 10, 30, 40, 50, 60, 20 blocks of 512 x 512 per date,
seed 0): neighbour correlation 0.75 in azimuth and 0.70 in range.

Examples:
    uv run python experiments/stores/slc_spectrum.py
    uv run python experiments/stores/slc_spectrum.py --cslc --burst T035-073251-IW3
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from closure_glrt import cli
from closure_glrt.io.uavsar import list_acquisitions, slc_shape
from closure_glrt.provenance import finish_run, sha256_file, start_run

logger = logging.getLogger(__name__)


def autocorrelation(x, max_lag):
    """|rho(k, l)| for 0 <= k, l <= max_lag of a 2-D complex block."""
    H, W = x.shape
    p = (np.abs(x) ** 2).sum()
    rho = np.zeros((max_lag + 1, max_lag + 1))
    for k in range(max_lag + 1):
        for m in range(max_lag + 1):
            rho[k, m] = np.abs((x[:H - k, :W - m] * np.conj(x[k:, m:])).sum()) / p
    return rho


class _Opera:
    """One OPERA CSLC date as (date, flight, path)."""

    def __init__(self, a):
        self.date, self.flight, self.path = a["date"], a.get("platform", ""), Path(a["path"])


def open_block_source(a, args):
    """(shape, read(r0, c0, b)) of one date."""
    if not args.cslc:
        H, W = slc_shape(a.path, args.segment)
        m = np.memmap(a.path, dtype="<c8", mode="r", shape=(H, W))
        return (H, W), lambda r0, c0, b: np.asarray(m[r0:r0 + b, c0:c0 + b], dtype=np.complex128)
    import h5py

    h = h5py.File(a.path, "r")
    d, car, fl = h["data/VV"], h["data/azimuth_carrier_phase"], h["data/flattening_phase"]

    # The CSLCs carry the TOPS azimuth carrier and the flattening phase. Both are (nearly) common
    # to the dates of a burst, so they cancel in interferograms, but not in the autocorrelation of
    # one image (0.002 with them, 0.69 / 0.34 at one pixel without).
    def read(r0, c0, b):
        ph = np.asarray(car[r0:r0 + b, c0:c0 + b]) + np.asarray(fl[r0:r0 + b, c0:c0 + b])
        return np.asarray(d[r0:r0 + b, c0:c0 + b], dtype=np.complex128) * np.exp(-1j * ph)
    return d.shape, read


def main(args):
    rng = np.random.default_rng(args.seed)
    if not args.cslc:
        acq = list_acquisitions(args.slc_dir or cli.uavsar_dir(), args.pol, args.segment)
    else:
        from closure_glrt.io.opera import list_bursts

        acq = [_Opera(a) for a in list_bursts(args.slc_dir or cli.cslc_dir())[args.burst]]
    picks = [acq[i] for i in args.dates if i < len(acq)]
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), [a.path for a in picks],
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    b = args.block
    s_az, s_rg, rho, blocks = np.zeros(b), np.zeros(b), np.zeros((args.max_lag + 1,) * 2), []
    s_2d = np.zeros((b, b))
    for a in picks:
        (H, W), read = open_block_source(a, args)
        n, tries = 0, 0
        while n < args.blocks_per_date and tries < 50 * args.blocks_per_date:
            tries += 1
            r0, c0 = int(rng.integers(0, H - b)), int(rng.integers(0, W - b))
            x = read(r0, c0, b)
            if (x == 0).any() or not np.isfinite(x).all():
                continue
            x /= np.sqrt((np.abs(x) ** 2).mean())
            s_az += (np.abs(np.fft.fft(x, axis=0)) ** 2).mean(axis=1) / b
            s_rg += (np.abs(np.fft.fft(x, axis=1)) ** 2).mean(axis=0) / b
            s_2d += np.abs(np.fft.fft2(x)) ** 2 / b ** 2
            rho += autocorrelation(x, args.max_lag)
            blocks.append((a.date, a.flight, r0, c0))
            n += 1
        logger.info("%s %s: %d blocks", a.date, a.flight, n)
    k = len(blocks)
    s_az, s_rg, rho, s_2d = s_az / k, s_rg / k, rho / k, s_2d / k
    out = run.small_dir / "spectrum.npz"
    np.savez(out, f_az=np.fft.fftfreq(b), s_az=s_az, f_rg=np.fft.fftfreq(b), s_rg=s_rg, s_2d=s_2d,
             rho=rho, blocks=np.array(blocks, dtype=object).astype(str))
    summary = {"blocks": k, "block": b, "pol": args.pol, "segment": args.segment,
               "burst": args.burst, "rho_az": rho[:, 0].round(4).tolist(),
               "rho_rg": rho[0, :].round(4).tolist(), "rho_11": round(float(rho[1, 1]), 4),
               "rho_10_x_rho_01": round(float(rho[1, 0] * rho[0, 1]), 4)}
    (run.small_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    logger.info("autocorrelation: %s", summary)
    finish_run(run, {"spectrum": out},
               input_sha256={str(a.path): sha256_file(a.path) for a in picks})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cslc", action="store_true", help="OPERA CSLC burst (with --burst).")
    parser.add_argument("--slc-dir", type=Path, default=None,
                        help="Default: $DATA_ROOT/uavsar/SDelta_23518 or .../OPERA_CSLC.")
    parser.add_argument("--burst", default=None, help="OPERA burst id, e.g. T035-073251-IW3.")
    parser.add_argument("--pol", default="VV")
    parser.add_argument("--segment", type=int, default=1)
    parser.add_argument("--dates", type=int, nargs="+", default=[10, 30, 40, 50, 60])
    parser.add_argument("--block", type=int, default=512)
    parser.add_argument("--blocks-per-date", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    cli.add_run_arguments(parser, task="slc-spectrum")
    args = parser.parse_args()
    if args.cslc and not args.burst:
        parser.error("--cslc needs --burst")
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
