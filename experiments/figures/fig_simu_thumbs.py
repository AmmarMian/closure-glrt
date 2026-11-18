r"""Thumbnails of the simulation diagram of the TGRS paper (Fig. of Section VI).

One zone of $5\times5$ cells of $15\times5$ pixels: the amplitude of a white complex Gaussian field
and of the same field filtered by the kernels of the measured UAVSAR speckle spectrum
(:func:`closure_glrt.simulation.kernel_from_spectrum`), as in :func:`simulate_zones`. Pixels are
repeated three times in range so that the displayed pixels are about square on the ground
(0.6 m in azimuth, 1.67 m in range). Writes ``simu_white.png`` and ``simu_filtered.png``.

Example:
    uv run --group figures python experiments/figures/fig_simu_thumbs.py \
        --spectrum <run>/spectrum.npz --out-dir ../paper_grsl/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from closure_glrt.simulation import _separable_speckle, kernel_from_spectrum


def save(img, path, zoom=4):
    a = np.abs(img)
    a = np.repeat(np.repeat(a / np.quantile(a, 0.98), 3, axis=1), zoom, 0)
    a = np.repeat(a, zoom, 1)
    plt.imsave(path, np.clip(a, 0, 1), cmap="gray", vmin=0, vmax=1)


def main(args):
    rng = np.random.default_rng(args.seed)
    shape = (args.zone[0] * args.window[0], args.zone[1] * args.window[1])
    sp = np.load(args.spectrum)
    h_az = kernel_from_spectrum(sp["s_az"], args.taps)
    h_rg = kernel_from_spectrum(sp["s_rg"], args.taps)
    white = _separable_speckle(rng, 1, shape, np.ones(1), np.ones(1))[0, 0]
    rng = np.random.default_rng(args.seed)
    filtered = _separable_speckle(rng, 1, shape, h_az, h_rg)[0, 0]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save(white, args.out_dir / "simu_white.png")
    save(filtered, args.out_dir / "simu_filtered.png")
    print("wrote", args.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spectrum", type=Path, required=True)
    p.add_argument("--window", type=int, nargs=2, default=[15, 5])
    p.add_argument("--zone", type=int, nargs=2, default=[5, 5])
    p.add_argument("--taps", type=int, default=8)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out-dir", type=Path, required=True)
    main(p.parse_args())
