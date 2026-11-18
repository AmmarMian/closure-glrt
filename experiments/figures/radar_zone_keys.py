r"""Zone keys of a grid of UAVSAR radar positions, to draw zone results in radar geometry.

The cross-sensor zones are squares of ``--zone-m`` metres on the UTM grid (EPSG 32610), keyed as
in ``experiments/sentinel1/cross_sensor.py``: ``floor(E / zone_m) * 1_000_000 + floor(N / zone_m)``.
This script samples the UAVSAR geolocation (``.llh`` of the 2 x 8 product: latitude, longitude,
height per pixel of 8 looks in azimuth and 2 in range) every ``--step`` SLC pixels (azimuth,
range) and stores the key of the zone that contains each sample. A zone result can then be drawn
on the radar grid of the segment (azimuth along the strip, slant range across it), where a
segment is a rectangle.

Output: ``radar_keys.npz`` with ``keys`` (rows: azimuth samples, columns: range samples),
``step`` and ``zone_m``.

Example:
    python experiments/figures/radar_zone_keys.py --store <scan run>/scan.zarr \
        --llh SDelta_23518_02_BC_s1_2x8.llh --out radar_keys_s1.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pyproj
import zarr


def main(args):
    H, W = zarr.open_group(args.store, mode="r").attrs["image_shape"]
    a = np.memmap(args.llh, dtype="<f4", mode="r").reshape(-1, W // 2, 3)
    sa, sr = args.step
    az = np.arange(sa // 2, H, sa) / 8            # SLC azimuth pixel -> llh row (8 looks)
    rg = np.arange(sr // 2, W, sr) / 2            # SLC range pixel -> llh column (2 looks)
    ia = np.clip(np.round(az).astype(int), 0, a.shape[0] - 1)
    ir = np.clip(np.round(rg).astype(int), 0, a.shape[1] - 1)
    sub = np.asarray(a[ia][:, ir])
    lat, lon = sub[..., 0], sub[..., 1]
    x, y = pyproj.Transformer.from_crs(4326, 32610, always_xy=True).transform(lon, lat)
    keys = (np.floor(np.asarray(x) / args.zone_m).astype(np.int64) * 1_000_000
            + np.floor(np.asarray(y) / args.zone_m).astype(np.int64))
    keys[~np.isfinite(lat) | (lat == 0)] = -1
    np.savez_compressed(args.out, keys=keys, step=np.array(args.step), zone_m=args.zone_m)
    print("wrote", args.out, keys.shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", type=Path, required=True, help="UAVSAR scan store (image shape).")
    p.add_argument("--llh", type=Path, required=True)
    p.add_argument("--step", type=int, nargs=2, default=[80, 30],
                   help="SLC pixels between samples (azimuth, range): about 48 m x 50 m.")
    p.add_argument("--zone-m", type=float, default=100.0)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
