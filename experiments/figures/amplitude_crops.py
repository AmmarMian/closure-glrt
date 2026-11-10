r"""UAVSAR and Sentinel-1 amplitudes of one zone on a common north-up UTM grid (input of Fig. 1).

UAVSAR: the 1x1 SLC of one acquisition is read over the rows and columns whose 2x8
geolocation (``.llh``) falls in the box. The intensity is averaged over the 2x8 blocks of the
``.llh`` grid (16 looks, about 5 m on the ground), and these blocks are binned onto the UTM
grid (mean of the blocks in a pixel; isolated empty pixels take their nearest neighbour).
Sentinel-1: the OPERA CSLC bursts of one date (already on a UTM grid, 5 m in x, 10 m in y) are
averaged over two pixels in x (10 x 10 m, 2 looks), mosaicked and resampled to the grid
(nearest). Output: an ``.npz`` with the two intensities, the grid and the dates.

Example (on the machine that holds the SLCs):
    U=$DATA_ROOT/uavsar/SDelta_23518
    python experiments/figures/amplitude_crops.py --center 617700 4218500 --size 3000 \
        --uavsar-slc $U/SDelta_23518_17060_012_170609_L090VV_02_BC_s1_1x1.slc \
        --llh $U/SDelta_23518_02_BC_s1_2x8.llh \
        --cslc $DATA_ROOT/sentinel1/OPERA_CSLC/T042/*20170606T*_VV_v1.1.h5 --out crops.npz
"""

import argparse
import re

import numpy as np

SLC_SHAPE = (66664, 9900)      # segment 1, 1x1
LLH_LOOKS = (8, 2)             # rows (azimuth), columns (range) of the 2x8 grid


def utm_grid(center, size, res):
    cx, cy = center
    n = int(round(size / res))
    x = cx - size / 2 + res * (np.arange(n) + 0.5)
    y = cy + size / 2 - res * (np.arange(n) + 0.5)          # north-up: decreasing
    return x, y


def bin_to_grid(xs, ys, v, x, y, res):
    """Mean of the values ``v`` at points ``(xs, ys)`` in each pixel of the grid; nan if empty."""
    j = np.floor((xs - (x[0] - res / 2)) / res).astype(int)
    i = np.floor(((y[0] + res / 2) - ys) / res).astype(int)
    ok = (i >= 0) & (i < len(y)) & (j >= 0) & (j < len(x)) & np.isfinite(v)
    k = i[ok] * len(x) + j[ok]
    s = np.bincount(k, v[ok], len(x) * len(y))
    c = np.bincount(k, None, len(x) * len(y))
    with np.errstate(invalid="ignore"):
        out = (s / c).reshape(len(y), len(x))
    return fill_isolated(out)


def fill_isolated(a, max_dist=1.5):
    """Empty pixels at most ``max_dist`` pixels from a filled one take its value (nearest)."""
    from scipy import ndimage

    empty = np.isnan(a)
    dist, (ii, jj) = ndimage.distance_transform_edt(empty, return_indices=True)
    out = a.copy()
    sel = empty & (dist <= max_dist)
    out[sel] = a[ii[sel], jj[sel]]
    return out


def uavsar_crop(slc_path, llh_path, x, y, res, slc_shape=SLC_SHAPE):
    import pyproj

    ar, ac = LLH_LOOKS
    rows, cols = slc_shape[0] // ar, slc_shape[1] // ac
    llh = np.memmap(llh_path, np.float32, "r", shape=(rows, cols, 3))
    to_utm = pyproj.Transformer.from_crs(4326, 32610, always_xy=True)
    half = (x[-1] - x[0]) / 2 + 50
    cx, cy = (x[0] + x[-1]) / 2, (y[0] + y[-1]) / 2
    sub = 16                                                  # coarse search for the box
    e, n = to_utm.transform(llh[::sub, ::sub, 1], llh[::sub, ::sub, 0])
    ii, jj = np.nonzero((np.abs(e - cx) < half + 200) & (np.abs(n - cy) < half + 200))
    if ii.size == 0:
        raise ValueError("box outside the UAVSAR footprint")
    i0, i1 = max(ii.min() * sub - sub, 0), min(ii.max() * sub + sub, rows)
    j0, j1 = max(jj.min() * sub - sub, 0), min(jj.max() * sub + sub, cols)
    e, n = to_utm.transform(llh[i0:i1, j0:j1, 1], llh[i0:i1, j0:j1, 0])
    slc = np.memmap(slc_path, np.complex64, "r", shape=slc_shape)
    block = np.abs(np.asarray(slc[i0 * ar:i1 * ar, j0 * ac:j1 * ac])) ** 2
    inten = block.reshape(i1 - i0, ar, j1 - j0, ac).mean((1, 3))
    return bin_to_grid(np.ravel(e), np.ravel(n), inten.ravel(), x, y, res)


def cslc_crop(paths, x, y):
    import h5py

    out = np.full((len(y), len(x)), np.nan)
    for p in paths:
        with h5py.File(p, "r") as h:
            xc, yc = h["data/x_coordinates"][:], h["data/y_coordinates"][:]
            dx, dy = xc[1] - xc[0], yc[1] - yc[0]
            j = np.round((x - xc[0]) / dx).astype(int)
            i = np.round((y - yc[0]) / dy).astype(int)
            jo, io = (j >= 0) & (j < len(xc)), (i >= 0) & (i < len(yc))
            if not jo.any() or not io.any():
                continue
            r0, r1, c0, c1 = i[io].min(), i[io].max() + 1, j[jo].min(), j[jo].max() + 1
            v = np.abs(h["data/VV"][r0:r1, c0:c1]).astype(np.float64) ** 2
        v[v == 0] = np.nan
        if abs(dx) < abs(dy):                                 # 5 m x 10 m: 2 looks to 10 x 10 m
            k = int(round(abs(dy) / abs(dx)))
            v = np.nanmean(np.stack([np.roll(v, -s, axis=1) for s in range(k)]), 0)
        tile = np.full((len(y), len(x)), np.nan)
        tile[np.ix_(io, jo)] = v[np.ix_(i[io] - r0, j[jo] - c0)]
        fill = np.isnan(out) & np.isfinite(tile)
        out[fill] = tile[fill]
    return out


def main(args):
    x, y = utm_grid(args.center, args.size, args.res)
    u = uavsar_crop(args.uavsar_slc, args.llh, x, y, args.res)
    s = cslc_crop(args.cslc, x, y)
    d_u = re.search(r"_(\d{6})_L090", args.uavsar_slc).group(1)
    d_s = sorted({re.search(r"_(\d{8})T\d{6}Z_", p).group(1) for p in args.cslc})
    np.savez_compressed(args.out, uavsar=u.astype(np.float32), s1=s.astype(np.float32), x=x, y=y,
                        uavsar_date="20" + d_u, s1_date=",".join(d_s), res=args.res)
    print("uavsar", "20" + d_u, f"filled {np.isfinite(u).mean():.3f}",
          "s1", d_s, f"filled {np.isfinite(s).mean():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--center", nargs=2, type=float, required=True,
                   help="UTM 10N easting northing (m)")
    p.add_argument("--size", type=float, default=3000.0, help="side of the square box (m)")
    p.add_argument("--res", type=float, default=5.0)
    p.add_argument("--uavsar-slc", required=True)
    p.add_argument("--llh", required=True)
    p.add_argument("--cslc", nargs="+", required=True)
    p.add_argument("--out", default="crops.npz")
    main(p.parse_args())
