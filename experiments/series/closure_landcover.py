r"""Closure of the zones by crop type (USDA Cropland Data Layer), from a zone scan (Section V-B).

The centre of every zone of a ``zone_scan.py`` run is geolocated with the 2x8 ``.llh`` file of
the segment and gets the CDL class of the year of the middle date of the triplet (mode of the
3 x 3 CDL pixels around it) and its purity (fraction of these pixels in that class). On the
zones of zone coherence (``--class-def``) at least ``--coh`` and purity at least ``--purity``,
per triplet and CDL class with at least ``--min-zones`` zones: the circular mean of $\hat\mu$
over all the zones, the detected share at ``--pfa``, and the fractions of the zones detected with
$\hat\mu>0$ and with $\hat\mu<0$. Output: ``by_class.csv``.

Example:
    uv run python experiments/series/closure_landcover.py --scan-run <runs>/runs/<zone-scan run> \
        --llh $DATA_ROOT/uavsar/SDelta_23518/SDelta_23518_02_BC_s1_2x8.llh --task landcover-s1
"""

import argparse
import csv
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import zarr
from scipy import stats

from closure_glrt import cli
from closure_glrt.provenance import finish_run, start_run

logger = logging.getLogger(__name__)
CDL = {1: "corn", 2: "cotton", 3: "rice", 4: "sorghum", 5: "soybeans", 6: "sunflower",
       12: "sweet corn", 21: "barley", 23: "spring wheat", 24: "winter wheat", 27: "rye",
       28: "oats", 33: "safflower", 36: "alfalfa", 37: "other hay", 41: "sugarbeets",
       42: "dry beans", 43: "potatoes", 44: "other crops", 47: "misc veg/fruit",
       48: "watermelons", 49: "onions", 53: "peas", 54: "tomatoes", 57: "herbs", 58: "clover",
       59: "sod/grass seed", 61: "fallow/idle", 66: "cherries", 67: "peaches", 68: "apples",
       69: "grapes", 71: "other tree crops", 72: "citrus", 74: "pecans", 75: "almonds",
       76: "walnuts", 77: "pears", 92: "aquaculture", 111: "open water", 121: "developed open",
       122: "developed low", 123: "developed medium", 124: "developed high", 131: "barren",
       141: "deciduous forest", 142: "evergreen forest", 143: "mixed forest", 152: "shrubland",
       176: "grass/pasture", 190: "woody wetlands", 195: "herbaceous wetlands",
       204: "pistachios", 205: "triticale", 211: "olives", 212: "oranges", 217: "pomegranates",
       225: "dbl wheat/corn", 226: "dbl oats/corn", 236: "dbl wheat/sorghum",
       237: "dbl barley/corn"}


def zone_lonlat(llh_path, image_shape, cell, zone, shape):
    """Lon/lat of the zone centres from the 2x8 .llh (8 azimuth rows, 2 range columns)."""
    H, W = image_shape
    rows, cols = -(-H // 8), -(-W // 2)
    llh = np.memmap(llh_path, dtype="<f4", mode="r", shape=(rows, cols, 3))
    zr, zc = zone[0] * cell[0], zone[1] * cell[1]
    y = np.arange(shape[0]) * zr + zr // 2
    x = np.arange(shape[1]) * zc + zc // 2
    sub = np.asarray(llh[np.minimum(y // 8, rows - 1)][:, np.minimum(x // 2, cols - 1)])
    return sub[..., 1], sub[..., 0]


def sample(path, lon, lat, k=1):
    """Raster values of band 1 in the (2k+1)^2 pixels around lon/lat; nan outside."""
    import rasterio
    from pyproj import Transformer

    with rasterio.open(path) as d:
        e, n = Transformer.from_crs(4326, d.crs, always_xy=True).transform(lon, lat)
        r, c = (np.asarray(v) for v in rasterio.transform.rowcol(d.transform, e, n))
        a = d.read(1).astype(np.float64)
        if d.nodata is not None:
            a[a == d.nodata] = np.nan
    off = np.arange(-k, k + 1)
    win = (len(r), 2 * k + 1, 2 * k + 1)
    R = np.broadcast_to(r[:, None, None] + off[None, :, None], win).reshape(len(r), -1)
    C = np.broadcast_to(c[:, None, None] + off[None, None, :], win).reshape(len(c), -1)
    inside = (R >= 0) & (R < a.shape[0]) & (C >= 0) & (C < a.shape[1])
    out = np.full(R.shape, np.nan)
    out[inside] = a[R[inside], C[inside]]
    return out


def circmean_deg(x):
    return float(np.degrees(np.angle(np.mean(np.exp(1j * x)))))


def main(args):
    small = Path(args.scan_run)                      # <root>/runs/<id> -> <root>/large/runs/<id>
    zones = zarr.open_group(args.zones or small.parent.parent / "large" / "runs" / small.name
                            / "zones.zarr", mode="r")
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), [args.llh],
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    anc = args.ancillary or cli.data_root() / "ancillary"
    store = zarr.open_group(zones.attrs["store"], mode="r")
    base = tuple(store.attrs["window"])
    a, b = (int(v) for v in args.factor)
    cell = (a * base[0], b * base[1])
    g = zones[f"{cell[0]}x{cell[1]}"]
    triplets = list(zones.attrs["triplets"])
    dates = [dt.date.fromisoformat(x["date"]) for x in store.attrs["acquisitions"]]
    shape = g["mu"].shape[1:]
    lon, lat = zone_lonlat(args.llh, store.attrs["image_shape"], cell, tuple(zones.attrs["zone"]),
                           shape)
    lon, lat = lon.ravel(), lat.ravel()
    rows, cdl_cache = [], {}
    for t in args.triplets:
        k = triplets.index(t)
        year = dates[t + 1].year
        if year not in cdl_cache:
            px = sample(anc / "cdl" / f"cdl_{year}.tif", lon, lat)
            cls = stats.mode(px, axis=1, nan_policy="omit").mode.astype(float)
            cdl_cache[year] = (cls, (px == cls[:, None]).mean(1))
        cls, purity = cdl_cache[year]
        mu, p, gbar = (np.asarray(g[f][k]).ravel() for f in ("mu", "p", args.class_def))
        keep = np.isfinite(mu) & (gbar >= args.coh) & (purity >= args.purity)
        for c in np.unique(cls[keep]):
            sel = keep & (cls == c)
            n = int(sel.sum())
            if n < args.min_zones:
                continue
            det = sel & (p < args.pfa)
            rows.append({"triplet": t, "dates": "/".join(str(d) for d in dates[t:t + 3]),
                         "cdl": int(c), "class": CDL.get(int(c), f"class {int(c)}"), "zones": n,
                         "mu_circmean_deg": circmean_deg(mu[sel]),
                         "share_detected": float(det.sum() / n),
                         "frac_detected_pos": float((det & (mu > 0)).sum() / n),
                         "frac_detected_neg": float((det & (mu < 0)).sum() / n)})
        logger.info("t=%d: %d zones kept", t, int(keep.sum()))
    with open(run.small_dir / "by_class.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    finish_run(run, {"by_class": run.small_dir / "by_class.csv"}, extra={"ancillary": str(anc)})
    logger.info("done: %s", run.small_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan-run", required=True, help="Run directory of zone_scan.py.")
    parser.add_argument("--zones", type=Path, default=None,
                        help="zones.zarr (default: the large dir of --scan-run).")
    parser.add_argument("--llh", type=Path, required=True, help="2x8 .llh of the segment.")
    parser.add_argument("--ancillary", type=Path, default=None,
                        help="Default: $DATA_ROOT/ancillary.")
    parser.add_argument("--factor", type=int, nargs=2, default=[1, 1])
    parser.add_argument("--triplets", type=int, nargs="+", default=[10, 12, 13, 40])
    parser.add_argument("--class-def", default="gbar_indep", choices=["gbar_indep", "gbar_own"])
    parser.add_argument("--coh", type=float, default=0.5)
    parser.add_argument("--purity", type=float, default=7 / 9)
    parser.add_argument("--min-zones", type=int, default=200)
    parser.add_argument("--pfa", type=float, default=0.01)
    cli.add_run_arguments(parser, task="closure-landcover")
    args = parser.parse_args()
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
