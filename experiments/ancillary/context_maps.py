r"""Context of a UAVSAR segment for Figs. 1 and 2: map grid, acquisition geometry and daily weather.

Writes ``context.zarr`` in the large directory of the run, read by
``experiments/figures/fig_study_area.py`` and ``fig_acquisitions.py``. Zones of ``--zone`` cells
of the scan (``--store``), indexed $k = iJ + j$ on the zone grid $(I, J)$:

- ``geo/lon``, ``geo/lat`` (I J) float32: zone centres; ``geo/index`` (H, W) int32: a north-up
  UTM grid of ``--res`` metres where each pixel holds the zone whose centre is the nearest within
  ``--max-dist`` metres, -1 elsewhere;
- ``weather/*`` (days) float32: scene means of ERA5 daily rain (mm), soil moisture 0-7 and 7-28 cm
  (m3/m3, daily mean), daily maximum temperature (deg C), reference evapotranspiration (mm), and
  the mean GHCN station rain (mm).

Attributes: ``geometry`` (UAVSAR flight track, look side, footprint, pixel spacing and incidence
range from the annotation and look-vector files; Sentinel-1 OPERA CSLC burst footprints from
``--footprints``; all in map pixels), ``acquisitions`` (date, UTC and local start time, weather at
acquisition), ``weather_days`` (first and last day), ``map`` (EPSG, top-left pixel centre,
spacing, shape), zone and cell sizes.

Inputs: the scan store, the ``.llh``, ``.lkv`` and one ``.ann`` of the segment (``$DATA_ROOT``),
the ancillary data (``just download-ancillary``), and ``data/opera_cslc_footprints.json`` (ASF
search metadata of the CSLC bursts: file name, footprint, orbit direction; the first entry of
each burst is used).

This is the part of our ``context-maps`` runs that the figures use; those runs also stored zone
closures, backscatter, land cover and Landsat imagery for an interactive explorer.

Example:
    uv run --extra geo python experiments/ancillary/context_maps.py \
        --store <runs>/large/runs/<uavsar-scan-s1 run>/scan.zarr --segment 1
"""

import argparse
import csv
import datetime as dt
import importlib.util
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial import cKDTree

from closure_glrt import cli
from closure_glrt.io.uavsar import parse_ann
from closure_glrt.provenance import finish_run, start_run

HERE = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lc = _module("closure_landcover", HERE.parent / "series" / "closure_landcover.py")
wx = _module("closure_weather", HERE.parent / "series" / "closure_weather.py")


def map_grid(lon, lat, res, max_dist):
    """North-up UTM grid: EPSG, x0, y0 of the top-left pixel centre, index (H, W) of the zones."""
    from pyproj import Transformer

    ok = np.isfinite(lon) & np.isfinite(lat)
    epsg = 32600 + int((np.nanmedian(lon) + 180) // 6) + 1
    e, n = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon[ok], lat[ok])
    x0, x1 = np.floor(e.min() / res) * res, np.ceil(e.max() / res) * res
    y0, y1 = np.ceil(n.max() / res) * res, np.floor(n.min() / res) * res
    W, H = int((x1 - x0) / res) + 1, int((y0 - y1) / res) + 1
    gx, gy = np.meshgrid(x0 + res * np.arange(W), y0 - res * np.arange(H))
    d, k = cKDTree(np.column_stack([e, n])).query(np.column_stack([gx.ravel(), gy.ravel()]),
                                                  distance_upper_bound=max_dist, workers=8)
    index = np.where(np.isfinite(d), np.flatnonzero(ok)[np.minimum(k, ok.sum() - 1)], -1)
    return epsg, float(x0), float(y0), index.astype(np.int32).reshape(H, W)


def acquisition_geometry(args, segment, epsg, x0, y0, res, shape):
    """UAVSAR flight track, look side, footprint and pixel geometry, Sentinel-1 bursts (map px)."""
    from pyproj import Transformer

    tr = Transformer.from_crs(4326, epsg, always_xy=True)

    def px(lon, lat):
        e, n = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.column_stack([(e - x0) / res, (y0 - n) / res]).round(2).tolist()

    ann = parse_ann(args.ann)
    corners = [[float(v) for v in ann[f"Segment {segment} Data Approximate Corner {c}"].split(",")]
               for c in range(1, 5)]
    lat_c, lon_c = zip(*corners)
    foot = px([lon_c[i] for i in (0, 1, 3, 2, 0)], [lat_c[i] for i in (0, 1, 3, 2, 0)])
    heading = float(ann["Peg Heading"])
    pe, pn = tr.transform(float(ann["Peg Longitude"]), float(ann["Peg Latitude"]))
    u = np.array([np.sin(np.radians(heading)), np.cos(np.radians(heading))])
    ce, cn = tr.transform(np.array(lon_c), np.array(lat_c))
    sproj = (np.column_stack([ce, cn]) - [pe, pn]) @ u
    track = [[(pe + s * u[0] - x0) / res, (y0 - (pn + s * u[1])) / res]
             for s in (sproj.min(), sproj.max())]
    lk = np.memmap(args.lkv, dtype="<f4", mode="r").reshape(-1, int(ann["lkv_1_2x8 Columns"]), 3)
    row = np.asarray(lk[lk.shape[0] // 2])
    inc = np.degrees(np.arccos(np.abs(row[:, 2]) / np.linalg.norm(row, axis=1)))[[0, -1]]
    inc = inc.round(1).tolist()
    sp_rg = float(ann["1x1 SLC Range Pixel Spacing"])
    sp_az = float(ann["1x1 SLC Azimuth Pixel Spacing"])
    uav = {"heading_deg": heading, "look": ann["Look Direction"],
           "altitude_m": float(ann["Average Altitude"]),
           "track": [[round(a, 2), round(b, 2)] for a, b in track], "footprint": foot,
           "spacing_m": {"azimuth": sp_az, "slant_range": sp_rg}, "incidence_deg": inc,
           "ground_range_spacing_m": [round(sp_rg / np.sin(np.radians(i)), 2) for i in inc],
           "line": "SDelta_23518 (UAVSAR L-band, quad-pol)"}
    bursts = {}
    H, W = shape
    for m in json.loads(Path(args.footprints).read_text()):
        b = re.search(r"T\d{3}-\d{6}-IW\d", m.get("fileName", ""))
        if not b or b.group(0) in bursts or not m.get("wkt", "").startswith("POLYGON"):
            continue
        w = m["wkt"]
        xy = [tuple(map(float, p.split())) for p in w[w.index("((") + 2:w.index("))")].split(",")]
        lo, la = zip(*xy)
        poly = px(lo, la)
        P = np.array(poly)
        if P[:, 0].max() < 0 or P[:, 0].min() > W or P[:, 1].max() < 0 or P[:, 1].min() > H:
            continue
        bursts[b.group(0)] = {"track": b.group(0)[:4], "direction": m.get("flightDirection"),
                              "polygon": poly}
    return {"uavsar": uav, "s1_bursts": bursts,
            "s1_note": "OPERA CSLC burst footprints from the ASF search metadata"}


def daily_weather(anc):
    """Scene means per day of the ERA5 series and of the GHCN station rain."""
    wtimes, era = wx.era5_scene(anc)
    rain_g = wx.ghcn_rain(anc)
    ud, inv = np.unique(wtimes.astype("datetime64[D]"), return_inverse=True)
    cnt = np.bincount(inv)
    daily = {"rain": np.bincount(inv, np.nan_to_num(era["precipitation"])),
             "sm0_7": np.bincount(inv, np.nan_to_num(era["soil_moisture_0_to_7cm"])) / cnt,
             "sm7_28": np.bincount(inv, np.nan_to_num(era["soil_moisture_7_to_28cm"])) / cnt,
             "et0": np.bincount(inv, np.nan_to_num(era["et0_fao_evapotranspiration"])),
             "tmax": np.array([era["temperature_2m"][inv == i].max() for i in range(len(ud))])}
    daily["rain_ghcn"] = np.array([rain_g.get(d.astype(dt.date), np.nan) for d in ud])
    return ud, daily, (wtimes, era, rain_g)


def main(args):
    anc = args.ancillary or cli.data_root() / "ancillary"
    root = zarr.open_group(args.store, mode="r")
    br, bc = root.attrs["window"]
    zone = tuple(args.zone)
    segment = root.attrs.get("segment", args.segment)
    days = [dt.datetime.strptime(a["date"], "%y%m%d").date() for a in root.attrs["acquisitions"]]
    with open(anc / "uavsar" / "acquisition_times.csv") as fh:
        times = [dt.datetime.fromisoformat(r["start_utc"]) for r in csv.DictReader(fh)]
    if [t.date() for t in times] != days:
        raise ValueError("acquisition_times.csv does not match the scan's acquisitions")
    inputs = [Path(args.store), args.llh, args.lkv, args.ann, args.footprints]
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), inputs,
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    out = zarr.open_group(run.large_dir / "context.zarr", mode="w")
    t0 = time.time()

    I_cells, J_cells = root[args.pol]["lags"].shape[:2]
    ni, nj = I_cells // zone[0], J_cells // zone[1]
    lon, lat = lc.zone_lonlat(args.llh, root.attrs["image_shape"], (br, bc), zone, (ni, nj))
    lon, lat = lon.ravel().astype(np.float64), lat.ravel().astype(np.float64)
    out.create_array("geo/lon", data=lon.astype(np.float32))
    out.create_array("geo/lat", data=lat.astype(np.float32))
    epsg, x0, y0, index = map_grid(lon, lat, args.res, args.max_dist)
    out.create_array("geo/index", data=index, chunks=index.shape)
    logger.info("map grid %s at %g m, %.0f %% of pixels hold a zone", index.shape, args.res,
                100 * (index >= 0).mean())

    ud, daily, (wtimes, era, rain_g) = daily_weather(anc)
    for k, v in daily.items():
        out.create_array(f"weather/{k}", data=np.asarray(v, np.float32))
    acq_wx = [{k: (None if not np.isfinite(v) else round(float(v), 4))
               for k, v in wx.acquisition_weather(w, wtimes, era, rain_g).items()} for w in times]

    geometry = acquisition_geometry(args, segment, epsg, x0, y0, args.res, index.shape)
    local = [(w + dt.timedelta(hours=-7 if 3 <= w.month <= 10 else -8)).strftime("%Y-%m-%d %H:%M")
             for w in times]
    out.attrs.update({
        "geometry": geometry, "segment": segment, "source_run": root.attrs.get("run_id"),
        "pol": args.pol, "zone_cells": list(zone), "cell_window": [br, bc], "zone_grid": [ni, nj],
        "acquisitions": [{"index": i, "date": str(d), "utc": w.isoformat(), "local": loc,
                          "weather": acq_wx[i]}
                         for i, (d, w, loc) in enumerate(zip(days, times, local))],
        "weather_days": [str(ud[0]), str(ud[-1])],
        "map": {"epsg": epsg, "x0": x0, "y0": y0, "res": args.res, "shape": list(index.shape)},
    })
    finish_run(run, {"context": run.large_dir / "context.zarr"},
               extra={"elapsed_min": (time.time() - t0) / 60, "ancillary": str(anc)})
    logger.info("done: %s", run.large_dir / "context.zarr")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", type=Path, required=True, help="UAVSAR scan store of the segment.")
    p.add_argument("--segment", type=int, default=1,
                   help="Segment (used for the default inputs; the store's own value wins).")
    p.add_argument("--uavsar-dir", type=Path, default=None,
                   help="Default: $DATA_ROOT/uavsar/SDelta_23518.")
    p.add_argument("--llh", type=Path, default=None, help="Default: the segment's 2x8 .llh.")
    p.add_argument("--lkv", type=Path, default=None,
                   help="Default: the 2x8 .lkv of segment 1 (same range geometry for both).")
    p.add_argument("--ann", type=Path, default=None,
                   help="Default: the HH annotation of the first acquisition.")
    p.add_argument("--footprints", type=Path,
                   default=cli.REPO / "data" / "opera_cslc_footprints.json")
    p.add_argument("--ancillary", type=Path, default=None, help="Default: $DATA_ROOT/ancillary.")
    p.add_argument("--pol", default="VV")
    p.add_argument("--zone", type=int, nargs=2, default=[5, 5])
    p.add_argument("--res", type=float, default=40.0, help="Map grid spacing (m).")
    p.add_argument("--max-dist", type=float, default=50.0,
                   help="Map grid: max distance to a zone centre (m).")
    cli.add_run_arguments(p, task="context-maps")
    a = p.parse_args()
    u = a.uavsar_dir or cli.uavsar_dir()
    a.llh = a.llh or u / f"SDelta_23518_02_BC_s{a.segment}_2x8.llh"
    a.lkv = a.lkv or u / "SDelta_23518_02_BC_s1_2x8.lkv"
    a.ann = a.ann or u / "SDelta_23518_09048_004_090718_L090HH_02_BC.ann"
    a.task = f"{a.task}-s{a.segment}"
    cli.setup_logging(a.debug)
    cli.log_arguments(a)
    main(a)
