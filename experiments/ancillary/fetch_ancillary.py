r"""Download the ancillary data of the application section (Sec. VII) over the SDelta UAVSAR scene.

All sources are public and need no account. Each subcommand writes under ``--out/<source>/``
and appends one line per file to ``--out/manifest.jsonl`` (source URL without query string,
sha256, size, retrieval time); files already present are skipped.

- ``times``: UAVSAR acquisition start times (UTC), from the ``Start Time of Acquisition`` of the
  HH ``.ann`` files of ``--ann-dir`` (default ``$DATA_ROOT/uavsar/SDelta_23518``), sorted by time:
  ``uavsar/acquisition_times.csv``.
- ``cdl``: USDA Cropland Data Layer (30 m, EPSG:5070), years ``--years``, from the Microsoft
  Planetary Computer (``usda-cdl``, Cloud-Optimised GeoTIFF tiles), mosaicked and clipped to the
  scene box: ``cdl/cdl_<year>.tif``.
- ``landsat``: Landsat Collection 2 Level-2 surface reflectance (Planetary Computer,
  ``landsat-c2-l2``), for every UAVSAR date the item nearest in time of each WRS path within
  ``--days`` days with cloud cover at most ``--max-cloud`` %; bands blue, green, red, nir08,
  swir16, qa_pixel, clipped to the scene box: ``landsat/<item id>_<band>.tif``.
- ``ghcn``: NOAA GHCN-Daily station records (precipitation, temperatures, evaporation, wind) of the
  stations within ``--pad`` degrees of the box: ``ghcn/stations.csv``, ``ghcn/daily.csv``.
- ``era5``: ERA5 / ERA5-Land hourly reanalysis from the Open-Meteo archive API on a grid of
  ``--grid`` points over the box (rain, temperature, humidity, dew point, wind, soil moisture,
  reference evapotranspiration): ``era5/point_<i>_<j>.json``.
- ``soil``: gNATSGO soil survey (Planetary Computer): rasters of map unit key, soil organic
  carbon and available water storage of the top 30 cm, drought class, clipped to the box
  (``soil/gnatsgo_<band>.tif``), and the ``muaggatt``, ``component`` and ``mapunit`` tables
  (CONUS parquet, ``soil/tables/``).
- ``dwr``: California DWR field-level crop maps, statewide 2014, 2016, 2018 and Delta 2015, 2017
  (shapefile zips from data.cnra.ca.gov, ``dwr/``).

Output: ``--out`` (default ``$DATA_ROOT/ancillary``), about 4.5 GB (3.9 GB of Landsat). The
experiments read this folder with ``--ancillary``.

Example:
    uv run --extra geo python experiments/ancillary/fetch_ancillary.py all
    uv run --extra geo python experiments/ancillary/fetch_ancillary.py era5 ghcn cdl
"""

import argparse
import datetime as dt
import hashlib
import json
import logging
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from closure_glrt import cli
from closure_glrt.io.uavsar import parse_ann

BOX = (
    -122.25,
    37.72,
    -121.29,
    38.37,
)  # lon/lat, union of the SDelta segments 1 and 2 (.llh), padded
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
BANDS = ("blue", "green", "red", "nir08", "swir16", "qa_pixel")
ERA5_VARS = (
    "precipitation",
    "rain",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
)
SOIL_RASTERS = ("mukey", "soc0_30", "aws0_30", "droughty")
SOIL_TABLES = ("muaggatt", "component", "mapunit")
DWR = (
    ("statewide-crop-mapping", r"i15_crop_mapping_(2014|2016|2018)_shp"),
    ("region-land-use-surveys", r"(i15_crop_mapping_delta_2015|i15_landuse_delta2017)_shp"),
)
logger = logging.getLogger(__name__)
CODE = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True,
    text=True,
    cwd=Path(__file__).parent,
).stdout.strip()


def http(url, data=None, timeout=120, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"} if data else {}
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            logger.warning("%s: %s, retry", url.split("?")[0], e)
            time.sleep(10 * (k + 1))


def record(out, path, url, **extra):
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    line = {
        "file": str(Path(path).relative_to(out)),
        "url": url.split("?")[0],
        "sha256": h,
        "bytes": Path(path).stat().st_size,
        "retrieved_utc": dt.datetime.now(dt.UTC).isoformat(),
        "code": CODE,
        **extra,
    }
    with open(out / "manifest.jsonl", "a") as fh:
        fh.write(json.dumps(line) + "\n")


def stac_search(collection, datetime, **query):
    body = {"collections": [collection], "bbox": list(BOX), "limit": 200, **query}
    if datetime:
        body["datetime"] = datetime
    return json.loads(http(f"{STAC}/search", json.dumps(body).encode()))["features"]


def sas_token(collection):
    return json.loads(http(f"{SAS}/{collection}"))["token"]


def clip_to_box(hrefs, token, dst):
    """Mosaic of the COGs ``hrefs`` over BOX (in their CRS), written to ``dst`` (deflate)."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import transform_bounds

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="5"
    ):
        srcs = [rasterio.open(f"/vsicurl/{h}?{token}") for h in hrefs]
        bounds = transform_bounds("EPSG:4326", srcs[0].crs, *BOX)
        mosaic, transform = merge(srcs, bounds=bounds)
        prof = srcs[0].profile.copy()
        prof.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            count=mosaic.shape[0],
        )
        prof.pop("photometric", None)
        tmp = dst.with_suffix(".part")
        with rasterio.open(tmp, "w", **prof) as d:
            d.write(mosaic)
            try:  # CDL class colours
                d.write_colormap(1, srcs[0].colormap(1))
            except ValueError:
                pass
        for s in srcs:
            s.close()
        tmp.rename(dst)


def uavsar_times(args):
    """[(.ann name, start datetime UTC)] of the acquisitions (HH annotations), sorted by time."""
    out = []
    for ann in sorted(Path(args.ann_dir).glob("*_L090HH_*.ann")):
        t = parse_ann(ann)["Start Time of Acquisition"]
        out.append(
            (
                ann.name,
                dt.datetime.strptime(t.replace("UTC", "").strip(), "%d-%b-%Y %H:%M:%S").replace(
                    tzinfo=dt.UTC
                ),
            )
        )
    if not out:
        raise SystemExit(f"no *_L090HH_*.ann in {args.ann_dir}: run `just download uavsar ann`")
    return sorted(out, key=lambda x: x[1])


def cmd_times(args):
    d = args.out / "uavsar"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "acquisition_times.csv", "w") as fh:
        fh.write("index,ann,start_utc,start_local_pacific\n")
        for i, (name, t) in enumerate(uavsar_times(args)):
            off = -7 if 3 <= t.month <= 10 else -8  # PDT / PST, enough for a hint
            local = (t + dt.timedelta(hours=off)).strftime("%Y-%m-%d %H:%M")
            fh.write(f"{i},{name},{t.isoformat()},{local}\n")
    logger.info("times: %s", d / "acquisition_times.csv")


def cmd_cdl(args):
    d = args.out / "cdl"
    d.mkdir(parents=True, exist_ok=True)
    token = sas_token("usda-cdl")
    for year in args.years:
        dst = d / f"cdl_{year}.tif"
        if dst.exists():
            continue
        items = [
            f
            for f in stac_search("usda-cdl", f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z")
            if f["properties"].get("usda_cdl:type") == "cropland" and "cropland" in f["assets"]
        ]
        hrefs = [f["assets"]["cropland"]["href"] for f in items]
        clip_to_box(hrefs, token, dst)
        record(
            args.out,
            dst,
            ";".join(hrefs),
            source="usda-cdl",
            year=year,
            items=[f["id"] for f in items],
        )
        logger.info("cdl %d: %d tiles", year, len(hrefs))


def cmd_landsat(args):
    d = args.out / "landsat"
    d.mkdir(parents=True, exist_ok=True)
    days = sorted({t.date() for _, t in uavsar_times(args)})
    token, t_tok = sas_token("landsat-c2-l2"), time.time()
    picked = {}
    for day in days:
        lo, hi = day - dt.timedelta(days=args.days), day + dt.timedelta(days=args.days)
        items = [
            f
            for f in stac_search("landsat-c2-l2", f"{lo}T00:00:00Z/{hi}T23:59:59Z")
            if f["properties"]["eo:cloud_cover"] <= args.max_cloud
        ]
        best = {}
        for f in items:  # nearest per path and row
            p = f["properties"]
            key = (p["landsat:wrs_path"], p["landsat:wrs_row"])
            gap = abs((dt.date.fromisoformat(p["datetime"][:10]) - day).days)
            if key not in best or gap < best[key][0]:
                best[key] = (gap, f)
        for gap, f in best.values():
            picked.setdefault(f["id"], (f, []))[1].append(str(day))
    logger.info("landsat: %d items for %d UAVSAR days", len(picked), len(days))
    with open(d / "items.json", "w") as fh:
        json.dump(
            {
                k: {
                    "datetime": f["properties"]["datetime"],
                    "platform": f["properties"]["platform"],
                    "cloud": f["properties"]["eo:cloud_cover"],
                    "uavsar_days": u,
                }
                for k, (f, u) in sorted(picked.items())
            },
            fh,
            indent=1,
        )
    for n, (iid, (f, _)) in enumerate(sorted(picked.items())):
        if time.time() - t_tok > 1800:  # tokens last about an hour
            token, t_tok = sas_token("landsat-c2-l2"), time.time()
        for band in BANDS:
            dst = d / f"{iid}_{band}.tif"
            if dst.exists() or band not in f["assets"]:
                continue
            href = f["assets"][band]["href"]
            try:
                clip_to_box([href], token, dst)
            except Exception as e:  # noqa: BLE001
                logger.warning("%s %s: %s", iid, band, e)
                continue
            record(args.out, dst, href, source="landsat-c2-l2", item=iid, band=band)
        if n % 20 == 0:
            logger.info("landsat %d / %d", n, len(picked))


def cmd_ghcn(args):
    d = args.out / "ghcn"
    d.mkdir(parents=True, exist_ok=True)
    url = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
    lon0, lat0, lon1, lat1 = BOX
    rows = []
    for line in http(url).decode().splitlines():
        sid, lat, lon = line[:11], float(line[12:20]), float(line[21:30])
        if lat0 - args.pad <= lat <= lat1 + args.pad and lon0 - args.pad <= lon <= lon1 + args.pad:
            rows.append((sid, lat, lon, float(line[31:37]), line[41:71].strip()))
    with open(d / "stations.csv", "w") as fh:
        fh.write("station,lat,lon,elev_m,name\n")
        fh.writelines(f'{s},{a},{o},{e},"{n}"\n' for s, a, o, e, n in rows)
    record(args.out, d / "stations.csv", url, source="ghcnd-stations")
    lines = []
    for k in range(0, len(rows), 25):  # the API refuses long station lists
        q = urllib.parse.urlencode(
            {
                "dataset": "daily-summaries",
                "stations": ",".join(r[0] for r in rows[k : k + 25]),
                "startDate": args.start,
                "endDate": args.end,
                "dataTypes": "PRCP,TMAX,TMIN,TAVG,EVAP,AWND,SNOW,WT01,WT16",
                "format": "csv",
                "units": "metric",
                "includeStationName": "true",
            }
        )
        part = (
            http(f"https://www.ncei.noaa.gov/access/services/data/v1?{q}", timeout=600)
            .decode()
            .splitlines()
        )
        lines += part if not lines else part[1:]
    (d / "daily.csv").write_text("\n".join(lines) + "\n")
    record(
        args.out,
        d / "daily.csv",
        "https://www.ncei.noaa.gov/access/services/data/v1",
        source="ghcn-daily",
        query={"stations": len(rows), "start": args.start, "end": args.end},
    )
    logger.info("ghcn: %d stations, %d lines", len(rows), len(lines))


def cmd_soil(args):
    d = args.out / "soil"
    d.mkdir(parents=True, exist_ok=True)
    token = json.loads(http(f"{SAS}/soils/gnatsgo"))["token"]  # the rasters' storage container
    items = stac_search("gnatsgo-rasters", None)
    for band in SOIL_RASTERS:
        dst = d / f"gnatsgo_{band}.tif"
        if dst.exists():
            continue
        hrefs = [f["assets"][band]["href"] for f in items]
        clip_to_box(hrefs, token, dst)
        record(args.out, dst, ";".join(hrefs), source="gnatsgo-rasters", band=band)
        logger.info("soil raster %s", band)
    blob = "https://soils.blob.core.windows.net/gnatsgo"
    for table in SOIL_TABLES:
        listing = http(
            f"{blob}?restype=container&comp=list&prefix=july2020/parquet/{table}.parquet/&{token}"
        ).decode()
        for name in re.findall(r"<Name>([^<]+)</Name>", listing):
            dst = d / "tables" / name.split("parquet/", 1)[1]
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(http(f"{blob}/{name}?{token}", timeout=1200))
            record(args.out, dst, f"{blob}/{name}", source="gnatsgo-tables", table=table)
        logger.info("soil table %s", table)


def cmd_dwr(args):
    d = args.out / "dwr"
    d.mkdir(parents=True, exist_ok=True)
    for pkg, pattern in DWR:
        res = json.loads(http(f"https://data.cnra.ca.gov/api/3/action/package_show?id={pkg}"))[
            "result"
        ]["resources"]
        for r in res:
            url = r["url"]
            if not re.search(pattern, url):
                continue
            dst = d / url.rsplit("/", 1)[1]
            dst = dst if dst.suffix == ".zip" else dst.with_name(dst.name + ".zip")
            if dst.exists():
                continue
            # the portal redirects to a signed S3 URL that urllib refuses; curl follows it
            subprocess.run(["curl", "-sSfL", "--retry", "4", "-o", str(dst), url], check=True)
            record(args.out, dst, url, source=f"cnra:{pkg}", name=r.get("name"))
            logger.info("dwr %s (%.0f MB)", dst.name, dst.stat().st_size / 1e6)


def cmd_era5(args):
    d = args.out / "era5"
    d.mkdir(parents=True, exist_ok=True)
    lon0, lat0, lon1, lat1 = BOX
    for i, lat in enumerate(np.linspace(lat0, lat1, args.grid[0])):
        for j, lon in enumerate(np.linspace(lon0, lon1, args.grid[1])):
            dst = d / f"point_{i}_{j}.json"
            if dst.exists():
                continue
            q = urllib.parse.urlencode(
                {
                    "latitude": round(lat, 4),
                    "longitude": round(lon, 4),
                    "start_date": args.start,
                    "end_date": args.end,
                    "hourly": ",".join(ERA5_VARS),
                    "timezone": "GMT",
                }
            )
            url = f"https://archive-api.open-meteo.com/v1/archive?{q}"
            dst.write_bytes(http(url, timeout=600))
            record(args.out, dst, url, source="open-meteo-archive", query=q)
            logger.info("era5 point %d %d", i, j)
            time.sleep(args.sleep)


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    cmds = {
        "times": cmd_times,
        "cdl": cmd_cdl,
        "ghcn": cmd_ghcn,
        "era5": cmd_era5,
        "landsat": cmd_landsat,
        "soil": cmd_soil,
        "dwr": cmd_dwr,
    }
    for name in cmds if "all" in args.cmd else args.cmd:
        cmds[name](args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "cmd", nargs="+", choices=["times", "cdl", "landsat", "ghcn", "era5", "soil", "dwr", "all"]
    )
    parser.add_argument("--out", type=Path, default=cli.data_root() / "ancillary")
    parser.add_argument(
        "--ann-dir", type=Path, default=cli.uavsar_dir(), help="UAVSAR .ann files (times, landsat)."
    )
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2009, 2018)))
    parser.add_argument(
        "--days", type=int, default=16, help="Landsat: max days from a UAVSAR date."
    )
    parser.add_argument("--max-cloud", type=float, default=30.0)
    parser.add_argument("--pad", type=float, default=0.15, help="GHCN: degrees around the box.")
    parser.add_argument(
        "--grid", type=int, nargs=2, default=[3, 4], help="ERA5: points along lat, lon."
    )
    parser.add_argument("--start", default="2009-06-01")
    parser.add_argument("--end", default="2017-12-31")
    parser.add_argument("--sleep", type=float, default=20.0, help="ERA5: seconds between requests.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    cli.setup_logging(debug=args.debug)
    cli.log_arguments(args)
    main(args)
