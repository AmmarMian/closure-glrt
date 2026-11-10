r"""Fig. 1 of the paper: study area and acquisitions (segment 1).

(a) Landsat natural colour image on the north-up 40 m grid of the context store, with the
footprint of the UAVSAR line SDelta_23518 (segment 1) and the Sentinel-1 bursts of the tracks
used; (b) dates of the 68 UAVSAR acquisitions and of the Sentinel-1 acquisitions of the matched
triplets, the same-day triplets labelled; (c) daily rain (mean of the GHCN stations); (d)
volumetric soil moisture of the top 7 cm (ERA5).

Input for now: the context store of the legacy run ``context-maps-s1`` (Landsat on the map
grid, footprints, weather series), until that experiment is ported. Output: a PDF.

Example:
    python experiments/figures/fig1_data.py --context <run>/context.zarr \
        --s1-triplets data/s1_triplets.csv --out fig1_data.pdf
"""

import argparse
import csv
from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

UAVSAR, S1, SAMEDAY = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1, 2, 3
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d2"


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42})


def pick_landsat(acquisitions, gap):
    """Index of the summer (June--August) Landsat-8 era image with the fewest gap pixels."""
    frac = gap.reshape(gap.shape[0], -1).mean(1)
    ok = [i for i, x in enumerate(acquisitions)
          if int(x["date"][:4]) >= 2013 and int(x["date"][5:7]) in (6, 7, 8)]
    ok = ok or list(range(len(acquisitions)))
    return min(ok, key=lambda i: frac[i])


def landsat_scene(landsat_dir, scenes, mp, res=30.0):
    """Natural colour reflectance of one Landsat C2 L2 date on the map extent (north-up, ``res`` m).

    ``scenes``: product ids mosaicked in order, the first one having priority;
    surface reflectance = DN * 2.75e-5 - 0.2; pixels flagged as fill, cloud or cloud shadow in
    ``qa_pixel`` are left empty.
    """
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    H0, W0 = mp["shape"]
    W, H = int(W0 * mp["res"] / res), int(H0 * mp["res"] / res)
    dst_t = from_origin(mp["x0"], mp["y0"], res, res)
    out = np.full((3, H, W), np.nan, np.float32)
    for sid in scenes:
        qa_path = f"{landsat_dir}/{sid}_qa_pixel.tif"
        with rasterio.open(qa_path) as src:
            qa = np.zeros((H, W), np.uint16)
            reproject(rasterio.band(src, 1), qa, dst_transform=dst_t, dst_crs=f"EPSG:{mp['epsg']}",
                      resampling=Resampling.nearest)
        bad = (qa == 0) | ((qa >> 3) & 1 == 1) | ((qa >> 4) & 1 == 1)
        for k, band in enumerate(("red", "green", "blue")):
            with rasterio.open(f"{landsat_dir}/{sid}_{band}.tif") as src:
                dn = np.zeros((H, W), np.float32)
                reproject(rasterio.band(src, 1), dn, dst_transform=dst_t,
                          dst_crs=f"EPSG:{mp['epsg']}", resampling=Resampling.bilinear)
            refl = np.where(bad | (dn == 0), np.nan, dn * 2.75e-5 - 0.2)
            fill = np.isnan(out[k]) & np.isfinite(refl)
            out[k][fill] = refl[fill]
    return np.moveaxis(out, 0, -1)


def stretch(x, gamma=0.75):
    """Common 1--99 % stretch of the three bands (keeps colour balance), gamma, white where empty."""
    v = x[np.isfinite(x)]
    lo, hi = np.percentile(v, [1, 99])
    y = np.clip((x - lo) / (hi - lo), 0, 1) ** gamma
    return np.where(np.isfinite(y), y, 1.0)


def rgb(bands):
    """Natural colour (red, green, blue) with a 2--98 % stretch per band."""
    out = np.stack([bands[2], bands[1], bands[0]], -1).astype(float)
    for k in range(3):
        v = out[..., k][out[..., k] > 0]
        lo, hi = np.percentile(v, [2, 98]) if v.size else (0, 255)
        out[..., k] = np.clip((out[..., k] - lo) / (hi - lo), 0, 1)
    return out


def main(args):
    style()
    g = zarr.open_group(args.context, mode="r")
    a = dict(g.attrs)
    mp, geo = a["map"], a["geometry"]
    H, W = mp["shape"]
    res = mp["res"]
    ext_km = [mp["x0"] / 1e3, (mp["x0"] + res * W) / 1e3, (mp["y0"] - res * H) / 1e3,
              mp["y0"] / 1e3]

    def to_km(poly):
        p = np.asarray(poly, float)                       # map pixel (col, row)
        return mp["x0"] / 1e3 + p[:, 0] * res / 1e3, mp["y0"] / 1e3 - p[:, 1] * res / 1e3

    if args.landsat_dir:
        img = stretch(landsat_scene(args.landsat_dir, args.scenes, mp))
        img_date = "-".join((args.scenes[0].split("_")[3][:4], args.scenes[0].split("_")[3][4:6],
                             args.scenes[0].split("_")[3][6:]))
    else:
        gap = g["image/gap"][:]
        t_img = pick_landsat(a["acquisitions"], gap)
        img = rgb(g["image/bands"][t_img])
        img_date = a["acquisitions"][t_img]["date"]

    acq = [date.fromisoformat(x["date"]) for x in a["acquisitions"]]
    with open(args.s1_triplets) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    s1 = sorted({date(int(d[:4]), int(d[4:6]), int(d[6:])) for r in rows
                 for d in (r["d1"], r["d2"], r["d3"]) if int(r["segment"]) == 1})
    tracks = sorted({r["track"] for r in rows if int(r["segment"]) == 1})
    d0 = date.fromisoformat(a["weather_days"][0])
    rain = g["weather/rain_ghcn"][:]
    sm = g["weather/sm0_7"][:]
    days = np.array([d0 + timedelta(days=int(i)) for i in range(len(rain))])

    fig = plt.figure(figsize=(3.5, 5.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[3.3, 0.55, 0.75, 0.75], hspace=0.16,
                          left=0.15, right=0.97, top=0.99, bottom=0.05)
    ax = fig.add_subplot(gs[0])
    ax.imshow(img, extent=ext_km, origin="upper", interpolation="nearest")
    for b, poly in geo["s1_bursts"].items():
        if poly["track"] in tracks:
            x, y = to_km(poly["polygon"])
            ax.plot(x, y, color=S1, lw=0.5, alpha=0.75)
    x, y = to_km(geo["uavsar"]["footprint"])
    ax.plot(x, y, color=UAVSAR, lw=1.4)
    ax.set_xlim(ext_km[:2])
    ax.set_ylim(ext_km[2:])
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km, UTM 10N)", labelpad=1)
    ax.set_ylabel("Northing (km)", labelpad=1)
    ax.text(0.03, 0.03, f"Landsat 8, {img_date}", transform=ax.transAxes, color="white",
            fontsize=6, va="bottom")
    ax.text(0.98, 0.97, "UAVSAR SDelta_23518 (s1)", transform=ax.transAxes, color=INK,
            fontsize=6, ha="right", va="top", bbox=dict(fc="white", ec=UAVSAR, lw=0.8, pad=1.5))
    ax.text(0.98, 0.89, "Sentinel-1 bursts " + ", ".join(tracks), transform=ax.transAxes,
            color=INK, fontsize=6, ha="right", va="top",
            bbox=dict(fc="white", ec=S1, lw=0.8, pad=1.5))
    ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, fontsize=7, va="top", ha="left", fontweight="bold", color="white")

    ax1 = fig.add_subplot(gs[1])
    ax1.vlines(acq, 0.55, 0.95, color=UAVSAR, lw=0.7)
    ax1.vlines(s1, 0.05, 0.45, color=S1, lw=0.7)
    for label, idx in (("$t=10$", 10), ("$t=23$", 23)):
        ax1.vlines(acq[idx], 0.55, 0.95, color=SAMEDAY, lw=1.6)
        ax1.text(acq[idx], 1.0, label, fontsize=6, ha="center", va="bottom", color=INK)
    ax1.set_ylim(0, 1.35)
    ax1.set_yticks([0.75, 0.25], ["UAVSAR", "S1"])
    ax1.tick_params(axis="y", length=0)
    ax1.text(0.0, 1.0, "(b)", transform=ax1.transAxes, fontsize=7, va="top", ha="left", fontweight="bold", color=INK)

    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax2.bar(days, rain, width=1.0, color=MUTED, lw=0)
    ax2.set_ylabel("Rain\n(mm/day)", labelpad=1)
    ax2.text(0.0, 1.0, "(c)", transform=ax2.transAxes, fontsize=7, va="top", ha="left", fontweight="bold", color=INK)

    ax3 = fig.add_subplot(gs[3], sharex=ax1)
    ax3.plot(days, sm, color=INK, lw=0.8)
    ax3.set_ylabel("Soil moist.\n(m$^3$/m$^3$)", labelpad=1)
    ax3.text(0.0, 1.0, "(d)", transform=ax3.transAxes, fontsize=7, va="top", ha="left", fontweight="bold", color=INK)
    for axx in (ax1, ax2, ax3):
        for s in ("top", "right"):
            axx.spines[s].set_visible(False)
        axx.grid(axis="x", color=GRID, lw=0.4)
        axx.set_xlim(date(2009, 6, 1), date(2018, 1, 1))
    for axx in (ax1, ax2):
        plt.setp(axx.get_xticklabels(), visible=False)
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(args.out)
    print("Landsat date", img_date, "S1 dates", len(s1), "tracks", tracks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--context", required=True)
    p.add_argument("--s1-triplets", required=True)
    p.add_argument("--out", default="fig1_data.pdf")
    p.add_argument("--landsat-dir", default=None,
                   help="Raw Landsat C2 L2 tiles (<id>_<band>.tif); default: the context composite.")
    p.add_argument("--scenes", nargs="+",
                   default=["LC08_L2SP_044034_20170614_02_T1", "LC08_L2SP_044033_20130721_02_T1"],
                   help="Landsat product ids, mosaicked in order (the first one has priority).")
    main(p.parse_args())
