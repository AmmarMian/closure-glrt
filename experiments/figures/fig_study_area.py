r"""Fig. 1 of the paper (two columns): study area and the two sensors over one zone.

(a) Landsat 8 natural colour image at its native 30 m on a north-up UTM grid, with the footprint
of the UAVSAR line SDelta_23518 (segment 1), the Sentinel-1 bursts of the tracks used and the
zone of (b, c); (b) UAVSAR L-band VV intensity of the zone; (c) Sentinel-1 C-band VV intensity of
the zone, three days earlier. (b) and (c) come from ``amplitude_crops.py``.

Inputs: the context store of the legacy run ``context-maps-s1`` (map grid, footprints) until that
experiment is ported, the raw Landsat C2 L2 tiles and the crops ``.npz``. Output: a PDF.

Example:
    python experiments/figures/fig_study_area.py --context <run>/context.zarr \
        --s1-triplets data/s1_triplets.csv --landsat-dir <landsat> --crops crops.npz \
        --out study_area.pdf
"""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.patches import Rectangle

from style import INK, PAGE, S1, UAVSAR, panel_label, style


def landsat_scene(landsat_dir, scenes, mp, res=30.0):
    """Natural colour reflectance of Landsat C2 L2 scenes on the map extent (north-up, ``res`` m).

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
    crs = f"EPSG:{mp['epsg']}"
    out = np.full((3, H, W), np.nan, np.float32)
    for sid in scenes:
        with rasterio.open(f"{landsat_dir}/{sid}_qa_pixel.tif") as src:
            qa = np.zeros((H, W), np.uint16)
            reproject(rasterio.band(src, 1), qa, dst_transform=dst_t, dst_crs=crs,
                      resampling=Resampling.nearest)
        bad = (qa == 0) | ((qa >> 3) & 1 == 1) | ((qa >> 4) & 1 == 1)
        for k, band in enumerate(("red", "green", "blue")):
            with rasterio.open(f"{landsat_dir}/{sid}_{band}.tif") as src:
                dn = np.zeros((H, W), np.float32)
                reproject(rasterio.band(src, 1), dn, dst_transform=dst_t, dst_crs=crs,
                          resampling=Resampling.nearest)
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


def db_image(ax, inten, extent, label):
    v = 10 * np.log10(inten)
    lo, hi = np.nanpercentile(v, [2, 98])
    ax.imshow(v, cmap="gray", vmin=lo, vmax=hi, extent=extent, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.02, 0.02, label, transform=ax.transAxes, fontsize=6, va="bottom", ha="left",
            color=INK, bbox=dict(fc="white", ec="none", pad=1.2))


def scale_bar(ax, extent, length=500.0):
    x0 = extent[0] + 0.62 * (extent[1] - extent[0])
    y0 = extent[3] - 0.06 * (extent[3] - extent[2])
    ax.plot([x0, x0 + length], [y0, y0], color="white", lw=2.2, solid_capstyle="butt")
    ax.plot([x0, x0 + length], [y0, y0], color=INK, lw=1.0, solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 - 0.025 * (extent[3] - extent[2]), f"{length:.0f} m",
            ha="center", va="top", fontsize=6, color=INK,
            bbox=dict(fc="white", ec="none", pad=0.8))


def fmt_date(s):
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


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

    img = stretch(landsat_scene(args.landsat_dir, args.scenes, mp))
    img_date = fmt_date(args.scenes[0].split("_")[3])
    with open(args.s1_triplets) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    tracks = sorted({r["track"] for r in rows if int(r["segment"]) == 1})
    crops = np.load(args.crops)
    x, y, cres = crops["x"], crops["y"], float(crops["res"])
    cext = [x[0] - cres / 2, x[-1] + cres / 2, y[-1] - cres / 2, y[0] + cres / 2]

    side = 4.25                                            # map height and width (in)
    fig = plt.figure(figsize=(PAGE, side + 0.35))
    left, bottom = 0.5 / PAGE, 0.3 / (side + 0.35)
    ax = fig.add_axes([left, bottom, side / PAGE, side / (side + 0.35)])
    ax.imshow(img, extent=ext_km, origin="upper")
    for poly in geo["s1_bursts"].values():
        if poly["track"] in tracks:
            bx, by = to_km(poly["polygon"])
            ax.plot(bx, by, color=S1, lw=0.5, alpha=0.75)
    fx, fy = to_km(geo["uavsar"]["footprint"])
    ax.plot(fx, fy, color=UAVSAR, lw=1.4)
    ax.add_patch(Rectangle((cext[0] / 1e3, cext[2] / 1e3), (cext[1] - cext[0]) / 1e3,
                           (cext[3] - cext[2]) / 1e3, fill=False, ec="white", lw=2.0))
    ax.add_patch(Rectangle((cext[0] / 1e3, cext[2] / 1e3), (cext[1] - cext[0]) / 1e3,
                           (cext[3] - cext[2]) / 1e3, fill=False, ec=INK, lw=0.8))
    ax.text(cext[1] / 1e3 + 0.4, cext[3] / 1e3, "(b, c)", fontsize=6, color=INK, va="top",
            bbox=dict(fc="white", ec="none", pad=0.8))
    ax.set_xlim(ext_km[:2])
    ax.set_ylim(ext_km[2:])
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km, UTM 10N)", labelpad=1)
    ax.set_ylabel("Northing (km)", labelpad=1)
    ax.text(0.02, 0.02, f"Landsat 8, {img_date}", transform=ax.transAxes, fontsize=6,
            va="bottom", color=INK, bbox=dict(fc="white", ec="none", pad=1.2))
    ax.text(0.98, 0.98, "UAVSAR SDelta_23518 (s1)", transform=ax.transAxes, color=INK,
            fontsize=6, ha="right", va="top", bbox=dict(fc="white", ec=UAVSAR, lw=0.8, pad=1.5))
    ax.text(0.98, 0.92, "Sentinel-1 bursts " + ", ".join(tracks), transform=ax.transAxes,
            color=INK, fontsize=6, ha="right", va="top",
            bbox=dict(fc="white", ec=S1, lw=0.8, pad=1.5))
    panel_label(ax, "(a)", bg="white")

    gap = 0.1
    cside = (side - gap) / 2
    cx = left + (side + 0.15) / PAGE
    for k, (key, name, band, looks) in enumerate((
            ("uavsar", "UAVSAR", "L-band VV", "16 looks, 5 m"),
            ("s1", "Sentinel-1", "C-band VV", "2 looks, 10 m"))):
        axc = fig.add_axes([cx, bottom + (1 - k) * (cside + gap) / (side + 0.35),
                            cside / PAGE, cside / (side + 0.35)])
        date_ = fmt_date(str(crops[f"{key}_date"]).split(",")[0])
        db_image(axc, crops[key], cext, f"{name} {band}, {date_}\n{looks}")
        if k == 0:
            scale_bar(axc, cext)
        panel_label(axc, "(b)" if k == 0 else "(c)", bg="white")
    fig.savefig(args.out)
    print("Landsat", img_date, "tracks", tracks, "zone", [round(v) for v in cext])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--context", required=True)
    p.add_argument("--s1-triplets", required=True)
    p.add_argument("--landsat-dir", required=True, help="Raw Landsat C2 L2 tiles (<id>_<band>.tif).")
    p.add_argument("--scenes", nargs="+",
                   default=["LC08_L2SP_044034_20170614_02_T1", "LC08_L2SP_044033_20130721_02_T1"],
                   help="Landsat product ids, mosaicked in order (the first one has priority).")
    p.add_argument("--crops", required=True, help="Output of amplitude_crops.py.")
    p.add_argument("--out", default="study_area.pdf")
    main(p.parse_args())
