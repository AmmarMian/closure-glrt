r"""UAVSAR and Sentinel-1 closures on common coherent zones of matched triplets (Section V-C).

Inputs: the UAVSAR scan store of a segment and its 2x8 ``.llh``; the CSLC burst scan store
(``experiments/stores/cslc_scan.py``, lags 0 to 2 of consecutive dates, used for the long-term
Sentinel-1 coherence); the CSLC triplet store of the segment (``experiments/stores/
cslc_triplets.py``, 3 x 3 cell covariances of the Sentinel-1 triplets matched to UAVSAR
triplets). VV for both sensors.

**Zones.** A UTM grid (EPSG 32610) of squares of ``--zone-m`` metres. Each UAVSAR cell
(``--uavsar-factor`` base cells, centre geolocated with the ``.llh``) and each Sentinel-1 cell
(20 m x 20 m) falls into one zone by its centre. Where bursts of a track overlap, a zone takes
the cells of the burst that gives it most valid cells.

**Matching.** A Sentinel-1 triplet is matched to the UAVSAR triplet $(t,t+1,t+2)$ whose dates
are all within ``--max-dt`` days of its own (UAVSAR acquisitions at ``UAVSAR_HOUR`` UTC).

**Selection (coherent in both sensors, independently of the tested data).** The selection
coherence of a cell is the median of its lag-1 coherences over the pairs of consecutive dates
of ``--period`` that share no date with the tested triplets (both sensors' dates excluded). For
Sentinel-1 it is computed on 40 m cells of track T035, the only track with a series, and used
for every track. A zone is kept if the medians over its cells pass ``--coh-uavsar`` and
``--coh-s1`` and both sensors have at least ``--min-cells`` valid cells in it.

**Per zone and sensor:** the exact scale-free GLRT (:func:`closure_glrt.glrt.glrt`, weights
$\kappa_0$ at the nominal looks), its p-value, $\hat\mu$, $\hat c$, the half-width $h$ of the
$1-\alpha$ confidence arc (:func:`closure_glrt.glrt.confidence_halfwidth`) and the small-angle
standard error $s=\sqrt{2(W-R)/((L-1)W)}$ of $\hat\mu$.

**Per matched triplet,** on the common coherent zones: detected shares at ``--pfa``; median
$\hat c$, $|\hat\mu|$ and $h$ of each sensor; Pearson and Spearman correlation of the two
$\hat\mu$ with a spatial block bootstrap interval (blocks of ``--block-zones`` zones a side);
sign agreement on the zones detected by both; the attenuation bound $r_{\max}=\sqrt{\varrho_U
\varrho_S}$, $\varrho=\max(0,1-\overline{s^2}/\operatorname{var}\hat\mu)$ the share of the spread
of $\hat\mu$ that is not estimation noise (robust version with medians and MAD), i.e. the
correlation expected if both sensors measured the same closures; and, as a reference, the
correlation of the UAVSAR $\hat\mu$ with Sentinel-1 triplets of the same track that share no
date (no link expected).

Outputs: ``summary.csv`` (one row per matched triplet) and ``zones.npz`` (per triplet: zone keys,
both sensors' ``mu``, ``p``, ``c1``, ``h``, ``se``, selection coherences, valid cells), small dir.
A zone key is ``floor(E / zone_m) * 1_000_000 + floor(N / zone_m)``.

Example:
    uv run python experiments/sentinel1/cross_sensor.py --uavsar-store <run>/scan.zarr \
        --llh SDelta_23518_02_BC_s1_2x8.llh --cslc-store <run>/scan.zarr \
        --triplets-store <run>/triplets.zarr --task cross-sensor-s1
"""

import argparse
import csv
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyproj
import zarr
from scipy import stats

from closure_glrt import cli
from closure_glrt.backend import get_data_on_device, to_numpy
from closure_glrt.closure import aggregate_cells, dezan_concentration
from closure_glrt.glrt import confidence_halfwidth, glrt, pvalue
from closure_glrt.provenance import finish_run, start_run

UAVSAR_HOUR = 22.4  # UTC start of the SDelta flights
logger = logging.getLogger(__name__)
np.seterr(all="ignore")


def coherence(L, a, k=1):
    """|gamma| between dates a and a+k of cells L (..., T, K+1)."""
    return np.abs(L[..., a, k]) / np.sqrt(L[..., a, 0].real * L[..., a + k, 0].real)


def closure_from_cov(P1, P2, P3, L12, L23, L13, n_looks):
    g12 = np.abs(L12) / np.sqrt(P1 * P2)
    g23 = np.abs(L23) / np.sqrt(P2 * P3)
    g13 = np.abs(L13) / np.sqrt(P1 * P3)
    phi = np.angle(L12 * L23 * np.conj(L13))
    kappa = dezan_concentration(g12, g23, g13, n_looks)
    ok = np.isfinite(phi) & np.isfinite(kappa) & (kappa > 0)
    return np.where(ok, phi, np.nan), np.where(ok, kappa, np.nan)


def zone_keys(x, y, zone_m):
    return (np.floor(x / zone_m).astype(np.int64) * 1_000_000
            + np.floor(y / zone_m).astype(np.int64))


def group(keys, *values, max_cells=64):
    """Cells -> zones: sorted unique keys, counts and (Z, Lmax) arrays padded with nan."""
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    uniq, start, counts = np.unique(k, return_index=True, return_counts=True)
    Lmax = min(counts.max(), max_cells)
    pos = np.arange(len(k)) - np.repeat(start, counts)
    keep = pos < Lmax
    zi = np.repeat(np.arange(len(uniq)), counts)[keep]
    out = []
    for v in values:
        a = np.full((len(uniq), Lmax), np.nan)
        a[zi, pos[keep]] = v[order][keep]
        out.append(a)
    return uniq, np.minimum(counts, Lmax), out


def zone_test(phi, kappa, n_min, alpha, backend):
    """Exact GLRT, p-value, mu, c1, arc half-width and standard error per padded zone."""
    valid = np.isfinite(phi) & np.isfinite(kappa)
    n = valid.sum(-1)
    res = {k: np.full(len(phi), np.nan) for k in ("p", "c1", "mu", "h", "se")}
    sel = np.flatnonzero(n >= n_min)
    if len(sel):
        p = np.where(valid[sel], phi[sel], 0.0)
        w = np.where(valid[sel], kappa[sel], 0.0)
        pd, wd = get_data_on_device(p, backend), get_data_on_device(w, backend)
        r = glrt(pd, wd, backend, details=True)
        res["p"][sel] = pvalue(r["lam"], n[sel])
        res["mu"][sel] = to_numpy(r["mu"])
        res["c1"][sel] = to_numpy(r["c1"])
        res["h"][sel] = to_numpy(confidence_halfwidth(pd, wd, alpha, backend))
        R, W = to_numpy(r["R"]), to_numpy(r["W"])
        res["se"][sel] = np.sqrt(2 * (W - R) / ((n[sel] - 1) * W))
    return res, n


class Uavsar:
    """VV lags of the scan store on cells of ``factor`` base cells, with their zone keys."""

    def __init__(self, store, llh, factor, zone_m):
        self.root = zarr.open_group(store, mode="r")
        r, c = self.root.attrs["window"]
        self.factor, self.n_looks = tuple(factor), factor[0] * factor[1] * r * c
        self.dates = [datetime.strptime(a["date"], "%y%m%d") + timedelta(hours=UAVSAR_HOUR)
                      for a in self.root.attrs["acquisitions"]]
        self.lags_store = self.root["VV"]["lags"]
        ni, nj = (s // f for s, f in zip(self.lags_store.shape[:2], factor))
        H, W = self.root.attrs["image_shape"]
        a = np.memmap(llh, dtype="<f4", mode="r").reshape(-1, W // 2, 3)
        az = (np.arange(ni) * factor[0] * r + (factor[0] * r - 1) / 2) / 8
        rg = (np.arange(nj) * factor[1] * c + (factor[1] * c - 1) / 2) / 2
        ia = np.clip(np.round(az).astype(int), 0, a.shape[0] - 1)
        ir = np.clip(np.round(rg).astype(int), 0, a.shape[1] - 1)
        lat, lon = a[ia][:, ir, 0], a[ia][:, ir, 1]
        x, y = pyproj.Transformer.from_crs(4326, 32610, always_xy=True).transform(lon, lat)
        self.x, self.y = np.asarray(x).ravel(), np.asarray(y).ravel()   # cell centres (m)
        self.keys = zone_keys(self.x, self.y, zone_m)
        self.shape, self._cache = (ni, nj), {}

    def lags(self, lo, hi):
        """Aggregated lags of dates lo..hi-1 (cached by range)."""
        if (lo, hi) not in self._cache:
            self._cache = {(lo, hi): aggregate_cells(
                np.asarray(self.lags_store[:, :, lo:hi], dtype=np.complex128), self.factor)}
        return self._cache[lo, hi]

    def triplet(self, t, L, lo):
        T = L[:, :, t - lo:t - lo + 3]
        return closure_from_cov(T[..., 0, 0].real, T[..., 1, 0].real, T[..., 2, 0].real,
                                T[..., 0, 1], T[..., 1, 1], T[..., 0, 2], self.n_looks)

    def selection_coherence(self, L, lo, exclude, idx):
        use = [s for s in idx if not {s, s + 1} & set(exclude)]
        g = np.stack([coherence(L, s - lo) for s in use], axis=-1)
        return np.nanmedian(g, axis=-1), len(use)


def s1_cell_keys(attrs, shape, zone_m):
    ni, nj = shape
    X, Y = np.meshgrid(attrs["x0"] + attrs["dx"] * np.arange(nj),
                       attrs["y0"] + attrs["dy"] * np.arange(ni))
    return zone_keys(X, Y, zone_m).ravel()


class S1Coherence:
    """Lag-1 coherences of the T035 pairs of the period on 40 m cells; zone medians on demand."""

    def __init__(self, cslc_root, period, zone_m):
        self.bursts = []
        for b in cslc_root.attrs["bursts"]:
            if not b.startswith("T035"):
                continue
            g = cslc_root[b]
            dates = [datetime.strptime(a["date"] + a["time"], "%Y%m%d%H%M%S")
                     for a in g.attrs["acquisitions"]]
            idx = [s for s in range(len(dates) - 1)
                   if period[0] <= dates[s] and dates[s + 1] <= period[1]]
            L = aggregate_cells(np.asarray(g["VV"]["lags"][:]), (2, 2))
            coh = np.stack([coherence(L, s) for s in idx], -1).astype(np.float32)
            ni, nj = coh.shape[:2]
            coh = coh.reshape(ni * nj, -1)
            del L
            attrs = dict(g.attrs)
            attrs.update(x0=attrs["x0"] + attrs["dx"] * 3.5, y0=attrs["y0"] + attrs["dy"] * 1.5,
                         dx=attrs["dx"] * 8, dy=attrs["dy"] * 4)  # 40 m cell centres
            keys = s1_cell_keys(attrs, (ni, nj), zone_m)
            pairs = [(dates[s].strftime("%Y%m%d"), dates[s + 1].strftime("%Y%m%d")) for s in idx]
            ok = np.isfinite(coh).any(-1)
            self.bursts.append((keys[ok], coh[ok], pairs))
            logger.info("S1 coherence %s: %d valid 40 m cells, %d pairs", b, ok.sum(), len(pairs))

    def zones(self, exclude_dates):
        keys_all, coh_all = [], []
        for keys, coh, pairs in self.bursts:
            use = [i for i, (a, b) in enumerate(pairs)
                   if a not in exclude_dates and b not in exclude_dates]
            keys_all.append(keys)
            coh_all.append(np.nanmedian(coh[:, use], -1))
        uniq, _, (c,) = group(np.concatenate(keys_all), np.concatenate(coh_all), max_cells=4096)
        return dict(zip(uniq, np.nanmedian(c, axis=-1)))


def s1_triplet_zones(group_tr, zone_m, n_looks):
    """Cells of all bursts of one Sentinel-1 triplet -> per zone padded phi, kappa."""
    per_burst = []
    for b in sorted(group_tr.group_keys()):
        g = group_tr[b]
        cov = np.asarray(g["cov"][:], dtype=np.complex128)
        phi, kappa = closure_from_cov(cov[..., 0].real, cov[..., 1].real, cov[..., 2].real,
                                      cov[..., 3], cov[..., 4], cov[..., 5], n_looks)
        per_burst.append((s1_cell_keys(g.attrs, cov.shape[:2], zone_m), phi.ravel(),
                          kappa.ravel()))
    best = {}                                # zone -> burst with most valid cells
    for n_b, (keys, phi, _) in enumerate(per_burst):
        u, cnt = np.unique(keys[np.isfinite(phi)], return_counts=True)
        for k, m in zip(u, cnt):
            if m > best.get(k, (0, -1))[0]:
                best[k] = (m, n_b)
    keys_all, phi_all, kap_all = [], [], []
    for n_b, (keys, phi, kappa) in enumerate(per_burst):
        mine = np.array([best.get(k, (0, -1))[1] == n_b for k in keys])
        keys_all.append(keys[mine])
        phi_all.append(phi[mine])
        kap_all.append(kappa[mine])
    return group(np.concatenate(keys_all), np.concatenate(phi_all), np.concatenate(kap_all))


def reliability(mu, se, robust=False):
    if robust:
        v = (1.4826 * np.median(np.abs(mu - np.median(mu)))) ** 2
        return float(max(0.0, 1 - np.median(se ** 2) / v)) if v > 0 else np.nan
    v = np.var(mu)
    return float(max(0.0, 1 - np.mean(se ** 2) / v)) if v > 0 else np.nan


def block_bootstrap(keys, a, b, block, n_boot, rng, corr):
    """2.5 and 97.5 % quantiles of corr(a, b) under resampling of square blocks of zones."""
    ix, iy = keys // 1_000_000, keys % 1_000_000
    _, inv = np.unique((ix // block) * 1_000_000 + iy // block, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
    rs = []
    for _ in range(n_boot):
        pick = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        rs.append(corr(a[pick], b[pick]))
    return np.quantile(rs, [0.025, 0.975])


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return float(stats.spearmanr(a, b)[0])


def compare(k, u, s, args, rng):
    """Statistics of one matched triplet on its common coherent zones (u, s: per-zone dicts)."""
    du, ds = u["p"] < args.pfa, s["p"] < args.pfa
    both = du & ds
    rel = {f"{n}_{sensor}": reliability(d["mu"], d["se"], robust=(n == "rel_robust"))
           for n in ("rel", "rel_robust") for sensor, d in (("uavsar", u), ("s1", s))}
    lo, hi = block_bootstrap(k, u["mu"], s["mu"], args.block_zones, args.boot, rng, pearson)
    slo, shi = block_bootstrap(k, u["mu"], s["mu"], args.block_zones, args.boot, rng, spearman)
    out = {"zones": len(k), "det_uavsar": du.mean(), "det_s1": ds.mean(),
           "det_both": both.mean(), "det_both_indep": du.mean() * ds.mean(),
           "n_both": int(both.sum()),
           "sign_agree_both": (np.sign(u["mu"][both]) == np.sign(s["mu"][both])).mean()
           if both.any() else np.nan}
    for sensor, d in (("uavsar", u), ("s1", s)):
        out[f"c1_{sensor}_med"] = np.median(d["c1"])
        out[f"absmu_{sensor}_med_deg"] = np.degrees(np.median(np.abs(d["mu"])))
        out[f"h_{sensor}_med_deg"] = np.degrees(np.median(d["h"]))
    out["s1_full_circle"] = float((s["h"] >= np.pi - 1e-6).mean())
    out.update({"pearson_mu": pearson(u["mu"], s["mu"]), "pearson_lo": lo, "pearson_hi": hi,
                "spearman_mu": spearman(u["mu"], s["mu"]), "spearman_lo": slo,
                "spearman_hi": shi, **rel,
                "r_max": np.sqrt(rel["rel_uavsar"] * rel["rel_s1"]),
                "r_max_robust": np.sqrt(rel["rel_robust_uavsar"] * rel["rel_robust_s1"])})
    return out


def main(args):
    rng = np.random.default_rng(args.seed)
    U = Uavsar(args.uavsar_store, args.llh, args.uavsar_factor, args.zone_m)
    cslc = zarr.open_group(args.cslc_store, mode="r")
    trip = zarr.open_group(args.triplets_store, mode="r")
    period = [datetime.strptime(d, "%Y-%m-%d") for d in args.period]
    r, c = trip.attrs["window"]
    s1_looks = r * c
    run = start_run(args.task, cli.params(args), *cli.run_roots(args),
                    [args.uavsar_store, args.llh, args.cslc_store, args.triplets_store],
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    t0 = time.time()

    matches = []                           # (S1 group, track, UAVSAR t, S1 dates, dt days)
    for name in sorted(trip.group_keys()):
        track = name.split("_")[0]
        some = next(iter(trip[name].group_keys()))
        s1_dates = [datetime.strptime(f.split("_")[4][:15], "%Y%m%dT%H%M%S")
                    for f in trip[name][some].attrs["files"]]
        for t in range(len(U.dates) - 2):
            dt = [(s - u).total_seconds() / 86400 for s, u in zip(s1_dates, U.dates[t:t + 3])]
            if max(map(abs, dt)) <= args.max_dt:
                matches.append((name, track, t, trip[name][some].attrs["dates"], dt))
    for m in matches:
        logger.info("match %s <-> UAVSAR t=%d, dt %s d", m[0], m[2], np.round(m[4], 2))

    idx = [s for s in range(len(U.dates) - 1)
           if period[0] <= U.dates[s] and U.dates[s + 1] <= period[1]]
    lo = min(idx + [m[2] for m in matches])
    hi = max([s + 2 for s in idx] + [m[2] + 3 for m in matches])
    L = U.lags(lo, hi)
    logger.info("UAVSAR cells %s (N = %d), dates %d..%d loaded, %.1f min", U.shape, U.n_looks,
                lo, hi - 1, (time.time() - t0) / 60)
    S1C = S1Coherence(cslc, period, args.zone_m)
    rows, saved, s1_cache = [], {}, {}

    def s1_zone_results(name):
        if name not in s1_cache:
            keys, _, (phi, kap) = s1_triplet_zones(trip[name], args.zone_m, s1_looks)
            s1_cache[name] = (keys, *zone_test(phi, kap, args.min_cells, args.alpha,
                                               args.backend))
        return s1_cache[name]

    for name, track, t, ds, dt in matches:
        ref = next((m for m in matches if m[1] == "T035" and m[2] == t), None)
        s1_coh = S1C.zones(set(ds) | set(ref[3] if ref else []))
        s_keys, s_res, s_n = s1_zone_results(name)
        u_coh, n_pairs = U.selection_coherence(L, lo, (t, t + 1, t + 2), idx)
        phi, kap = U.triplet(t, L, lo)
        u_keys, _, (u_phi, u_kap, u_c) = group(U.keys, phi.ravel(), kap.ravel(), u_coh.ravel())
        u_res, u_n = zone_test(u_phi, u_kap, args.min_cells, args.alpha, args.backend)
        u_zcoh = np.nanmedian(u_c, -1)
        common, iu, is_ = np.intersect1d(u_keys, s_keys, return_indices=True)
        s_zcoh = np.array([s1_coh.get(k, np.nan) for k in common])
        keep = ((u_zcoh[iu] >= args.coh_uavsar) & (s_zcoh >= args.coh_s1)
                & np.isfinite(u_res["p"][iu]) & np.isfinite(s_res["p"][is_]))
        ku, ks, k = iu[keep], is_[keep], common[keep]
        u = {f: v[ku] for f, v in u_res.items()}
        s = {f: v[ks] for f, v in s_res.items()}
        row = {"s1_triplet": name, "track": track, "t": t,
               "uavsar_dates": " ".join(d.strftime("%Y-%m-%d") for d in U.dates[t:t + 3]),
               "dt_days": " ".join(f"{x:+.2f}" for x in dt), "overlap_zones": len(common),
               "uavsar_pairs_for_coherence": n_pairs}
        if len(k) > 10:
            row.update(compare(k, u, s, args, rng))
            ref_r = []                     # Sentinel-1 triplets of the track sharing no date
            for other, tr2, _, ds2, _ in matches:
                if tr2 != track or set(ds2) & set(ds) or other == name:
                    continue
                o_keys, o_res, _ = s1_zone_results(other)
                _, ju, jo = np.intersect1d(k, o_keys, return_indices=True)
                ok = np.isfinite(o_res["mu"][jo])
                if ok.sum() > 10:
                    ref_r.append(pearson(u["mu"][ju][ok], o_res["mu"][jo][ok]))
            row.update({"pearson_ref_mean": np.mean(ref_r) if ref_r else np.nan,
                        "pearson_ref_max_abs": np.max(np.abs(ref_r)) if ref_r else np.nan,
                        "n_ref": len(ref_r)})
        rows.append(row)
        saved[f"{name}_t{t}"] = {"keys": k, "coh_uavsar": u_zcoh[ku], "coh_s1": s_zcoh[keep],
                                 "n_uavsar": u_n[ku], "n_s1": s_n[ks],
                                 **{f"{f}_uavsar": v for f, v in u.items()},
                                 **{f"{f}_s1": v for f, v in s.items()}}
        logger.info("%s t=%d: %d overlap, %d common coherent zones, det U %.3f S1 %.3f, h U %.2f "
                    "S1 %.2f deg, r %.3f (r_max %.3f), %.1f min", name, t, len(common), len(k),
                    row.get("det_uavsar", np.nan), row.get("det_s1", np.nan),
                    row.get("h_uavsar_med_deg", np.nan), row.get("h_s1_med_deg", np.nan),
                    row.get("pearson_mu", np.nan), row.get("r_max", np.nan),
                    (time.time() - t0) / 60)

    summary = run.small_dir / "summary.csv"
    fields = list(dict.fromkeys(f for r_ in rows for f in r_))
    with open(summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(run.small_dir / "zones.npz",
                        **{f"{n}__{f}": v for n, d in saved.items() for f, v in d.items()})
    finish_run(run, {"summary": summary, "zones": run.small_dir / "zones.npz"},
               extra={"elapsed_min": (time.time() - t0) / 60})
    logger.info("done: %s", run.small_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uavsar-store", type=Path, required=True)
    parser.add_argument("--llh", type=Path, required=True)
    parser.add_argument("--cslc-store", type=Path, required=True)
    parser.add_argument("--triplets-store", type=Path, required=True)
    parser.add_argument("--uavsar-factor", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--zone-m", type=float, default=100.0)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--coh-uavsar", type=float, default=0.5)
    parser.add_argument("--coh-s1", type=float, default=0.5)
    parser.add_argument("--period", nargs=2, default=["2016-07-01", "2017-11-30"])
    parser.add_argument("--max-dt", type=float, default=6.0)
    parser.add_argument("--pfa", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.05, help="Level of the confidence arcs.")
    parser.add_argument("--block-zones", type=int, default=10)
    parser.add_argument("--boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="numpy")
    cli.add_run_arguments(parser, task="cross-sensor")
    args = parser.parse_args()
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
