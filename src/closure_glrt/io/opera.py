"""Reader for OPERA CSLC-S1 burst files (HDF5): listing, grid, row strips."""

import re
from pathlib import Path

import h5py
import numpy as np

_CSLC_RE = re.compile(r"OPERA_L2_CSLC-S1_(T\d{3}-\d{6}-IW\d)_(\d{8})T(\d{6})Z_\d{8}T\d{6}Z_(S1[ABC])_VV_v[\d.]+\.h5$")


def list_bursts(cslc_dir: Path) -> dict[str, list[dict]]:
    """Burst id -> acquisitions (date, time, platform, path) in time order; one file per date."""
    bursts: dict[str, list[dict]] = {}
    for p in sorted(Path(cslc_dir).glob("T*/*.h5")):
        m = _CSLC_RE.match(p.name)
        if m:
            burst, date, hms, platform = m.groups()
            bursts.setdefault(burst, []).append({"date": date, "time": hms, "platform": platform, "path": p})
    for burst, acq in bursts.items():
        acq.sort(key=lambda a: (a["date"], a["time"]))
        dates = [a["date"] for a in acq]
        if len(set(dates)) != len(dates):
            raise ValueError(f"{burst}: several files for one date")
    return bursts


def grid(path: Path) -> dict:
    with h5py.File(path, "r") as h:
        d = h["data"]
        return {"shape": list(d["VV"].shape), "epsg": int(d["projection"][()]),
                "x0": float(d["x_coordinates"][0]), "y0": float(d["y_coordinates"][0]),
                "dx": float(d["x_spacing"][()]), "dy": float(d["y_spacing"][()])}


def read_rows(path: str, row0: int, row1: int) -> np.ndarray:
    with h5py.File(path, "r") as h:
        return h["data/VV"][row0:row1].astype(np.complex64)
