r"""UAVSAR co-registered SLC stacks: annotation files, file listing, strip scan.

A stack directory holds, per acquisition and polarisation, a flat
little-endian complex64 SLC ``<line>_<flight>_<date>_L090<pol>_02_BC_s<seg>_1x1.slc``
(row-major, ``Rows x Columns`` read from the ``.ann``) and its ``.ann``.
Acquisitions sorted by file name are chronological. ``parse_ann`` is adapted
from ``polsar_view.readers.uavsar`` (sar_reader).

:class:`StripReader` reads all acquisitions of one polarisation strip by
strip (each file sequentially, so its SHA-256 is computed on the way);
:func:`strip_statistics` gives, per strip, the lagged cell covariances of
:func:`closure_glrt.closure.block_lag_covariance` and the per-pixel amplitude
mean and dispersion over time.
"""

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from closure_glrt.closure import block_lag_covariance

_ANN_PATTERN = re.compile(r"^([^;=\n]+?)\s*\([^)]*\)\s*=\s*([^;]+?)(?:\s*;.*)?$")
_SLC_RE = re.compile(r"^(?P<line>\w+?_\d{5})_(?P<flight>\d{5}_\d{3})_(?P<date>\d{6})_L090"
                     r"(?P<pol>HH|HV|VH|VV)_(?P<ver>\d{2})_BC_s(?P<seg>\d)_1x1\.slc$")


def parse_ann(ann_path: Path) -> dict[str, str]:
    """Parse a UAVSAR .ann file into ``{parameter name: raw value}`` (comments removed)."""
    params: dict[str, str] = {}
    with open(ann_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith(";") or not line:
                continue
            m = _ANN_PATTERN.match(line)
            if m:
                params[m.group(1).strip()] = m.group(2).strip()
    return params


@dataclass(frozen=True)
class Acquisition:
    flight: str            # e.g. "17020_004"
    date: str              # YYMMDD
    path: Path             # SLC file of the requested polarisation and segment


def list_acquisitions(directory: Path, pol: str, segment: int = 1) -> list[Acquisition]:
    """Acquisitions of one polarisation and segment, in chronological (file name) order."""
    out = []
    for p in sorted(Path(directory).glob(f"*_L090{pol}_*_s{segment}_1x1.slc")):
        m = _SLC_RE.match(p.name)
        if m:
            out.append(Acquisition(m["flight"], m["date"], p))
    return out


def slc_shape(slc_path: Path, segment: int = 1) -> tuple[int, int]:
    """(rows, columns) of a 1x1 SLC from its .ann, checked against the file size."""
    ann = Path(str(slc_path).replace(f"_s{segment}_1x1.slc", ".ann"))
    params = parse_ann(ann)
    rows = int(params[f"slc_{segment}_1x1 Rows"])
    cols = int(params[f"slc_{segment}_1x1 Columns"])
    if Path(slc_path).stat().st_size != rows * cols * 8:
        raise ValueError(f"{slc_path}: size {Path(slc_path).stat().st_size} != {rows}x{cols}x8")
    return rows, cols


class StripReader:
    """Iterate over ``(row0, X)``, ``X`` of shape (T, rows, W) complex64, strip by strip.

    Every file is read sequentially to its end (the last strip may be shorter),
    so after a full iteration ``hashes`` holds the SHA-256 of each whole file.
    The next strip is read while the caller processes the current one.
    """

    def __init__(self, paths: list[Path], shape: tuple[int, int], strip_rows: int,
                 n_threads: int = 4):
        self.paths, self.shape, self.strip_rows, self.n_threads = list(paths), shape, strip_rows, n_threads
        self.hashes: dict[str, str] = {}

    def __iter__(self):
        H, W = self.shape
        T = len(self.paths)
        hashes = [hashlib.sha256() for _ in self.paths]
        files = [open(p, "rb") for p in self.paths]

        def read_one(t, buf):
            n = files[t].readinto(memoryview(buf).cast("B"))
            if n != buf.nbytes:
                raise IOError(f"short read in {self.paths[t]}: {n} of {buf.nbytes} bytes")
            hashes[t].update(memoryview(buf).cast("B"))

        def read_strip(row0):
            X = np.empty((T, min(self.strip_rows, H - row0), W), dtype="<c8")
            list(pool.map(lambda t: read_one(t, X[t]), range(T)))
            return row0, X

        try:
            with ThreadPoolExecutor(self.n_threads) as pool, ThreadPoolExecutor(1) as prefetch:
                starts = list(range(0, H, self.strip_rows))
                future = prefetch.submit(read_strip, starts[0])
                for i in range(len(starts)):
                    row0, X = future.result()
                    if i + 1 < len(starts):
                        future = prefetch.submit(read_strip, starts[i + 1])
                    yield row0, X
            for f in files:
                if f.read(1):
                    raise IOError(f"{f.name} is longer than {H}x{W} complex64")
            self.hashes = {str(p): h.hexdigest() for p, h in zip(self.paths, hashes)}
        finally:
            for f in files:
                f.close()


def strip_statistics(X: np.ndarray, window: tuple[int, int], max_lag: int, n_threads: int = 8):
    """Lagged cell covariances and per-pixel amplitude statistics of one strip.

    Parameters
    ----------
    X : np.ndarray of shape (T, R, W), complex64
    window : (r, c) cell size; only the ``R // r`` full cell rows are used.

    Returns
    -------
    lags : (R // r, W // c, T, max_lag + 1) complex64
    amp_mean, amp_disp : (R, W) float32, mean of |x| over time and std(|x|) / mean(|x|)
    """
    T, R, W = X.shape
    r, c = window
    Wm = W // c
    bounds = np.linspace(0, Wm, n_threads + 1).astype(int) * c
    stack = np.moveaxis(X, 0, -1)                                   # (R, W, T) view

    def lag_block(j):
        a, b = bounds[j], bounds[j + 1]
        return block_lag_covariance(stack[:, a:b], window, max_lag).astype(np.complex64)

    def amp_block(j):
        a, b = j * W // n_threads, (j + 1) * W // n_threads
        A = np.abs(X[:, :, a:b])
        m = A.mean(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return m, (A.std(axis=0) / m).astype(np.float32)

    with ThreadPoolExecutor(n_threads) as pool:
        lags = np.concatenate(list(pool.map(lag_block, range(n_threads))), axis=1)
        amps = list(pool.map(amp_block, range(n_threads)))
    return lags, np.concatenate([a[0] for a in amps], axis=1), np.concatenate([a[1] for a in amps], axis=1)
