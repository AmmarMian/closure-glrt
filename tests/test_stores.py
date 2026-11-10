"""Smoke test of the store builders on a tiny synthetic UAVSAR stack."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import zarr

from closure_glrt.closure import block_lag_covariance

REPO = Path(__file__).resolve().parents[1]


def _fake_stack(d: Path, T=5, H=60, W=40, seed=0):
    rng = np.random.default_rng(seed)
    stack = np.empty((H, W, T), np.complex64)
    for t in range(T):
        flight, date = f"1000{t}_00{t}", f"1007{10 + t:02d}"
        base = f"SDelta_23518_{flight}_{date}_L090VV_02_BC"
        (d / f"{base}.ann").write_text(
            f"slc_1_1x1 Rows (pixels) = {H}\nslc_1_1x1 Columns (pixels) = {W}\n")
        x = (rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W))).astype("<c8")
        x.tofile(d / f"{base}_s1_1x1.slc")
        stack[..., t] = x
    return stack


def test_uavsar_scan_store(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    stack = _fake_stack(data)
    runs = tmp_path / "runs"
    cmd = [sys.executable, str(REPO / "experiments/stores/uavsar_scan.py"), "--stack-dir", str(data),
           "--strip-rows", "30", "--threads", "2", "--io-threads", "2", "--runs-root", str(runs),
           "--allow-dirty"]
    subprocess.run(cmd, check=True, capture_output=True)
    (run_dir,) = (runs / "runs").iterdir()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["params"]["pols"] == ["VV"]
    store = zarr.open_group(manifest["outputs"]["scan"]["path"], mode="r")
    lags = store["VV"]["lags"][:]
    ref = block_lag_covariance(stack, (15, 5), 2, sum_dtype=np.complex128).astype(np.complex64)
    assert np.array_equal(lags, ref, equal_nan=True)
    assert len(store.attrs["acquisitions"]) == 5
