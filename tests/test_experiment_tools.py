"""Tooling of the experiments: figure registry, latest-run lookup, ancillary inputs, map grid."""
import importlib.util
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(rel):
    path = REPO / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REGISTRY = """
[roots]
runs = "supplement/runs"
abs = "/data/store"

[[figure]]
name = "fit"
label = "fig:fit"
title = "Fit"
script = "fig_fit.py"
args = ["--run", "{runs}/r1", "--pair", "s1={abs}/x", "--k", "1"]

[[figure]]
name = "thumbs"
label = "fig:simu"
tex = ["simu_diagram.tex"]
title = "Thumbnails"
script = "fig_simu_thumbs.py"
args = ["--spectrum", "{abs}/spectrum.npz"]
pdf_args = ["--out-dir", "{figures}"]
data = false
host = "server"
"""


def test_make_figures_commands_and_table(tmp_path):
    mf = _load("experiments/figures/make_figures.py")
    paper = tmp_path / "paper"
    (paper / "figures").mkdir(parents=True)
    reg = paper / "figures" / "figures.toml"
    reg.write_text(REGISTRY)
    figs, roots = mf.load(reg, ["abs=/other"])                       # --root override
    out = paper / "figures"
    cmd, args = mf.command(figs[0], roots, paper, out)
    run = str((paper / "supplement/runs").resolve())
    assert args == ["--run", f"{run}/r1", "--pair", "s1=/other/x", "--k", "1"]
    assert cmd[1:] == ["fig_fit.py", *args, "--out", str(out / "pdf" / "fit.pdf"),
                       "--data", str(out / "data" / "fit")]
    assert mf.inputs(args) == [Path(f"{run}/r1"), Path("/other/x")]  # what must exist locally
    cmd, _ = mf.command(figs[1], roots, paper, out)                  # pdf_args, no pgfplots data
    assert cmd[-2:] == ["--out-dir", str(out)] and "--data" not in cmd
    with pytest.raises(KeyError, match="unknown root"):
        mf.resolve("{nope}/x", roots, paper)
    mf.table(figs, reg.with_name("README.md"))
    text = reg.with_name("README.md").read_text()
    assert "`fig:fit` (`fit.tex`)" in text and "| server |" in text and "| local |" in text


def test_latest_run(tmp_path):
    script = REPO / "scripts" / "latest_run.sh"
    for d in ("runs/20261101T1000_zone-scan-s1_aaaaaaa", "runs/20261102T1000_zone-scan-s1_bbbbbbb",
              "runs/20261103T1000_zone-scan-s10_ccccccc",               # other task, same prefix
              "large/runs/20261101T0900_uavsar-scan-s1_ddddddd"):
        (tmp_path / d).mkdir(parents=True)
    env = {**os.environ, "RUNS_ROOT": str(tmp_path)}
    get = lambda *a: subprocess.run([str(script), *a], env=env, capture_output=True,  # noqa: E731
                                    text=True)
    assert get("zone-scan-s1").stdout.strip().endswith("20261102T1000_zone-scan-s1_bbbbbbb")
    assert get("uavsar-scan-s1", "large").stdout.strip().endswith("_uavsar-scan-s1_ddddddd")
    r = get("uavsar-scan-s1")                                           # not among small runs
    assert r.returncode != 0 and "run its recipe first" in r.stderr


def test_acquisition_times_from_annotations(tmp_path):
    fa = _load("experiments/ancillary/fetch_ancillary.py")
    ann = "Start Time of Acquisition         (&)        = {} UTC\n"
    files = {"SDelta_23518_11067_002_110829_L090HH_02_BC.ann": "29-Aug-2011 21:10:00",
             "SDelta_23518_11066_001_110829_L090HH_02_BC.ann": "29-Aug-2011 21:40:05",
             "SDelta_23518_09048_004_090718_L090HH_02_BC.ann": " 8-Jul-2009 00:47:33",
             "SDelta_23518_09048_004_090718_L090VV_02_BC.ann": "01-Jan-2000 00:00:00"}  # not HH
    for name, t in files.items():
        (tmp_path / name).write_text(ann.format(t))
    times = fa.uavsar_times(Namespace(ann_dir=tmp_path))
    assert [n[13:28] for n, _ in times] == ["09048_004_09071", "11067_002_11082",
                                            "11066_001_11082"]   # time order, not file order
    assert times[0][1].isoformat() == "2009-07-08T00:47:33+00:00"
    with pytest.raises(SystemExit, match="download uavsar ann"):
        fa.uavsar_times(Namespace(ann_dir=tmp_path / "empty"))


def test_context_map_grid():
    pytest.importorskip("pyproj")
    cm = _load("experiments/ancillary/context_maps.py")
    from pyproj import Transformer

    # zone centres 10 m off a 100 m UTM 10N lattice (off the pixel lattice: no rounding ties),
    # one of them missing (nan)
    e, n = np.meshgrid(600_000 + 100 * np.arange(5), 4_200_000 + 100 * np.arange(4))
    lon, lat = Transformer.from_crs(32610, 4326, always_xy=True).transform(e.ravel() + 10.0,
                                                                          n.ravel() + 10.0)
    lon, lat = np.array(lon), np.array(lat)
    lon[7] = np.nan
    epsg, x0, y0, index = cm.map_grid(lon, lat, res=50.0, max_dist=20.0)
    assert epsg == 32610 and (x0, y0) == (600_000.0, 4_200_350.0)       # top-left pixel centre
    assert index.shape == (8, 10)
    for k in range(20):                                # each zone in the pixel 14 m away
        r, c = (y0 - n.ravel()[k]) // 50, (e.ravel()[k] - x0) // 50
        assert index[int(r), int(c)] == (-1 if k == 7 else k)
    assert (index >= 0).sum() == 19                    # every other pixel is >= 36 m away: -1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
