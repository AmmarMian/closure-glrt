"""Zone tiling and cell aggregation, file readers (UAVSAR annotations, OPERA names), provenance."""
import json
import stat
import subprocess

import numpy as np
import pytest
from conftest import ALL_BACKEND_PARAMS

from closure_glrt import provenance
from closure_glrt.backend import get_data_on_device, to_numpy
from closure_glrt.closure import aggregate_cells, tile_zones
from closure_glrt.io.uavsar import parse_ann


def _grid(I=7, J=11, K=3):
    """Cell grid whose value encodes its position: x[i, j, k] = 1000 i + 10 j + k."""
    i, j, k = np.meshgrid(np.arange(I), np.arange(J), np.arange(K), indexing="ij")
    return (1000 * i + 10 * j + k).astype(np.float64)


@pytest.mark.parametrize("backend_name", ALL_BACKEND_PARAMS)
def test_tile_zones_row_major_and_borders(backend_name):
    x = _grid()
    z = to_numpy(tile_zones(get_data_on_device(x, backend_name), (3, 5), backend_name))
    assert z.shape == (2, 2, 15, 3)                       # 7 // 3, 11 // 5; borders dropped
    for zi in range(2):
        for zj in range(2):
            for c in range(15):                           # cell c = (a, b), a = c // 5, b = c % 5
                a, b = divmod(c, 5)
                assert np.array_equal(z[zi, zj, c], x[3 * zi + a, 5 * zj + b])


def test_aggregate_cells_is_block_mean():
    x = _grid() + 1j * _grid()
    y = aggregate_cells(x, (2, 3))
    assert y.shape == (3, 3, 3)
    assert np.allclose(y[1, 2], x[2:4, 6:9].mean(axis=(0, 1)))
    assert np.allclose(aggregate_cells(x, (1, 1)), x)


def test_parse_ann(tmp_path):
    p = tmp_path / "a.ann"
    p.write_text(
        "; comment line\n"
        "slc_1_1x1 Rows                    (pixels)   = 66664      ; rows of segment 1\n"
        "Start Time of Acquisition         (&)        = 18-Jul-2009 00:47:33 UTC\n"
        "Peg Heading                       (deg)      = -88.9\n"
        "line without units = ignored\n")
    a = parse_ann(p)
    assert a["slc_1_1x1 Rows"] == "66664"
    assert a["Start Time of Acquisition"] == "18-Jul-2009 00:47:33 UTC"
    assert a["Peg Heading"] == "-88.9"
    assert "line without units" not in a


def test_opera_list_bursts(tmp_path):
    pytest.importorskip("h5py")
    from closure_glrt.io.opera import list_bursts

    d = tmp_path / "T042"
    d.mkdir()
    names = ["OPERA_L2_CSLC-S1_T042-088914-IW3_20170606T140742Z_20240428T183619Z_S1A_VV_v1.1.h5",
             "OPERA_L2_CSLC-S1_T042-088914-IW3_20170302T140737Z_20240427T100000Z_S1B_VV_v1.1.h5",
             "OPERA_L2_CSLC-S1_T042-088915-IW3_20170302T140740Z_20240427T100000Z_S1A_VV_v1.1.h5",
             "OPERA_L2_CSLC-S1_T042-088914-IW3_20170302T140737Z_static_layers.h5"]   # not a CSLC
    for n in names:
        (d / n).touch()
    b = list_bursts(tmp_path)
    assert sorted(b) == ["T042-088914-IW3", "T042-088915-IW3"]
    assert [a["date"] for a in b["T042-088914-IW3"]] == ["20170302", "20170606"]   # time order
    assert b["T042-088914-IW3"][0]["platform"] == "S1B"
    (d / names[0].replace("20240428T183619Z", "20250101T000000Z")).touch()        # 2 files, 1 date
    with pytest.raises(ValueError, match="several files"):
        list_bursts(tmp_path)


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """A fresh git repository standing in for the code repository."""
    repo = tmp_path / "code"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True,  # noqa: E731
                                    capture_output=True)
    run("init", "-q")
    (repo / "f.txt").write_text("v1\n")
    run("add", "f.txt")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c1")
    monkeypatch.setattr(provenance, "REPO", repo)
    return repo


def test_run_manifest_ledger_and_read_only(git_repo, tmp_path):
    root = tmp_path / "runs"
    inp = tmp_path / "input.bin"
    inp.write_bytes(b"abc")
    run = provenance.start_run("demo", {"x": 1}, root, root / "large", [inp])
    commit = subprocess.run(["git", "-C", str(git_repo), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    assert run.run_id.endswith(f"_demo_{commit[:7]}")
    out = run.small_dir / "result.csv"
    out.write_text("a,b\n1,2\n")
    provenance.finish_run(run, {"result": out}, input_sha256={str(inp): "h"})

    m = json.loads(run.manifest_path.read_text())
    assert m["status"] == "completed" and m["git"]["commit"] == commit and not m["git"]["dirty"]
    assert m["params"] == {"x": 1} and m["inputs"][0]["sha256"] == "h"
    assert m["outputs"]["result"]["sha256"] == provenance.sha256_file(out)
    ledger = (root / "ledger.jsonl").read_text().splitlines()
    events = [json.loads(line)["event"] for line in ledger]
    assert events == ["start", "end"]
    assert not out.stat().st_mode & stat.S_IWUSR                 # finished runs are read-only


def test_run_refuses_dirty_tree(git_repo, tmp_path):
    (git_repo / "f.txt").write_text("v2\n")
    root = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="dirty"):
        provenance.start_run("demo", {}, root, root / "large", [])
    run = provenance.start_run("demo", {}, root, root / "large", [], allow_dirty=True)
    assert json.loads(run.manifest_path.read_text())["git"]["dirty"]


def test_sha256_path_of_directory_ignores_order_and_chunking(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d, order in ((a, ("x", "y")), (b, ("y", "x"))):
        (d / "sub").mkdir(parents=True)
        for n in order:
            (d / "sub" / n).write_text(n)
    assert provenance.sha256_path(a) == provenance.sha256_path(b)
    (b / "sub" / "x").write_text("changed")
    assert provenance.sha256_path(a) != provenance.sha256_path(b)
