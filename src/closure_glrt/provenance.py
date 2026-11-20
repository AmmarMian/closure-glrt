r"""Provenance of runs that produce data files (NAS processing).

A run has an id ``<UTC YYYYmmddTHHMM>_<task>_<short commit>`` and two
directories with that name: ``<project_root>/runs/<id>`` (manifest, log,
small outputs) and ``<cold_root>/runs/<id>`` (large outputs). The project
ledger ``<project_root>/ledger.jsonl`` gets one line when a run starts and one
when it ends; it is append-only.

``manifest.json`` records the git commit (a run refuses to start from a dirty
tree unless ``allow_dirty``), the command, parameters, host, package versions,
the inputs (path, size, mtime and, when the run computed it, SHA-256) and the
outputs (path and SHA-256; a directory is hashed as the SHA-256 of its sorted
``relative path  file SHA-256`` lines). Once finished, both run directories
are made read-only: a redo is a new run.

Example:
    run = start_run("scan-sdelta", vars(args), project_root, cold_root, inputs)
    ...  # write into run.small_dir / run.large_dir
    finish_run(run, {"scan": run.large_dir / "scan.zarr"}, input_sha256=hashes)
"""

import getpass
import hashlib
import json
import os
import platform
import resource
import socket
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGES = ("numpy", "scipy", "zarr", "torch", "closure-glrt")


@dataclass
class Run:
    run_id: str
    small_dir: Path
    large_dir: Path
    ledger: Path
    manifest: dict = field(default_factory=dict)

    @property
    def manifest_path(self) -> Path:
        return self.small_dir / "manifest.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def git_state() -> dict:
    """Commit, branch and dirty flag of the repository holding this code."""
    status = _git("status", "--porcelain")
    return {"commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status), "status": status.splitlines()}


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def sha256_file(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def sha256_path(path: Path) -> str:
    """SHA-256 of a file, or of a directory as its sorted ``relpath  sha256`` lines."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    lines = [f"{p.relative_to(path).as_posix()}  {sha256_file(p)}"
             for p in sorted(path.rglob("*")) if p.is_file()]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _append(ledger: Path, entry: dict) -> None:
    with open(ledger, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _write(run: Run) -> None:
    run.manifest_path.write_text(json.dumps(run.manifest, indent=2, default=str))


def start_run(task: str, params: dict, project_root: Path, cold_root: Path,
              inputs: list[Path], allow_dirty: bool = False) -> Run:
    """Create the run directories, write the initial manifest and the ledger start line."""
    git = git_state()
    if git["dirty"] and not allow_dirty:
        raise RuntimeError(f"git tree of {REPO} is dirty; commit first:\n"
                           + "\n".join(git["status"]))
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M}_{task}_{git['commit'][:7]}"
    project_root, cold_root = Path(project_root), Path(cold_root)
    run = Run(run_id, project_root / "runs" / run_id, cold_root / "runs" / run_id,
              project_root / "ledger.jsonl")
    run.small_dir.mkdir(parents=True)
    run.large_dir.mkdir(parents=True)
    run.manifest = {
        "run_id": run_id, "task": task, "status": "running", "started": _now(), "finished": None,
        "host": socket.gethostname(), "user": getpass.getuser(), "platform": platform.platform(),
        "cwd": os.getcwd(), "command": sys.argv, "params": params, "git": git,
        "versions": _versions(), "small_dir": str(run.small_dir), "large_dir": str(run.large_dir),
        "inputs": [{"path": str(p), "size": p.stat().st_size, "mtime": p.stat().st_mtime}
                   for p in map(Path, inputs)],
        "outputs": {},
    }
    _write(run)
    _append(run.ledger, {"time": _now(), "run_id": run_id, "task": task, "event": "start",
                         "commit": git["commit"], "dirty": git["dirty"]})
    return run


def _make_read_only(root: Path) -> None:
    mask = ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    for p in sorted(root.rglob("*"), reverse=True) + [root]:
        p.chmod(p.stat().st_mode & mask)


def finish_run(run: Run, outputs: dict[str, Path], input_sha256: dict[str, str] | None = None,
               status: str = "completed", extra: dict | None = None) -> None:
    """Hash outputs, complete the manifest, append the ledger end line, make the run read-only."""
    run.manifest["outputs"] = {name: {"path": str(p), "sha256": sha256_path(p)}
                               for name, p in outputs.items()}
    if input_sha256:
        for item in run.manifest["inputs"]:
            item["sha256"] = input_sha256.get(item["path"])
    run.manifest.update(status=status, finished=_now(),
                        peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    if extra:
        run.manifest.update(extra)
    _write(run)
    _append(run.ledger, {"time": _now(), "run_id": run.run_id, "task": run.manifest["task"],
                         "event": "end", "status": status,
                         "manifest_sha256": sha256_file(run.manifest_path)})
    _make_read_only(run.small_dir)
    _make_read_only(run.large_dir)
