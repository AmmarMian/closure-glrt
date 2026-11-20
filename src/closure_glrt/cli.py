"""Shared command-line plumbing of the experiments: logging, default paths, provenance roots.

Paths come from the environment, with defaults inside the repository:

* ``DATA_ROOT`` (default ``<repo>/data/raw``): raw data, layout of the README;
* ``RUNS_ROOT`` (default ``<repo>/runs``): run directories ``runs/<run_id>`` (manifest, log,
  small outputs) and ``runs/large/runs/<run_id>`` (stores), see :mod:`closure_glrt.provenance`.
"""

import argparse
import logging
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", REPO / "data" / "raw"))


def runs_root() -> Path:
    return Path(os.environ.get("RUNS_ROOT", REPO / "runs"))


def uavsar_dir() -> Path:
    return data_root() / "uavsar" / "SDelta_23518"


def cslc_dir() -> Path:
    return data_root() / "sentinel1" / "OPERA_CSLC"


def setup_logging(debug: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        force=True)


def log_arguments(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    width = max((len(k) for k in vars(args)), default=0)
    logger.info("Configuration:")
    for k, v in vars(args).items():
        logger.info("  %s = %s", k.ljust(width), v)


def add_run_arguments(parser: argparse.ArgumentParser, task: str) -> None:
    """--task, --runs-root, --allow-dirty, --debug (common to every experiment)."""
    parser.add_argument("--task", default=task)
    parser.add_argument("--runs-root", type=Path, default=None,
                        help="Default: $RUNS_ROOT or <repo>/runs.")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Run from a git tree with uncommitted changes "
                             "(recorded in the manifest).")
    parser.add_argument("--debug", action="store_true")


def run_roots(args) -> tuple[Path, Path]:
    """(project_root, cold_root) for :func:`closure_glrt.provenance.start_run`."""
    root = args.runs_root or runs_root()
    return root, root / "large"


def params(args) -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
