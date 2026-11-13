r"""Monte Carlo look-up table of the closure standard deviation of Yuan and Lopez-Dekker (2026).

$\sigma_n$ of the closure with 2048 independent circular Gaussian looks, ``--reps`` repetitions,
on a grid of the three coherences of step ``--step`` (:func:`closure_glrt.competitors.yuan_lut`).

Output (run directory): ``yuan_lut.npz`` with ``grid`` and ``table``.

Example:
    uv run python experiments/stores/yuan_lut.py --task yuan-lut
"""

import argparse
import logging
import time

import numpy as np

from closure_glrt import cli
from closure_glrt.competitors import yuan_lut
from closure_glrt.provenance import finish_run, start_run

logger = logging.getLogger(__name__)


def main(args):
    run = start_run(args.task, cli.params(args), *cli.run_roots(args), [],
                    allow_dirty=args.allow_dirty)
    logging.getLogger().addHandler(logging.FileHandler(run.small_dir / "log.txt"))
    t0 = time.time()
    grid, table = yuan_lut(np.random.default_rng(args.seed), args.step, args.reps)
    path = run.small_dir / "yuan_lut.npz"
    np.savez(path, grid=grid, table=table)
    finish_run(run, {"lut": path}, extra={"elapsed_min": (time.time() - t0) / 60})
    logger.info("done: %s (%d valid nodes)", path, int(np.isfinite(table).sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--reps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    cli.add_run_arguments(parser, task="yuan-lut")
    args = parser.parse_args()
    cli.setup_logging(args.debug)
    cli.log_arguments(args)
    main(args)
