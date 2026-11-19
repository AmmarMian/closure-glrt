"""Data tables of the figures for pgfplots.

Each figure script draws its PDF with matplotlib and, with ``--data DIR``, also writes the arrays
it plots as whitespace-separated tables ``DIR/<name>.dat`` (one header line with the column
names, ``nan`` for missing values). The paper draws the figure with pgfplots from these tables
(``\\addplot table[x=..., y=...] {figures/data/<figure>/<name>.dat}``), so that rerunning the
script on new runs updates the paper figure.
"""

from pathlib import Path

import numpy as np


def write_dat(folder, name, **columns):
    """Write ``folder/name.dat`` with one column per keyword (1-D arrays of equal length)."""
    if folder is None:
        return None
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cols = {k: np.asarray(v).ravel() for k, v in columns.items()}
    n = {len(v) for v in cols.values()}
    if len(n) != 1:
        raise ValueError(f"{name}: columns of different lengths {n}")
    path = folder / f"{name}.dat"
    with open(path, "w") as f:
        f.write(" ".join(cols) + "\n")
        for row in zip(*cols.values()):
            f.write(" ".join(_fmt(v) for v in row) + "\n")
    return path


def _fmt(v):
    if isinstance(v, (str, np.str_)):
        return "{" + str(v) + "}" if " " in str(v) else str(v)
    v = float(v)
    return "nan" if not np.isfinite(v) else f"{v:.6g}"


def write_macros(folder, name, **macros):
    """Write ``folder/name.tex`` defining one LaTeX macro per keyword (text computed from the
    data, e.g. dates of a triplet in a title): ``\\def\\<key>{<value>}``."""
    if folder is None:
        return None
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.tex"
    with open(path, "w") as f:
        f.write("% written by the figure script, do not edit\n")
        for k, v in macros.items():
            if not k.isalpha():
                raise ValueError(f"macro name {k!r} must be letters only")
            f.write(f"\\def\\{k}{{{v}}}\n")
    return path
