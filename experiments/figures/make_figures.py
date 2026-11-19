r"""Regenerate the figures of a paper from its registry (``figures/figures.toml`` of the paper).

For each figure of the registry, run its script (in this folder) with its arguments, the PDF
output (``--out <paper>/figures/pdf/<name>.pdf`` unless ``pdf_args`` is given) and the data of
the pgfplots version (``--data <paper>/figures/data/<name>``). A figure whose inputs are missing
on this machine (e.g. ``host = "server"`` inputs) is skipped and reported; run the same command
on the host that has them, with ``--out-dir``, and copy the outputs back.

With ``--table``, write the list of the figures (``figures/README.md``: figure, script,
arguments, outputs) from the registry instead of running anything.

Examples:
    uv run --group figures python experiments/figures/make_figures.py \
        --registry ../paper_grsl/figures/figures.toml
    uv run --group figures python experiments/figures/make_figures.py \
        --registry ../paper_grsl/figures/figures.toml --only fit series
    python experiments/figures/make_figures.py --registry figures.toml --paper <paper dir> \
        --only xsensor --out-dir <dir>    # on the host with the inputs, then copy back
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:                 # Python < 3.11
    import tomli as tomllib

HERE = Path(__file__).resolve().parent


def load(registry, overrides):
    reg = tomllib.loads(Path(registry).read_text())
    roots = dict(reg.get("roots", {}))
    roots.update(dict(o.split("=", 1) for o in overrides))
    return reg["figure"], roots


def resolve(arg, roots, paper):
    def sub(m):
        key = m.group(1)
        if key not in roots:
            raise KeyError(f"unknown root {{{key}}} in {arg!r}")
        p = Path(roots[key])
        return str(p if p.is_absolute() else (paper / p).resolve())
    return re.sub(r"\{(\w+)\}", sub, arg)


def inputs(args):
    """Paths among the resolved arguments (values of SEG=path too)."""
    out = []
    for a in args:
        v = a.split("=", 1)[1] if "=" in a and not a.startswith("-") else a
        if v.startswith("/"):
            out.append(Path(v))
    return out


def command(fig, roots, paper, out_dir):
    pdf = out_dir / "pdf"
    data = out_dir / "data" / fig["name"]
    args = [resolve(a, roots, paper) for a in fig.get("args", [])]
    if "pdf_args" in fig:
        outs = [a.replace("{pdf}", str(pdf)) for a in fig["pdf_args"]]
    else:
        outs = ["--out", str(pdf / f"{fig['name']}.pdf")]
    return [sys.executable, fig["script"], *args, *outs, "--data", str(data)], args


def table(figs, path):
    lines = ["# Figures of the paper",
             "",
             "Written by `closure_glrt/experiments/figures/make_figures.py --table` from "
             "`figures.toml`; do not edit by hand.",
             "Each script (in `closure_glrt/experiments/figures/`) writes the PDF "
             "(`figures/pdf/`) and the data of the pgfplots figure (`figures/data/<name>/`), "
             "which the paper inputs as `figures/<name>.tex`.",
             "",
             "| Figure | Content | Script | Arguments | Host |",
             "|---|---|---|---|---|"]
    for f in figs:
        args = " ".join(f"`{a}`" if a.startswith("-") else a for a in f.get("args", []))
        tex = ", ".join(f"`{t}`" for t in f.get("tex", [f"{f['name']}.tex"]))
        lines.append(f"| `{f['label']}` ({tex}) | {f['title']} | `{f['script']}` "
                     f"| {args} | {f.get('host', 'local')} |")
    Path(path).write_text("\n".join(lines) + "\n")
    print("wrote", path)


def main(a):
    paper = Path(a.paper).resolve() if a.paper else Path(a.registry).resolve().parent.parent
    figs, roots = load(a.registry, a.root)
    if a.table:
        table(figs, Path(a.registry).with_name("README.md"))
        return
    out_dir = Path(a.out_dir).resolve() if a.out_dir else paper / "figures"
    (out_dir / "pdf").mkdir(parents=True, exist_ok=True)
    done, skipped, failed = [], [], []
    for fig in figs:
        if a.only and fig["name"] not in a.only:
            continue
        cmd, args = command(fig, roots, paper, out_dir)
        missing = [p for p in inputs(args) if not p.exists()]
        if missing:
            skipped.append((fig["name"], missing[0]))
            continue
        print("==", fig["name"], ":", " ".join(cmd[1:]), flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        (done if r.returncode == 0 else failed).append(fig["name"])
    print("\ndone:", ", ".join(done) or "-")
    for name, p in skipped:
        print(f"skipped {name}: missing input {p}")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", required=True, help="figures.toml of the paper")
    p.add_argument("--paper", default=None,
                   help="Paper directory (default: parent of the registry's folder).")
    p.add_argument("--only", nargs="+", default=None, help="Names of the figures to run.")
    p.add_argument("--root", nargs="+", default=[], help="Override a root: NAME=path.")
    p.add_argument("--out-dir", default=None,
                   help="Write pdf/ and data/ here instead of <paper>/figures.")
    p.add_argument("--table", action="store_true", help="Write figures/README.md and stop.")
    main(p.parse_args())
