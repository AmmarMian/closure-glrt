r"""Fig. 3 of the paper (one column): fit of the closure model on real zones.

(a) Precision profile of :mod:`experiments.fit.closure_fit`: concentration of the residuals in
bins of $\kappa_{0,\ell}$, divided by $\kappa_{0,\ell}$, for real zones of the two cell sizes
and for the simulated reference; the model predicts flat profiles at the level $c$, the nominal
number of looks the level 1. (b) Normal scores $s=\Phi^{-1}(u)$ of the cells of one coherence
class against standard normal quantiles, under $\mathcal{VM}(0,\hat c\kappa_{0,\ell})$ (real and
simulated) and $\mathcal{VM}(0,\kappa_{0,\ell})$.

Input: the run directory of ``closure_fit.py``. Output: a PDF.

Example:
    python experiments/figures/fig_fit.py --run <runs>/runs/<run_id> --out fit.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from style import COLUMN, GRID, INK, MUTED, S1, UAVSAR, style


def main(args):
    style()
    prof = np.load(Path(args.run) / "profiles.npz")
    sc = np.load(Path(args.run) / "scores.npz")
    t = args.triplet
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLUMN, 1.9),
                                   gridspec_kw=dict(wspace=0.42, left=0.13, right=0.98,
                                                    top=0.92, bottom=0.2))
    series = []
    for win, col in zip(args.windows, (UAVSAR, S1)):
        lab = win.replace("x", "×")
        series += [("uavsar", win, col, f"real, {lab}"), ("sim_h0", win, col, f"simulated, {lab}")]
    for src, win, col, label in series:
        k = f"{src}|{win}|{t}"
        if f"{k}|kappa0" not in prof:
            continue
        k0, sc_ = prof[f"{k}|kappa0"], prof[f"{k}|scale"]
        keep = k0 >= args.min_kappa                    # below: residuals nearly uniform
        real = src == "uavsar"
        ax1.plot(k0[keep], sc_[keep], "o" if real else "s", ms=3.0, color=col,
                 mfc=col if real else "white", mew=0.8, ls="-" if real else ":", lw=0.6,
                 label=label)
    ax1.axhline(1.0, color=INK, lw=0.6, ls="--")
    ax1.text(0.03, 0.86, "nominal looks ($c=1$)", transform=ax1.transAxes, ha="left",
             va="top", fontsize=6, color=INK)
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1.1)
    ax1.set_xlabel(r"$\kappa_{0,\ell}$", labelpad=1)
    ax1.set_ylabel(r"$\hat\kappa/\kappa_{0,\ell}$", labelpad=1)
    ax1.legend(fontsize=5.5, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.64),
               handlelength=1.8, ncol=1)
    ax1.set_title("(a)", loc="left", fontweight="bold", pad=2)

    z = stats.norm.ppf(sc["probs"])
    cl = args.cls
    ax2.plot(z, z, color=GRID, lw=1.6)
    for src, law, col, ls, label in (
            ("uavsar", "vm_c", UAVSAR, "-", r"real, $\hat c\kappa_{0,\ell}$"),
            ("sim_h0", "vm_c", MUTED, "--", r"simulated, $\hat c\kappa_{0,\ell}$"),
            ("uavsar", "vm_1", S1, "-", r"real, $\kappa_{0,\ell}$")):
        key = f"{src}|{args.windows[0]}|{t}|{cl}|{law}"
        if key in sc:
            ax2.plot(z, sc[key], color=col, ls=ls, lw=1.0, label=label)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-8, 8)
    ax2.set_xlabel("Normal quantile", labelpad=1)
    ax2.set_ylabel("Normal score $s_\\ell$", labelpad=1)
    ax2.legend(fontsize=5.5, frameon=False, loc="upper left", handlelength=1.6)
    ax2.text(0.97, 0.04, rf"$\gamma_{{\min}}$ {cl.replace('-', '–')}", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=6)
    ax2.set_title("(b)", loc="left", fontweight="bold", pad=2)
    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.savefig(args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--triplet", type=int, default=10)
    p.add_argument("--windows", nargs=2, default=["15x5", "30x10"])
    p.add_argument("--cls", default="0.5-0.7", help="Coherence class of (b).")
    p.add_argument("--min-kappa", type=float, default=4.0,
                   help="Bins of (a) with a smaller median kappa0 are left out (c kappa0 < 1).")
    p.add_argument("--out", default="fit.pdf")
    main(p.parse_args())
