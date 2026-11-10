r"""Fit figure of the paper (one column): the closure model on real zones.

(a) Spread of the residuals $\hat\Phi_\ell-\hat\mu$ against the precision $\kappa_{0,\ell}$: in bins of
$\kappa_{0,\ell}$, the 95 % quantile of $|\hat\Phi_\ell-\hat\mu|$ on real zones (cells of the two
sizes) and on simulated zones, with the quantile predicted by $\mathcal{VM}(0,\hat c\kappa_{0,\ell})$
(the model) and by $\mathcal{VM}(0,\kappa_{0,\ell})$ (nominal looks). (b) Histogram of the normal
scores of all the cells of all the zones, under the model and with $c=1$, against the standard
normal density.

Input: the run directory of ``closure_fit.py``. Output: a PDF.

Example:
    python experiments/figures/fig_fit.py --run <runs>/runs/<run_id> --out fit.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from style import COLUMN, INK, MUTED, S1, UAVSAR, style


def main(args):
    style()
    prof = np.load(Path(args.run) / "profiles.npz")
    sc = np.load(Path(args.run) / "scores.npz")
    t, w0, w1 = args.triplet, *args.windows
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLUMN, 1.95),
                                   gridspec_kw=dict(wspace=0.38, left=0.12, right=0.98,
                                                    top=0.92, bottom=0.2))
    k = f"uavsar|{w0}|{t}"
    k0 = prof[f"{k}|kappa0"]
    keep = k0 >= args.min_kappa
    ax1.plot(k0[keep], np.degrees(prof[f"{k}|q95_nominal"][keep]), color=INK, ls="--", lw=0.8,
             label=r"$\mathcal{VM}(0,\kappa_{0,\ell})$")
    ax1.plot(k0[keep], np.degrees(prof[f"{k}|q95_model"][keep]), color=UAVSAR, lw=0.8,
             label=r"$\mathcal{VM}(0,\hat c\kappa_{0,\ell})$")
    k1 = f"uavsar|{w1}|{t}"
    if f"{k1}|q95_model" in prof:
        m1 = prof[f"{k1}|kappa0"] >= args.min_kappa
        ax1.plot(prof[f"{k1}|kappa0"][m1], np.degrees(prof[f"{k1}|q95_model"][m1]), color=S1,
                 lw=0.8)
    for src, win, col, label in (("uavsar", w0, UAVSAR, f"real, {w0}"),
                                 ("sim_h0", w0, UAVSAR, f"simulated, {w0}"),
                                 ("uavsar", w1, S1, f"real, {w1}")):
        kk = f"{src}|{win}|{t}"
        if f"{kk}|q95" not in prof:
            continue
        x = prof[f"{kk}|kappa0"]
        m = x >= args.min_kappa
        real = src == "uavsar"
        ax1.plot(x[m], np.degrees(prof[f"{kk}|q95"][m]), "o" if real else "s", ms=2.8, color=col,
                 mfc=col if real else "white", mew=0.7, ls="none",
                 label=label.replace("x", "×"))
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Precision $\kappa_{0,\ell}$", labelpad=1)
    ax1.set_ylabel(r"95 % of $|\hat\Phi_\ell-\hat\mu|$ (deg)", labelpad=1)
    ax1.legend(fontsize=5.3, frameon=False, loc="lower left", handlelength=1.6)
    ax1.set_title("(a)", loc="left", fontweight="bold", pad=2)

    edges = sc["z_edges"]
    mid = (edges[:-1] + edges[1:]) / 2
    curves = (("uavsar", "vm_c", UAVSAR, "-", r"real, $\hat c\kappa_{0,\ell}$"),
              ("sim_h0", "vm_c", MUTED, ":", r"simulated, $\hat c\kappa_{0,\ell}$"),
              ("uavsar", "vm_1", S1, "-", r"real, $\kappa_{0,\ell}$"))
    for src, law, col, ls, label in curves:
        key = f"{src}|{w0}|{t}|all|{law}|hist"
        if key in sc:
            h = np.where(sc[key] > 0, sc[key], np.nan)
            ax2.step(mid, h, where="mid", color=col, ls=ls, lw=0.9, label=label)
    ax2.plot(mid, stats.norm.pdf(mid), color=INK, lw=0.6, ls="--", label=r"$\mathcal{N}(0,1)$")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-6, 1)
    ax2.set_xlim(-6, 6)
    ax2.set_xlabel(r"Normal score $\zeta_\ell$", labelpad=1)
    ax2.set_ylabel("Density", labelpad=1)
    ax2.legend(fontsize=5.0, frameon=False, loc="lower center", handlelength=1.6,
               bbox_to_anchor=(0.5, -0.02))
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
    p.add_argument("--min-kappa", type=float, default=4.0,
                   help="Bins of (a) with a smaller median kappa0 are left out.")
    p.add_argument("--out", default="fit.pdf")
    main(p.parse_args())
