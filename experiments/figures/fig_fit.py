r"""Fit figure of the paper (one column): the closure model on real zones.

(a) Spread of the residuals $\hat\Phi_\ell-\hat\mu$ against the precision $\kappa_{0,\ell}$: in bins of
$\kappa_{0,\ell}$, the 95 % quantile of $|\hat\Phi_\ell-\hat\mu|$ on real zones and on simulated
zones, with the quantile predicted by $\mathcal{VM}(0,\hat c\kappa_{0,\ell})$
(the model) and by $\mathcal{VM}(0,\kappa_{0,\ell})$ (nominal looks). (b) QQ plot of the normal
scores of all the cells of all the zones, under the model and with $c=1$, against the standard
normal law.

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
    t, w0 = args.triplet, args.window
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
    for src, win, col, label in (("uavsar", w0, UAVSAR, f"real, {w0}"),
                                 ("sim_h0", w0, UAVSAR, f"simulated, {w0}")):
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

    z = stats.norm.ppf(sc["probs"])
    ax2.plot([-4.5, 4.5], [-4.5, 4.5], color=INK, lw=0.6, ls="--", label=r"$\mathcal{N}(0,1)$")
    curves = (("uavsar", "vm_c", UAVSAR, "-", r"real, $\hat c\kappa_{0,\ell}$"),
              ("sim_h0", "vm_c", MUTED, ":", r"simulated, $\hat c\kappa_{0,\ell}$"),
              ("uavsar", "vm_1", S1, "-", r"real, $\kappa_{0,\ell}$"))
    for src, law, col, ls, label in curves:
        key = f"{src}|{w0}|{t}|all|{law}"
        if key in sc:
            ax2.plot(z, sc[key], color=col, ls=ls, lw=1.0, label=label)
    ax2.set_xlim(-4.2, 4.2)
    ax2.set_ylim(-6, 6)
    ax2.set_xlabel("Standard normal quantile", labelpad=1)
    ax2.set_ylabel(r"Quantile of $\zeta_\ell$", labelpad=1)
    ax2.legend(fontsize=5.0, frameon=False, loc="upper left", handlelength=1.6,
               bbox_to_anchor=(-0.02, 1.02))
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
    p.add_argument("--window", default="15x5")
    p.add_argument("--min-kappa", type=float, default=4.0,
                   help="Bins of (a) with a smaller median kappa0 are left out.")
    p.add_argument("--out", default="fit.pdf")
    main(p.parse_args())
