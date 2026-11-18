r"""Fit figure of the paper (one column): the closure model on real zones, UAVSAR (top) and,
with ``--s1-triplet``, Sentinel-1 (bottom), against a von Mises law with one free concentration
per zone (equal weights) and the nominal number of looks ($c=1$).

(a) Spread of the residuals $\hat\Phi_\ell-\hat\mu$ against the precision $\kappa_{0,\ell}$: in
bins of $\kappa_{0,\ell}$, the 95 % quantile of $|\hat\Phi_\ell-\hat\mu|$ on real and simulated
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
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy import stats
from style import COLUMN, INK, MUTED, S1, SAMEDAY, UAVSAR, style

LAWS = (("vm_c", UAVSAR, "-", r"model $\hat c\kappa_{0,\ell}$"),
        ("vm_free", SAMEDAY, "-.", r"free $\hat\kappa$, equal weights"),
        ("vm_1", S1, "--", r"nominal $\kappa_{0,\ell}$ ($c=1$)"))


def spread_panel(ax, prof, src, sim, win, t, min_kappa):
    """95 % quantile of |residual| per bin of kappa0: real and simulated points, three laws."""
    k = f"{src}|{win}|{t}"
    k0 = prof[f"{k}|kappa0"]
    keep = k0 >= min_kappa
    for key, (law, col, ls, label) in zip(("q95_model", "q95_nominal"), (LAWS[0], LAWS[2])):
        ax.plot(k0[keep], np.degrees(prof[f"{k}|{key}"][keep]), color=col, ls=ls, lw=0.9,
                label=label)
    for s, real in ((src, True), (sim, False)):
        kk = f"{s}|{win}|{t}"
        if f"{kk}|q95" not in prof:
            continue
        x = prof[f"{kk}|kappa0"]
        m = x >= min_kappa
        ax.plot(x[m], np.degrees(prof[f"{kk}|q95"][m]), "o" if real else "s", ms=2.6,
                color=INK, mfc=INK if real else "white", mew=0.6, ls="none",
                label="real" if real else r"simulated $H_0$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"precision $\kappa_{0,\ell}$", labelpad=1)
    ax.set_ylabel(r"95 % of $|\hat\Phi_\ell-\hat\mu|$ (deg)", labelpad=1)


def qq_panel(ax, sc, src, sim, win, t):
    z = stats.norm.ppf(sc["probs"])
    ax.plot([-4.5, 4.5], [-4.5, 4.5], color=MUTED, lw=0.6, ls=":")
    for law, col, ls, label in LAWS:
        key = f"{src}|{win}|{t}|all|{law}"
        if key in sc:
            ax.plot(z, sc[key], color=col, ls=ls, lw=1.0, label=label)
    key = f"{sim}|{win}|{t}|all|vm_c"
    if key in sc:
        ax.plot(z, sc[key], color=INK, ls="none", marker="s", ms=2.0, mfc="white", mew=0.5,
                markevery=6)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-6, 6)
    ax.set_xlabel("standard normal quantile", labelpad=1)
    ax.set_ylabel(r"quantile of the normal score", labelpad=1)


def main(args):
    style()
    prof = np.load(Path(args.run) / "profiles.npz")
    sc = np.load(Path(args.run) / "scores.npz")
    rows = [("uavsar", "sim_h0", args.window, args.triplet, "UAVSAR, $t=10$, 15×5")]
    if args.s1_triplet:
        rows.append(("sentinel1", "sim_h0_s1", args.s1_window, args.s1_triplet,
                     "Sentinel-1, $t=65$, 2×4"))
    fig, axes = plt.subplots(len(rows), 2, figsize=(COLUMN, 1.75 * len(rows) + 0.55),
                             squeeze=False,
                             gridspec_kw=dict(wspace=0.42, hspace=0.62, left=0.12, right=0.98,
                                              top=0.95, bottom=0.2))
    letters = iter("abcd")
    for (src, sim, win, t, name), (a1, a2) in zip(rows, axes):
        spread_panel(a1, prof, src, sim, win, t, args.min_kappa)
        qq_panel(a2, sc, src, sim, win, t)
        a1.set_title(f"({next(letters)}) {name}", loc="left", fontweight="bold", pad=2,
                     fontsize=6.5)
        a2.set_title(f"({next(letters)}) normal scores", loc="left", fontweight="bold", pad=2,
                     fontsize=6.5)
        for ax in (a1, a2):
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 1].get_legend_handles_labels()
    order = [l2.index(LAWS[0][3]), l2.index(LAWS[1][3]), l2.index(LAWS[2][3])]
    pts = [i for i, lab in enumerate(l1) if lab in ("real", r"simulated $H_0$")]
    fig.legend([h2[i] for i in order] + [h1[i] for i in pts],
               [l2[i] for i in order] + [l1[i] for i in pts], loc="lower center",
               ncol=2, frameon=False, fontsize=5.6, handlelength=2.0,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--triplet", type=int, default=10)
    p.add_argument("--window", default="15x5")
    p.add_argument("--min-kappa", type=float, default=4.0,
                   help="Bins of (a) with a smaller median kappa0 are left out.")
    p.add_argument("--s1-triplet", default=None, help="Sentinel-1 group of the run (row 2).")
    p.add_argument("--s1-window", default="2x4")
    p.add_argument("--out", default="fit.pdf")
    main(p.parse_args())
