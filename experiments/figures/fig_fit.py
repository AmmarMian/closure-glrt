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
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy import stats
from style import INK, MUTED, PAGE, style

# Colorblind-safe encoding: each law has its own line style and marker; colors only repeat it
# (Okabe-Ito blue, vermillion, black), and every curve is labelled directly on the plot.
BLUE, VERM = "#0072B2", "#D55E00"
LAWS = {"vm_c": dict(color=BLUE, ls="-", lw=1.7, marker=None,
                     text="model $\\hat c\\kappa_{0,\\ell}$"),
        "vm_free": dict(color=VERM, ls="--", lw=1.4, marker="^",
                        text="free concentration $\\hat\\kappa$, equal weights"),
        "vm_1": dict(color=INK, ls=":", lw=1.6, marker="x", text="nominal looks ($c=1$)")}
FONT = 8.5


def label_at(ax, x, y, text, color, dx=1.08, dy=1.0, ha="left", va="center", log=True):
    """Text next to the point (x, y) of a curve, in data coordinates."""
    ax.text(x * dx if log else x + dx, y * dy if log else y + dy, text, color=color,
            fontsize=FONT - 0.5, ha=ha, va=va)


def spread_panel(ax, prof, src, sim, win, t, min_kappa):
    """95 % quantile of |residual| per bin of kappa0: real and simulated points, model and c=1."""
    k = f"{src}|{win}|{t}"
    k0 = prof[f"{k}|kappa0"]
    keep = k0 >= min_kappa
    x = k0[keep]
    for key, law, dy in (("q95_model", "vm_c", 1.25), ("q95_nominal", "vm_1", 0.8)):
        st = LAWS[law]
        y = np.degrees(prof[f"{k}|{key}"][keep])
        ax.plot(x, y, color=st["color"], ls=st["ls"], lw=st["lw"])
        if st["marker"]:
            ax.plot(x[::3], y[::3], ls="none", marker=st["marker"], ms=4.5, color=st["color"],
                    mfc="white", mew=0.9)
    for s_, real in ((src, True), (sim, False)):
        kk = f"{s_}|{win}|{t}"
        if f"{kk}|q95" not in prof:
            continue
        xs = prof[f"{kk}|kappa0"]
        m = (xs >= min_kappa) & (xs <= 1.5 * x.max())     # range of the real cells
        ax.plot(xs[m], np.degrees(prof[f"{kk}|q95"][m]), "o" if real else "s",
                ms=4.2 if real else 4.6, color=INK, mfc=INK if real else "white", mew=0.9,
                ls="none", zorder=3, label="real zones" if real else "simulated zones ($H_0$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"precision $\kappa_{0,\ell}$ of the cell", fontsize=FONT)
    ax.set_ylabel(r"95% quantile of $|\hat\Phi_\ell-\hat\mu|$ (deg)", fontsize=FONT)


def qq_panel(ax, sc, src, sim, win, t, ylim=5.0):
    z = stats.norm.ppf(sc["probs"])
    ax.plot([-ylim, ylim], [-ylim, ylim], color=MUTED, lw=0.8, ls=(0, (1, 2)))
    for law, st in LAWS.items():
        key = f"{src}|{win}|{t}|all|{law}"
        if key not in sc:
            continue
        y = sc[key]
        ax.plot(z, y, color=st["color"], ls=st["ls"], lw=st["lw"])
        if st["marker"]:
            ax.plot(z[::8], y[::8], ls="none", marker=st["marker"], ms=4.5, color=st["color"],
                    mfc="white", mew=0.9)
    key = f"{sim}|{win}|{t}|all|vm_c"
    if key in sc:
        ax.plot(z[::6], sc[key][::6], ls="none", marker="s", ms=4.2, mfc="white", mew=0.8,
                color=INK)
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel("standard normal quantile", fontsize=FONT)
    ax.set_ylabel("quantile of the normal scores", fontsize=FONT)


def main(args):
    style()
    plt.rcParams.update({"xtick.labelsize": FONT - 0.5, "ytick.labelsize": FONT - 0.5})
    prof = np.load(Path(args.run) / "profiles.npz")
    sc = np.load(Path(args.run) / "scores.npz")
    rows = [("uavsar", "sim_h0", args.window, args.triplet,
             "UAVSAR")]
    if args.s1_triplet:
        rows.append(("sentinel1", "sim_h0_s1", args.s1_window, args.s1_triplet,
                     "Sentinel-1"))
    fig, axes = plt.subplots(len(rows), 2, figsize=(PAGE, 2.75 * len(rows)), squeeze=False,
                             gridspec_kw=dict(wspace=0.25, hspace=0.42, left=0.08, right=0.99,
                                              top=0.86, bottom=0.08))
    letters = iter("abcd")
    for (src, sim, win, t, name), (a1, a2) in zip(rows, axes):
        spread_panel(a1, prof, src, sim, win, t, args.min_kappa)
        qq_panel(a2, sc, src, sim, win, t)
        a1.set_title(f"({next(letters)}) {name}", loc="left", fontweight="bold", fontsize=FONT)
        a2.set_title(f"({next(letters)}) {name}", loc="left", fontweight="bold", fontsize=FONT)
        for ax in (a1, a2):
            ax.grid(color="#e4e3dd", lw=0.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"],
                      ms=5, mfc="white", mew=1.0, label=st["text"]) for st in LAWS.values()]
    handles += [Line2D([], [], color=INK, ls="none", marker="o", ms=5, label="real zones"),
                Line2D([], [], color=INK, ls="none", marker="s", ms=5, mfc="white",
                       label="simulated zones ($H_0$), model"),
                Line2D([], [], color=MUTED, ls=(0, (1, 2)), lw=0.9,
                       label="$\\mathcal{N}(0,1)$ (b, d)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=FONT,
               handlelength=3.6, columnspacing=1.6, bbox_to_anchor=(0.5, 1.0))
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
