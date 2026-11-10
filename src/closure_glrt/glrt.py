r"""Scale-free GLRT for a nonzero closure over a zone of cells.

Model: the closures of the $L$ cells of a zone are independent,
$\hat\Phi_\ell\sim\mathcal{VM}(\mu,c\,w_\ell)$, with known weights $w_\ell>0$
(De Zan's $\kappa_{0,\ell}$) and an unknown scale $c>0$ common to the zone.
Test $\mathcal H_0:\mu=0$ against $\mathcal H_1:\mu\neq0$, $c$ free under both.

With $\rho=\sum_\ell w_\ell e^{j\hat\Phi_\ell}=Re^{j\hat\mu}$, $C=\operatorname{Re}\rho$,
$g(c)=\sum_\ell\log I_0(cw_\ell)$ and its Legendre transform
$g^*(X)=\sup_{c\ge0}[cX-g(c)]$, the GLRT statistic is

$$\Lambda=2\bigl[g^*(R)-g^*(\max(C,0))\bigr].$$

It is unchanged when all weights are multiplied by one constant, so the nominal number of
looks in $\kappa_0$ drops out. Its threshold comes from $t^2\sim F(1,L-1)$ in the Gaussian
limit, valid for every $c$: reject when $\Lambda>L\log(1+F^{-1}_{1,L-1}(1-P_{fa})/(L-1))$.

Every function works on the last axis: closures and weights of shape ``(..., L)`` give one
value per zone, shape ``(...)``. Cells with weight 0 do not contribute (padding). Inputs may
be numpy arrays or torch tensors (same type out), which runs the root searches on a GPU.
Padding cells must have weight 0 and a finite closure (e.g. 0).
"""

import numpy as np
from scipy import special, stats

__all__ = [
    "weighted_resultant",
    "scale_ml",
    "legendre",
    "glrt",
    "pvalue",
    "threshold",
    "glrt_known_scale",
    "confidence_halfwidth",
]


class _Numpy:
    lib = np
    i0e, i1e = staticmethod(special.i0e), staticmethod(special.i1e)
    atan2 = staticmethod(np.arctan2)


def _ns(x):
    """numpy or torch namespace of ``x``."""
    if type(x).__module__.startswith("torch"):
        import torch

        class _Torch:
            lib = torch
            i0e, i1e = staticmethod(torch.special.i0e), staticmethod(torch.special.i1e)
            atan2 = staticmethod(torch.atan2)

        return _Torch
    return _Numpy


def _as_float(x, like=None):
    if like is not None and _ns(like) is not _Numpy:
        return like.new_tensor(x) if not hasattr(x, "to") else x.to(like)
    if _ns(x) is not _Numpy:
        return x.double()
    return np.asarray(x, dtype=np.float64)


def weighted_resultant(phi, weights):
    r"""$R$, $\hat\mu$, $C$ and $W=\sum_\ell w_\ell$ of $\sum_\ell w_\ell e^{j\hat\Phi_\ell}$."""
    phi = _as_float(phi)
    w = _as_float(weights, like=phi)
    ns = _ns(phi)
    xp = ns.lib
    re, im = (w * xp.cos(phi)).sum(-1), (w * xp.sin(phi)).sum(-1)
    return xp.hypot(re, im), ns.atan2(im, re), re, w.sum(-1)


def _g(c, w, ns):
    """g(c) = sum_l log I0(c w_l); c (...), w (..., L)."""
    x = c[..., None] * w
    return (ns.lib.log(ns.i0e(x)) + x).sum(-1)


def _gprime(c, w, ns):
    """g'(c) = sum_l w_l A(c w_l), A = I1 / I0."""
    x = c[..., None] * w
    return (w * ns.i1e(x) / ns.i0e(x)).sum(-1)


def _bisect_logc(fun, target, w, ns, n_iter):
    """Root in c of an increasing ``fun(c)`` = target, by bisection in log c on [e^-40, e^40]."""
    xp = ns.lib
    lo = xp.full_like(target, -40.0)
    hi = xp.full_like(target, 40.0)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        above = fun(xp.exp(mid)) > target
        hi = xp.where(above, mid, hi)
        lo = xp.where(above, lo, mid)
    return xp.exp((lo + hi) / 2)


def scale_ml(X, weights, n_iter=100):
    r"""Root $\hat c$ of $g'(\hat c)=\sum_\ell w_\ell A(\hat cw_\ell)=X$.

    $X=R$ gives the maximum likelihood scale under $\mathcal H_1$, $X=\max(C,0)$ under
    $\mathcal H_0$. Returns 0 where $X\le0$ and ``inf`` where $X\ge W$.
    """
    X = _as_float(X)
    w = _as_float(weights, like=X)
    ns = _ns(X)
    xp = ns.lib
    c = _bisect_logc(lambda c: _gprime(c, w, ns), X, w, ns, n_iter)
    c = xp.where(X <= 0, xp.zeros_like(c), c)
    return xp.where(X >= w.sum(-1), xp.full_like(c, float("inf")), c)


def legendre(X, weights, c=None):
    r"""$g^*(X)=\hat cX-g(\hat c)$ at the root $\hat c$ of :func:`scale_ml`; 0 for $X\le0$."""
    X = _as_float(X)
    w = _as_float(weights, like=X)
    ns = _ns(X)
    xp = ns.lib
    if c is None:
        c = scale_ml(X, w)
    out = c * X - _g(c, w, ns)
    return xp.where(X <= 0, xp.zeros_like(out), out)


def glrt(phi, weights, details=False):
    r"""Exact statistic $\Lambda=2[g^*(R)-g^*(\max(C,0))]$ per zone.

    With ``details=True`` returns a dict with ``lam``, ``mu`` ($\hat\mu$), ``c1`` and ``c0``
    (scales under $\mathcal H_1$ and $\mathcal H_0$), ``R``, ``C``, ``W``.
    """
    R, mu, C, W = weighted_resultant(phi, weights)
    ns = _ns(R)
    xp = ns.lib
    w = xp.broadcast_to(_as_float(weights, like=R), tuple(np.shape(phi)))
    C0 = xp.where(C > 0, C, xp.zeros_like(C))
    c1, c0 = scale_ml(R, w), scale_ml(C0, w)
    lam = 2 * (legendre(R, w, c1) - legendre(C0, w, c0))
    if details:
        return {"lam": lam, "mu": mu, "c1": c1, "c0": c0, "R": R, "C": C, "W": W}
    return lam


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=np.float64)


def pvalue(lam, n_cells):
    r"""p-value of $\Lambda$: $P(F_{1,L-1}>(L-1)(e^{\Lambda/L}-1))$ (numpy output)."""
    lam, L = _to_numpy(lam), np.asarray(_to_numpy(n_cells), dtype=np.float64)
    f = (L - 1) * np.expm1(np.clip(lam, 0, None) / L)
    return stats.f.sf(f, 1, L - 1)


def threshold(n_cells, pfa):
    r"""Threshold $L\log(1+F^{-1}_{1,L-1}(1-P_{fa})/(L-1))$ on $\Lambda$."""
    L = np.asarray(n_cells, dtype=np.float64)
    return L * np.log1p(stats.f.ppf(1 - pfa, 1, L - 1) / (L - 1))


def glrt_known_scale(phi, weights, scale=1.0):
    r"""GLRT with the scale known, $\Lambda=2c(R-C)$, compared with $\chi^2_1$.

    ``scale=1`` with $w_\ell=\kappa_{0,\ell}$ at the nominal looks is the nominal-looks test;
    ``scale`` $=N_{\mathrm{eff}}/N$ is the plug-in ENL test.
    """
    R, _, C, _ = weighted_resultant(phi, weights)
    return 2 * scale * (R - C)


def confidence_halfwidth(phi, weights, alpha=0.05, n_iter=100):
    r"""Half-width $h$ of the $1-\alpha$ confidence arc $|\mu-\hat\mu|\le h$ (exact statistic).

    Testing $\mu=\mu_0$ is testing 0 on the closures rotated by $-\mu_0$, which keeps $R$ and
    replaces $C$ by $R\cos(\hat\mu-\mu_0)$. The statistic decreases in $|\hat\mu-\mu_0|$, so the
    accepted set is an arc whose edge satisfies $g^*(R\cos h)=g^*(R)-\eta/2$, $\eta$ the
    threshold at level $\alpha$. On $X>0$, $g^*$ is parametrised by $c$ as
    $(X,g^*)=(g'(c),\,cg'(c)-g(c))$, increasing in $c$: one bisection gives $c^*$ and
    $h=\arccos(g'(c^*)/R)$. $h=\pi$ when $g^*(R)\le\eta/2$ (the zone does not constrain $\mu$).
    """
    R, _, _, _ = weighted_resultant(phi, weights)
    ns = _ns(R)
    xp = ns.lib
    w = xp.broadcast_to(_as_float(weights, like=R), tuple(np.shape(phi)))
    L = (w > 0).sum(-1)
    eta = _as_float(threshold(_to_numpy(L), alpha), like=R)
    target = legendre(R, w) - eta / 2
    gstar = lambda c: c * _gprime(c, w, ns) - _g(c, w, ns)  # noqa: E731
    cstar = _bisect_logc(gstar, target, w, ns, n_iter)
    ratio = _gprime(cstar, w, ns) / R
    ratio = xp.where(ratio > 1, xp.ones_like(ratio), ratio)
    h = xp.arccos(ratio)
    return xp.where(target <= 0, xp.full_like(h, np.pi), h)
