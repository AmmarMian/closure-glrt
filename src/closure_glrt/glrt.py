r"""Scale-free GLRT for a nonzero closure over a zone of cells.

Model: the closures of the $L$ cells of a zone are independent,
$\hat\Phi_\ell\sim\mathcal{VM}(\mu,c\,w_\ell)$, with known weights $w_\ell>0$
(De Zan's $\kappa_{0,\ell}$) and an unknown scale $c>0$ common to the zone.
Test $\mathcal H_0:\mu=0$ against $\mathcal H_1:\mu\neq0$, $c$ free under both.

With $\rho=\sum_\ell w_\ell e^{j\hat\Phi_\ell}=Re^{j\hat\mu}$, $C=\operatorname{Re}\rho$,
$W=\sum_\ell w_\ell$, $g(c)=\sum_\ell\log I_0(cw_\ell)$ and its Legendre transform
$g^*(X)=\sup_{c\ge0}[cX-g(c)]$, the GLRT statistic is

$$\Lambda=2\bigl[g^*(R)-g^*(\max(C,0))\bigr].$$

It is unchanged when all weights are multiplied by one constant, so the nominal number of
looks in $\kappa_0$ drops out. Its threshold comes from $t^2\sim F(1,L-1)$ in the Gaussian
limit, valid for every $c$: reject when $\Lambda>L\log(1+F^{-1}_{1,L-1}(1-P_{fa})/(L-1))$.

**Numerics.** Closures from complex64 data are float32, so the statistic is written to be
accurate in single precision: deficits $D=W-X$ are computed without subtraction
($W-R=\sum_\ell 2w_\ell\sin^2\frac{\hat\Phi_\ell-\hat\mu}{2}$,
$W-C=\sum_\ell 2w_\ell\sin^2\frac{\hat\Phi_\ell}{2}$), $1-A(x)$, $A=I_1/I_0$, uses its
asymptotic series at large $x$, and $g^*(X)=-\hat cD-\sum_\ell\log\tilde I_0(\hat cw_\ell)$
with $\tilde I_0(x)=e^{-x}I_0(x)$ and $\sum_\ell w_\ell[1-A(\hat cw_\ell)]=D$.

Backend-agnostic (:mod:`closure_glrt.backend`): closures and weights of shape ``(..., L)``
on the backend ``backend_name`` give one value per zone, shape ``(...)``, in their dtype.
Padding cells have weight 0 and a finite closure (e.g. 0). p-values and thresholds are
computed on the host (scipy), at the boundary.
"""

from typing import Union

import numpy as np
from scipy import stats

from closure_glrt.backend import (
    Array,
    Backend,
    bessel_i0e,
    bessel_i1e,
    cast_like,
    get_backend_module,
    get_data_on_device,
    to_numpy,
)
from closure_glrt.detection import Detector

__all__ = [
    "weighted_resultant",
    "scale_ml",
    "legendre",
    "glrt",
    "pvalue",
    "threshold",
    "glrt_known_scale",
    "confidence_halfwidth",
    "known_scale_halfwidth",
    "ScaleFreeClosureGLRT",
]

_X_ASYMPT = 30.0
_N_ITER = 100


def weighted_resultant(phi: Array, weights: Array, backend_name: Union[str, Backend] = "numpy"):
    r"""$R$, $\hat\mu$, $C$ and $W$ of $\sum_\ell w_\ell e^{j\hat\Phi_\ell}$."""
    be = get_backend_module(backend_name)
    re, im = (weights * be.cos(phi)).sum(-1), (weights * be.sin(phi)).sum(-1)
    return be.hypot(re, im), be.arctan2(im, re), re, weights.sum(-1)


def _deficits(phi, w, backend_name):
    """mu-hat, W - R, W - C (without cancellation), and R."""
    be = get_backend_module(backend_name)
    re, im = (w * be.cos(phi)).sum(-1), (w * be.sin(phi)).sum(-1)
    mu = be.arctan2(im, re)
    d1 = (2 * w * be.sin((phi - mu[..., None]) / 2) ** 2).sum(-1)
    d0 = (2 * w * be.sin(phi / 2) ** 2).sum(-1)
    return mu, d1, d0, be.hypot(re, im), re


def _one_minus_A(x, backend_name):
    r"""$1-I_1(x)/I_0(x)$; series $\frac1{2x}+\frac1{8x^2}+\frac1{8x^3}+\frac{25}{128x^4}+\frac{13}{32x^5}$ for large $x$."""
    be = get_backend_module(backend_name)
    large = x > _X_ASYMPT
    one = be.ones_like(x)
    xs = be.where(large, one, x)
    direct = 1 - bessel_i1e(backend_name, xs) / bessel_i0e(backend_name, xs)
    u = 1 / be.where(large, x, one)
    series = u * (0.5 + u * (0.125 + u * (0.125 + u * (25 / 128 + u * 13 / 32))))
    return be.where(large, series, direct)


def _deficit_fun(c, w, backend_name):
    """S(c) = sum_l w_l [1 - A(c w_l)] = W - g'(c), decreasing from W to 0."""
    return (w * _one_minus_A(c[..., None] * w, backend_name)).sum(-1)


def _log_i0e_sum(c, w, backend_name):
    be = get_backend_module(backend_name)
    return be.log(bessel_i0e(backend_name, c[..., None] * w)).sum(-1)


def _bisect_logc(fun, target, backend_name, decreasing, n_iter=_N_ITER):
    """Root in c of a monotone ``fun(c) = target``, bisection in log c on [e^-40, e^40]."""
    be = get_backend_module(backend_name)
    lo = be.full_like(target, -40.0)
    hi = be.full_like(target, 40.0)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        f = fun(be.exp(mid))
        go_right = (f > target) if decreasing else (f < target)
        lo = be.where(go_right, mid, lo)
        hi = be.where(go_right, hi, mid)
    return be.exp((lo + hi) / 2)


def _scale_from_deficit(D, w, backend_name):
    """c with S(c) = D; 0 where D >= W (X <= 0), inf where D <= 0 (X >= W)."""
    be = get_backend_module(backend_name)
    c = _bisect_logc(lambda c: _deficit_fun(c, w, backend_name), D, backend_name, True)
    c = be.where(D >= w.sum(-1), be.zeros_like(c), c)
    return be.where(D <= 0, be.full_like(c, float("inf")), c)


def _gstar_from_deficit(D, w, backend_name, c=None):
    """g*(W - D) = -c D - sum log I0e(c w); 0 where D >= W, inf where D <= 0."""
    be = get_backend_module(backend_name)
    if c is None:
        c = _scale_from_deficit(D, w, backend_name)
    finite = be.isfinite(c)
    cs = be.where(finite, c, be.zeros_like(c))
    out = -cs * D - _log_i0e_sum(cs, w, backend_name)
    out = be.where(D >= w.sum(-1), be.zeros_like(out), out)
    return be.where(finite, out, be.full_like(out, float("inf")))


def scale_ml(X: Array, weights: Array, backend_name: Union[str, Backend] = "numpy") -> Array:
    r"""Root $\hat c$ of $g'(\hat c)=\sum_\ell w_\ell A(\hat cw_\ell)=X$ (0 if $X\le0$, inf if $X\ge W$).

    For $X$ close to $W$ prefer :func:`glrt` with ``details=True``, which forms $W-X$
    without cancellation.
    """
    return _scale_from_deficit(weights.sum(-1) - X, weights, backend_name)


def legendre(X: Array, weights: Array, backend_name: Union[str, Backend] = "numpy") -> Array:
    r"""$g^*(X)$; 0 for $X\le0$."""
    return _gstar_from_deficit(weights.sum(-1) - X, weights, backend_name)


def glrt(phi: Array, weights: Array, backend_name: Union[str, Backend] = "numpy",
         details: bool = False):
    r"""Exact statistic $\Lambda=2[g^*(R)-g^*(\max(C,0))]$ per zone.

    With ``details=True`` returns a dict with ``lam``, ``mu`` ($\hat\mu$), ``c1`` and ``c0``
    (ML scales under $\mathcal H_1$ and $\mathcal H_0$), ``R``, ``C``, ``W``.
    """
    be = get_backend_module(backend_name)
    w = be.broadcast_to(weights, phi.shape)
    mu, d1, d0, R, C = _deficits(phi, w, backend_name)
    W = w.sum(-1)
    d0 = be.where(d0 > W, W, d0)  # C <= 0: g*(max(C, 0)) = g*(0) = 0
    c1 = _scale_from_deficit(d1, w, backend_name)
    c0 = _scale_from_deficit(d0, w, backend_name)
    lam = 2 * (_gstar_from_deficit(d1, w, backend_name, c1)
               - _gstar_from_deficit(d0, w, backend_name, c0))
    lam = be.where(lam > 0, lam, be.zeros_like(lam))
    if details:
        return {"lam": lam, "mu": mu, "c1": c1, "c0": c0, "R": R, "C": C, "W": W}
    return lam


def pvalue(lam, n_cells) -> np.ndarray:
    r"""p-value of $\Lambda$, $P(F_{1,L-1}>(L-1)(e^{\Lambda/L}-1))$, on the host."""
    lam = to_numpy(lam).astype(np.float64)
    L = np.asarray(to_numpy(n_cells), dtype=np.float64)
    f = (L - 1) * np.expm1(np.clip(lam, 0, None) / L)
    return stats.f.sf(f, 1, L - 1)


def threshold(n_cells, pfa: float) -> np.ndarray:
    r"""Threshold $L\log(1+F^{-1}_{1,L-1}(1-P_{fa})/(L-1))$ on $\Lambda$, on the host."""
    L = np.asarray(n_cells, dtype=np.float64)
    return L * np.log1p(stats.f.ppf(1 - pfa, 1, L - 1) / (L - 1))


def glrt_known_scale(phi: Array, weights: Array, scale: float = 1.0,
                     backend_name: Union[str, Backend] = "numpy") -> Array:
    r"""GLRT with the scale known, $\Lambda=2c(R-C)$, compared with $\chi^2_1$.

    ``scale=1`` with $w_\ell=\kappa_{0,\ell}$ at the nominal looks is the nominal-looks test;
    ``scale`` $=N_{\mathrm{eff}}/N$ is the plug-in ENL test.
    """
    be = get_backend_module(backend_name)
    w = be.broadcast_to(weights, phi.shape)
    _, d1, d0, _, _ = _deficits(phi, w, backend_name)
    return 2 * scale * (d0 - d1)


def confidence_halfwidth(phi: Array, weights: Array, alpha: float = 0.05,
                         backend_name: Union[str, Backend] = "numpy") -> Array:
    r"""Half-width $h$ of the $1-\alpha$ confidence arc $|\mu-\hat\mu|\le h$ (exact statistic).

    Testing $\mu=\mu_0$ is testing 0 on the closures rotated by $-\mu_0$, which keeps $R$ and
    replaces $C$ by $R\cos(\hat\mu-\mu_0)$. The statistic increases with $|\hat\mu-\mu_0|$, so
    the accepted set is an arc whose edge satisfies $g^*(R\cos h)=g^*(R)-\eta/2$, $\eta$ the
    threshold at level $\alpha$. On $X>0$, $g^*$ is parametrised by $c$:
    $g^*=-cS(c)-\sum\log\tilde I_0(cw_\ell)$ at $X=W-S(c)$, increasing in $c$; one bisection
    gives $c^*$, then $1-\cos h=(S(c^*)-(W-R))/R$. $h=\pi$ when $g^*(R)\le\eta/2$.
    """
    be = get_backend_module(backend_name)
    w = be.broadcast_to(weights, phi.shape)
    _, d1, _, R, _ = _deficits(phi, w, backend_name)
    eta = threshold(to_numpy((w > 0).sum(-1)), alpha)
    eta = cast_like(get_data_on_device(eta, backend_name), R, backend_name)
    target = _gstar_from_deficit(d1, w, backend_name) - eta / 2

    def gstar_c(c):
        return -c * _deficit_fun(c, w, backend_name) - _log_i0e_sum(c, w, backend_name)

    cstar = _bisect_logc(gstar_c, target, backend_name, decreasing=False)
    delta = (_deficit_fun(cstar, w, backend_name) - d1) / R   # 1 - cos h
    delta = be.where(delta > 0, delta, be.zeros_like(delta))
    delta = be.where(delta < 2, delta, 2 * be.ones_like(delta))
    h = 2 * be.arcsin(be.sqrt(delta / 2))
    return be.where(target <= 0, be.full_like(h, np.pi), h)


def known_scale_halfwidth(phi: Array, weights: Array, scale: float = 1.0, alpha: float = 0.05,
                          backend_name: Union[str, Backend] = "numpy") -> Array:
    r"""Half-width of the $1-\alpha$ arc obtained by inverting :func:`glrt_known_scale`.

    Testing $\mu_0$ gives $2cR(1-\cos(\hat\mu-\mu_0))\le\chi^2_{1,1-\alpha}$, hence
    $1-\cos h=\chi^2_{1,1-\alpha}/(2cR)$, and $h=\pi$ when this exceeds 2.
    """
    be = get_backend_module(backend_name)
    w = be.broadcast_to(weights, phi.shape)
    _, _, _, R, _ = _deficits(phi, w, backend_name)
    q = float(stats.chi2.ppf(1 - alpha, 1))
    delta = q / (2 * scale * R)
    delta = be.where(delta < 2, delta, 2 * be.ones_like(delta))
    return 2 * be.arcsin(be.sqrt(delta / 2))


class ScaleFreeClosureGLRT(Detector):
    """Zone detector: :func:`glrt` as ``compute``, the $F(1, L-1)$ threshold as ``get_threshold``.

    Parameters
    ----------
    n_cells : int
        Number of cells $L$ per zone.
    backend_name : str or Backend
    """

    def __init__(self, n_cells: int, backend_name: Union[str, Backend] = "numpy"):
        self.n_cells = n_cells
        self.backend_name = backend_name

    def compute(self, X: Array, weights: Array) -> Array:
        """Statistic per zone; ``X`` are the closures ``(..., L)``, ``weights`` the precisions."""
        return glrt(X, weights, self.backend_name)

    def get_threshold(self, pfa: float) -> float:
        return float(threshold(self.n_cells, pfa))
