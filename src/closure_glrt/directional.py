r"""Circular laws and goodness of fit for the closure model.

Von Mises $\mathcal{VM}(\mu,\kappa)$ (scipy), wrapped normal $\mathcal{WN}(\mu,\sigma^2)$, and the
leave-one-out probability integral transform (PIT) used to check
$\hat\Phi_\ell\sim\mathcal{VM}(\mu,c\,w_\ell)$ on zones where the true closure is unknown.

Reference: K. V. Mardia and P. E. Jupp, Directional Statistics, Wiley, 2000.
"""

import numpy as np
from scipy import stats

from closure_glrt.glrt import scale_ml

__all__ = [
    "wrap",
    "fit_von_mises_ml",
    "WrappedNormal",
    "fit_wrapped_normal",
    "vm_cdf",
    "leave_one_out",
    "pit",
    "leave_one_out_pit",
    "ks_uniform",
]


def wrap(x):
    """Angles wrapped to (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(x, dtype=np.float64)))


def fit_von_mises_ml(phi):
    r"""Maximum likelihood $(\hat\mu,\hat\kappa)$ of one von Mises law."""
    kappa, mu, _ = stats.vonmises.fit(np.asarray(phi, dtype=np.float64).ravel(), fscale=1)
    return float(mu), float(kappa)


class WrappedNormal:
    r"""Wrapped normal law on $[-\pi,\pi)$; ``pdf`` and ``cdf`` like a frozen scipy law."""

    def __init__(self, mu, sigma):
        self.mu, self.sigma = float(mu), float(sigma)
        n = int(np.ceil(6 * self.sigma / (2 * np.pi))) + 2
        self._k = np.arange(-n, n + 1)

    def pdf(self, x):
        x = np.asarray(x, dtype=np.float64)[..., None]
        return stats.norm.pdf(x + 2 * np.pi * self._k, self.mu, self.sigma).sum(-1)

    def cdf(self, x):
        x = np.clip(np.asarray(x, dtype=np.float64), -np.pi, np.pi)[..., None]
        shift = 2 * np.pi * self._k
        return (stats.norm.cdf(x + shift, self.mu, self.sigma)
                - stats.norm.cdf(-np.pi + shift, self.mu, self.sigma)).sum(-1)


def fit_wrapped_normal(phi):
    r"""Moment fit: $\hat\mu=\arg\bar z$, $\hat\sigma^2=-2\log|\bar z|$, $\bar z$ the mean of $e^{j\varphi}$."""
    z = np.mean(np.exp(1j * np.asarray(phi, dtype=np.float64).ravel()))
    return float(np.angle(z)), float(np.sqrt(-2 * np.log(np.abs(z))))


def vm_cdf(d, kappa):
    r"""CDF of $\mathcal{VM}(0,\kappa)$ at offsets ``d`` in $(-\pi,\pi]$ (0 at $-\pi$)."""
    return stats.vonmises.cdf(np.asarray(d), np.asarray(kappa), loc=0.0)


def leave_one_out(phi, weights, batch: int = 20_000):
    r"""Leave-one-out residual and scale of every cell of every zone.

    For the cells $\ell=1..L$ of a zone (last axis), $\hat\mu_{-\ell}=\arg\sum_{m\ne\ell}w_me^{j\hat\Phi_m}$,
    $d_\ell=\operatorname{wrap}(\hat\Phi_\ell-\hat\mu_{-\ell})$, and $\hat c_{-\ell}$ is the maximum
    likelihood scale of the other cells around their own direction (root of
    $\sum_{m\ne\ell}w_mA(cw_m)=|\sum_{m\ne\ell}w_me^{j\hat\Phi_m}|$). Zones ``(Z, L)`` are processed
    in batches of ``batch`` zones (the scale needs a ``(batch, L, L)`` array). Host-side, numpy.
    """
    phi = np.asarray(phi, dtype=np.float64)
    w = np.broadcast_to(np.asarray(weights, dtype=np.float64), phi.shape)
    L = phi.shape[-1]
    phi2, w2 = phi.reshape(-1, L), w.reshape(-1, L)
    d = np.empty_like(phi2)
    c = np.empty_like(phi2)
    idx = np.arange(L)
    for s in range(0, len(phi2), batch):
        p, ww = phi2[s:s + batch], w2[s:s + batch]
        z = ww * np.exp(1j * p)
        rest = z.sum(-1, keepdims=True) - z
        d[s:s + batch] = wrap(p - np.angle(rest))
        w_rest = np.broadcast_to(ww[:, None, :], (len(p), L, L)).copy()
        w_rest[:, idx, idx] = 0.0
        c[s:s + batch] = scale_ml(np.abs(rest), w_rest)
    return d.reshape(phi.shape), c.reshape(phi.shape)


def pit(d, kappa, scale, law: str = "vm_c"):
    r"""PIT $u=F(d)$ of residuals ``d`` under one law of the closure noise.

    ``"vm_c"``: $\mathcal{VM}(0,\hat c\kappa)$ (the model); ``"vm_1"``: $\mathcal{VM}(0,\kappa)$ (scale one);
    ``"wn_c"``: wrapped normal of variance $1/(\hat c\kappa)$.
    """
    d, kappa, scale = (np.asarray(x, dtype=np.float64) for x in (d, kappa, scale))
    if law == "vm_1":
        return vm_cdf(d, kappa)
    if law == "vm_c":
        return vm_cdf(d, scale * kappa)
    if law == "wn_c":
        sigma = 1.0 / np.sqrt(scale * kappa)
        k = np.arange(-4, 5)
        u = (stats.norm.cdf((d[..., None] + 2 * np.pi * k) / sigma[..., None])
             - stats.norm.cdf((-np.pi + 2 * np.pi * k) / sigma[..., None])).sum(-1)
        return np.clip(u, 0.0, 1.0)
    raise ValueError(f"unknown law {law!r}")


def leave_one_out_pit(phi, weights, law="vm_c"):
    r"""PIT of each cell against the closure of the other cells of its zone (see :func:`leave_one_out`).

    Under the model the $u_\ell$ are uniform on (0, 1), up to the small effect of estimating
    $\hat\mu_{-\ell}$ and $\hat c_{-\ell}$ (compare real data with a simulated baseline). Returns
    ``u`` and ``d`` of the shape of ``phi``.
    """
    d, c = leave_one_out(phi, weights)
    w = np.broadcast_to(np.asarray(weights, dtype=np.float64), np.shape(d))
    return pit(d, w, c, law), d


def ks_uniform(u):
    """Kolmogorov--Smirnov distance of a sample to the uniform law on (0, 1)."""
    return float(stats.kstest(np.asarray(u).ravel(), "uniform").statistic)
