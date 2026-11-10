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


def leave_one_out_pit(phi, weights, law="vm_c"):
    r"""PIT of each cell of each zone against the closure of the other cells of its zone.

    For zone cells $\ell=1..L$ (last axis), $\hat\mu_{-\ell}=\arg\sum_{m\ne\ell}w_me^{j\hat\Phi_m}$,
    $d_\ell=\operatorname{wrap}(\hat\Phi_\ell-\hat\mu_{-\ell})$ and $u_\ell=F(d_\ell)$ with

    * ``"vm_c"``: $\mathcal{VM}(0,\hat c_{-\ell}w_\ell)$, $\hat c_{-\ell}$ the ML scale of the other
      cells around their own direction (the model);
    * ``"vm_1"``: $\mathcal{VM}(0,w_\ell)$ (scale one: weights taken as calibrated);
    * ``"wn_c"``: wrapped normal of variance $1/(\hat c_{-\ell}w_\ell)$.

    Under the model the $u_\ell$ are uniform on (0, 1). Returns ``u`` and ``d`` of the shape of
    ``phi``. Zones are processed as a batch (no loop over zones).
    """
    phi = np.asarray(phi, dtype=np.float64)
    w = np.broadcast_to(np.asarray(weights, dtype=np.float64), phi.shape)
    z = w * np.exp(1j * phi)
    rest = z.sum(-1, keepdims=True) - z
    mu = np.angle(rest)
    d = wrap(phi - mu)
    if law == "vm_1":
        return vm_cdf(d, w), d
    # ML scale of the other cells around their own direction: sum_{m!=l} w_m cos(phi_m - mu_-l)
    L = phi.shape[-1]
    X = np.real(rest * np.exp(-1j * mu))
    w_rest = np.broadcast_to(w[..., None, :], phi.shape + (L,)).copy()
    idx = np.arange(L)
    w_rest[..., idx, idx] = 0.0
    c = scale_ml(X, w_rest)
    kappa = c * w
    if law == "vm_c":
        return vm_cdf(d, kappa), d
    if law == "wn_c":
        sigma = 1.0 / np.sqrt(kappa)
        n = 4
        k = np.arange(-n, n + 1)
        u = (stats.norm.cdf((d[..., None] + 2 * np.pi * k) / sigma[..., None])
             - stats.norm.cdf((-np.pi + 2 * np.pi * k) / sigma[..., None])).sum(-1)
        return np.clip(u, 0.0, 1.0), d
    raise ValueError(f"unknown law {law!r}")


def ks_uniform(u):
    """Kolmogorov--Smirnov distance of a sample to the uniform law on (0, 1)."""
    return float(stats.kstest(np.asarray(u).ravel(), "uniform").statistic)
