r"""Multilook closure phase, sample coherences and De Zan's precision.

For three coregistered acquisitions and a cell of $N$ pixels with snapshots
$\boldsymbol z_n\in\mathbb C^3$, the sample covariance is
$\hat{\boldsymbol\Sigma}=\frac1N\sum_n\boldsymbol z_n\boldsymbol z_n^{\mathsf H}$, the sample
coherences $\hat\gamma_{ab}=[\hat{\boldsymbol\Sigma}]_{ab}/\sqrt{[\hat{\boldsymbol\Sigma}]_{aa}
[\hat{\boldsymbol\Sigma}]_{bb}}$ and the closure
$\hat\Phi=\arg(\hat\gamma_{12}\hat\gamma_{23}\hat\gamma_{31})$.

Backend-agnostic (:mod:`closure_glrt.backend`): every function takes ``backend_name`` and
works on arrays already on that backend. Precision follows the data: complex64 data give
complex64 coherences and float32 closures, magnitudes and precisions; nothing is upcast.

Reference: F. De Zan, M. Zonno, P. López-Dekker, "Phase inconsistencies and multiple
scattering in SAR interferometry", IEEE TGRS 53(12):6608-6616, 2015.
"""

import math
from typing import Union

from closure_glrt.backend import Array, Backend, concatenate, get_backend_module

__all__ = [
    "sample_coherence_matrix",
    "closure_phase",
    "dezan_closure_variance",
    "dezan_concentration",
    "block_sample_covariance",
    "block_lag_covariance",
    "triplet_quantities",
]


def sample_coherence_matrix(X: Array, backend_name: Union[str, Backend] = "numpy") -> Array:
    r"""Sample coherence matrices of data ``X`` of shape ``(..., N, T)`` (N looks, T dates).

    Returns ``(..., T, T)`` with $[\hat\Gamma]_{ab}=\sum_n x_{an}x_{bn}^*/
    (\lVert x_a\rVert\lVert x_b\rVert)$: Hermitian, unit diagonal, dtype of ``X``.
    """
    be = get_backend_module(backend_name)
    C = be.swapaxes(X, -1, -2) @ X.conj()
    p = be.real(be.einsum("...aa->...a", C))
    return C / be.sqrt(p[..., :, None] * p[..., None, :])


def closure_phase(G: Array, i: int = 0, j: int = 1, k: int = 2,
                  backend_name: Union[str, Backend] = "numpy") -> Array:
    r"""$\arg(G_{ij}G_{jk}G_{ki})$ in $(-\pi,\pi]$ for coherence or covariance matrices ``G``."""
    be = get_backend_module(backend_name)
    return be.angle(G[..., i, j] * G[..., j, k] * G[..., k, i])


def dezan_closure_variance(g12, g23, g31, n_looks, closure=0.0,
                           backend_name: Union[str, Backend] = "numpy"):
    r"""First-order closure variance (De Zan et al. 2015), extended to a true closure $\Phi$.

    $$\sigma^2(\Phi)=\frac{1}{2N}\Bigl[3+\gamma_{12}^{-2}+\gamma_{23}^{-2}+\gamma_{31}^{-2}
      -2\cos\Phi\,\frac{\gamma_{12}^2+\gamma_{23}^2+\gamma_{31}^2}{\gamma_{12}\gamma_{23}\gamma_{31}}\Bigr].$$

    It is the Cramér--Rao bound of the closure for $N$ independent circular Gaussian looks.
    ``closure`` is a Python float or an array of the backend.
    """
    be = get_backend_module(backend_name)
    cos_phi = math.cos(closure) if isinstance(closure, (int, float)) else be.cos(closure)
    s = (g12**2 + g23**2 + g31**2) / (g12 * g23 * g31)
    bracket = 3.0 + 1.0 / g12**2 + 1.0 / g23**2 + 1.0 / g31**2 - 2.0 * cos_phi * s
    return bracket / (2.0 * n_looks)


def dezan_concentration(g12, g23, g31, n_looks, closure=0.0,
                        backend_name: Union[str, Backend] = "numpy"):
    r"""Precision $\kappa_0=1/\sigma^2$ of a cell (the weight $w_\ell$ of the test)."""
    return 1.0 / dezan_closure_variance(g12, g23, g31, n_looks, closure, backend_name)


def block_sample_covariance(stack: Array, window: tuple[int, int],
                            backend_name: Union[str, Backend] = "numpy") -> Array:
    r"""Sample covariance of every non-overlapping cell of a stack ``(H, W, T)``.

    Cells of ``window = (r, c)`` pixels; cell ``(i, j)`` covers rows ``i*r:(i+1)*r`` and
    columns ``j*c:(j+1)*c`` (incomplete border cells are dropped). Returns
    ``(H//r, W//c, T, T)`` in the dtype of ``stack``.
    """
    be = get_backend_module(backend_name)
    r, c = window
    H, W, T = stack.shape
    Hm, Wm = H // r, W // c
    X = stack[: Hm * r, : Wm * c].reshape(Hm, r, Wm, c, T)
    X = be.swapaxes(X, 1, 2).reshape(Hm, Wm, r * c, T)
    return (be.swapaxes(X, -1, -2) @ X.conj()) / (r * c)


def block_lag_covariance(stack: Array, window: tuple[int, int], max_lag: int = 2,
                         backend_name: Union[str, Backend] = "numpy") -> Array:
    r"""Entries $[\hat\Sigma_\ell]_{t,t+k}$, $k\le$ ``max_lag``, of every cell of a stack ``(H, W, T)``.

    Returns ``(H//r, W//c, T, max_lag+1)`` in the dtype of ``stack``; ``nan`` where
    $t+k\ge T$. Enough for every triplet of dates at most ``max_lag`` apart, at a memory cost
    $T(k_{\max}+1)$ instead of $T^2$.
    """
    be = get_backend_module(backend_name)
    r, c = window
    H, W, T = stack.shape
    Hm, Wm = H // r, W // c
    X = stack[: Hm * r, : Wm * c].reshape(Hm, r, Wm, c, T)
    blocks = []
    for k in range(max_lag + 1):
        prod = (X[..., : T - k] * X[..., k:].conj()).sum(-2).sum(1) / (r * c)  # (Hm, Wm, T-k)
        if k:
            # nan for t + k >= T; full_like on a slice keeps dtype and device
            pad = be.full_like(blocks[0][..., :k], float("nan"))
            prod = concatenate(backend_name, [prod, pad], axis=-1)
        blocks.append(prod)
    return be.stack(blocks, -1)


def triplet_quantities(lags: Array, n_looks: int,
                       backend_name: Union[str, Backend] = "numpy"):
    r"""Closure, precision and weakest coherence of every consecutive triplet $(t,t+1,t+2)$.

    ``lags`` has shape ``(..., T, 3)`` (lags 0, 1, 2 of :func:`block_lag_covariance`).
    Returns ``phi, kappa0, gmin``, each ``(..., T-2)`` in the real dtype matching ``lags``;
    ``kappa0`` uses the sample coherences and the nominal ``n_looks`` (a common factor is
    irrelevant to the test). Cells with a zero power or a nonpositive precision give ``nan``.
    """
    be = get_backend_module(backend_name)
    P = be.real(lags[..., 0])
    g12 = be.abs(lags[..., :-2, 1]) / be.sqrt(P[..., :-2] * P[..., 1:-1])
    g23 = be.abs(lags[..., 1:-1, 1]) / be.sqrt(P[..., 1:-1] * P[..., 2:])
    g13 = be.abs(lags[..., :-2, 2]) / be.sqrt(P[..., :-2] * P[..., 2:])
    phi = be.angle(lags[..., :-2, 1] * lags[..., 1:-1, 1] * lags[..., :-2, 2].conj())
    kappa = dezan_concentration(g12, g23, g13, n_looks, backend_name=backend_name)
    gmin = be.minimum(be.minimum(g12, g23), g13)
    bad = ~(be.isfinite(phi) & be.isfinite(kappa) & (kappa > 0) & be.isfinite(gmin))
    nan = be.full_like(phi, float("nan"))
    return be.where(bad, nan, phi), be.where(bad, nan, kappa), gmin
