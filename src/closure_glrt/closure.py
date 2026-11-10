r"""Multilook closure phase, sample coherences and De Zan's precision.

For three coregistered acquisitions and a cell of $N$ pixels with snapshots
$\boldsymbol z_n\in\mathbb C^3$, the sample covariance is
$\hat{\boldsymbol\Sigma}=\frac1N\sum_n\boldsymbol z_n\boldsymbol z_n^{\mathsf H}$, the sample
coherences $\hat\gamma_{ab}=[\hat{\boldsymbol\Sigma}]_{ab}/\sqrt{[\hat{\boldsymbol\Sigma}]_{aa}
[\hat{\boldsymbol\Sigma}]_{bb}}$ and the closure
$\hat\Phi=\arg(\hat\gamma_{12}\hat\gamma_{23}\hat\gamma_{31})$.

Reference: F. De Zan, M. Zonno, P. López-Dekker, "Phase inconsistencies and multiple
scattering in SAR interferometry", IEEE TGRS 53(12):6608-6616, 2015.
"""

import numpy as np

__all__ = [
    "sample_coherence_matrix",
    "closure_phase",
    "dezan_closure_variance",
    "dezan_concentration",
    "block_sample_covariance",
    "block_lag_covariance",
    "triplet_quantities",
]


def sample_coherence_matrix(X):
    r"""Sample coherence matrices of data ``X`` of shape ``(..., N, T)`` (N looks, T dates).

    Returns ``(..., T, T)`` with $[\hat\Gamma]_{ab}=\sum_n x_{an}x_{bn}^*/
    (\lVert x_a\rVert\lVert x_b\rVert)$: Hermitian, unit diagonal.
    """
    X = np.asarray(X)
    C = np.einsum("...na,...nb->...ab", X, X.conj())
    p = np.real(np.einsum("...aa->...a", C))
    return C / np.sqrt(p[..., :, None] * p[..., None, :])


def closure_phase(G, i=0, j=1, k=2):
    r"""$\arg(G_{ij}G_{jk}G_{ki})$ in $(-\pi,\pi]$ for coherence or covariance matrices ``G``."""
    G = np.asarray(G)
    return np.angle(G[..., i, j] * G[..., j, k] * G[..., k, i])


def dezan_closure_variance(g12, g23, g31, n_looks, closure=0.0):
    r"""First-order closure variance (De Zan et al. 2015), extended to a true closure $\Phi$.

    $$\sigma^2(\Phi)=\frac{1}{2N}\Bigl[3+\gamma_{12}^{-2}+\gamma_{23}^{-2}+\gamma_{31}^{-2}
      -2\cos\Phi\,\frac{\gamma_{12}^2+\gamma_{23}^2+\gamma_{31}^2}{\gamma_{12}\gamma_{23}\gamma_{31}}\Bigr].$$

    It is the Cramér--Rao bound of the closure for $N$ independent circular Gaussian looks.
    """
    s = (g12**2 + g23**2 + g31**2) / (g12 * g23 * g31)
    bracket = 3.0 + 1.0 / g12**2 + 1.0 / g23**2 + 1.0 / g31**2 - 2.0 * np.cos(closure) * s
    return bracket / (2.0 * n_looks)


def dezan_concentration(g12, g23, g31, n_looks, closure=0.0):
    r"""Precision $\kappa_0=1/\sigma^2$ of a cell (the weight $w_\ell$ of the test)."""
    return 1.0 / dezan_closure_variance(g12, g23, g31, n_looks, closure)


def block_sample_covariance(stack, window):
    r"""Sample covariance of every non-overlapping cell of a stack ``(H, W, T)``.

    Cells of ``window = (r, c)`` pixels; cell ``(i, j)`` covers rows ``i*r:(i+1)*r`` and columns
    ``j*c:(j+1)*c`` (incomplete border cells are dropped). Returns ``(H//r, W//c, T, T)``.
    """
    r, c = window
    H, W, T = stack.shape
    Hm, Wm = H // r, W // c
    X = np.asarray(stack[: Hm * r, : Wm * c], dtype=np.complex128)
    X = X.reshape(Hm, r, Wm, c, T).transpose(0, 2, 1, 3, 4).reshape(Hm, Wm, r * c, T)
    return np.einsum("ijna,ijnb->ijab", X, X.conj()) / (r * c)


def block_lag_covariance(stack, window, max_lag=2):
    r"""Entries $[\hat\Sigma_\ell]_{t,t+k}$, $k\le$ ``max_lag``, of every cell of a stack ``(H, W, T)``.

    Returns ``(H//r, W//c, T, max_lag+1)``; ``nan`` where $t+k\ge T$. Enough for every triplet
    of dates at most ``max_lag`` apart, at a memory cost $T(k_{\max}+1)$ instead of $T^2$.
    """
    r, c = window
    H, W, T = stack.shape
    Hm, Wm = H // r, W // c
    X = np.asarray(stack[: Hm * r, : Wm * c], dtype=np.complex64).reshape(Hm, r, Wm, c, T)
    out = np.full((Hm, Wm, T, max_lag + 1), np.nan + 0j, dtype=np.complex128)
    for k in range(max_lag + 1):
        prod = X[..., : T - k] * np.conj(X[..., k:])
        out[:, :, : T - k, k] = prod.sum(axis=(1, 3), dtype=np.complex128) / (r * c)
    return out


def triplet_quantities(lags, n_looks):
    r"""Closure, precision and weakest coherence of every consecutive triplet $(t,t+1,t+2)$.

    ``lags`` has shape ``(..., T, 3)`` (lags 0, 1, 2 of :func:`block_lag_covariance`).
    Returns ``phi, kappa0, gmin``, each ``(..., T-2)``; ``kappa0`` uses the sample coherences
    and the nominal ``n_looks`` (any common factor is irrelevant to the test). Cells with a
    zero power or a nonpositive precision give ``nan``.
    """
    lags = np.asarray(lags)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = lags[..., 0].real.astype(np.float64)
        g12 = np.abs(lags[..., :-2, 1]) / np.sqrt(P[..., :-2] * P[..., 1:-1])
        g23 = np.abs(lags[..., 1:-1, 1]) / np.sqrt(P[..., 1:-1] * P[..., 2:])
        g13 = np.abs(lags[..., :-2, 2]) / np.sqrt(P[..., :-2] * P[..., 2:])
        phi = np.angle(lags[..., :-2, 1] * lags[..., 1:-1, 1] * np.conj(lags[..., :-2, 2]))
        kappa = dezan_concentration(g12, g23, g13, n_looks)
    gmin = np.minimum(np.minimum(g12, g23), g13)
    bad = ~(np.isfinite(phi) & np.isfinite(kappa) & (kappa > 0) & np.isfinite(gmin))
    phi, kappa = np.where(bad, np.nan, phi), np.where(bad, np.nan, kappa)
    return phi, kappa, gmin
