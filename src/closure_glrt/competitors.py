r"""Closure tests of the literature, applied to a zone as one multilooked cell.

Both tests below act on the closure of a single multilooked covariance. To compare them with the
scale-free test on the same zone, the lagged covariances of the $L$ cells of the zone are
averaged (the zone is multilooked as one cell of $LN$ pixels), as both papers do when they
average pixels into larger cells.

* Zwieback et al. (IEEE TGRS 2016): Wald test of a zero closure with the linearized variance of
  the closure at the nominal number of looks (independent looks). For one triplet the statistic is
  $\hat\Phi^2/\sigma^2_{\mathrm{DZ}}(\hat\gamma;LN)\sim\chi^2_1$ under $H_0$, where
  $\sigma^2_{\mathrm{DZ}}$ is De Zan's first-order variance (:func:`dezan_closure_variance`).
* Yuan and Lopez-Dekker (IEEE TGRS 2026): significance ratio $\hat\Phi/\sigma$ compared with the
  normal quantile, with $\sigma=\sigma_n(\hat\gamma)\sqrt{2048/N_l}$, $\sigma_n$ read in a Monte
  Carlo look-up table built with 2048 independent looks and $N_l$ the equivalent number of looks
  of the multilooked area (their eq. (18) and Appendix B). Here $N_l=LN/A$, with $A$ the
  correlation area measured on the images in place of their system-derived oversampling factors.

All functions are numpy (host side).
"""

import numpy as np
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from closure_glrt.closure import dezan_closure_variance

__all__ = ["zone_multilook", "zwieback_zone", "yuan_lut", "YuanLUT", "yuan_zone"]

YUAN_N = 2048


def zone_multilook(lz):
    r"""Closure and the three coherence magnitudes of the zone multilooked as one cell.

    ``lz`` ``(..., L, 3, 3)`` lagged covariances of one triplet (scan layout ``[t, k]``) of the
    cells of each zone, all cells with the same number of pixels. Returns ``phi, g12, g23, g13``
    of shape ``(...)``.
    """
    with np.errstate(invalid="ignore"):
        m = np.asarray(lz).mean(axis=-3)          # nan only on the unused entries t + k > 2
    P = m[..., 0].real
    g12 = np.abs(m[..., 0, 1]) / np.sqrt(P[..., 0] * P[..., 1])
    g23 = np.abs(m[..., 1, 1]) / np.sqrt(P[..., 1] * P[..., 2])
    g13 = np.abs(m[..., 0, 2]) / np.sqrt(P[..., 0] * P[..., 2])
    phi = np.angle(m[..., 0, 1] * m[..., 1, 1] * np.conj(m[..., 0, 2]))
    return phi, g12, g23, g13


def zwieback_zone(phi, g12, g23, g13, n_looks, alpha=None):
    r"""Wald statistic $\hat\Phi^2/\sigma^2_{\mathrm{DZ}}$ (compare with $\chi^2_1$), and, with
    ``alpha``, the half-width $z_{1-\alpha/2}\sigma_{\mathrm{DZ}}$ of the matching interval."""
    var = dezan_closure_variance(g12, g23, g13, n_looks)
    stat = phi ** 2 / var
    if alpha is None:
        return stat
    return stat, np.minimum(stats.norm.ppf(1 - alpha / 2) * np.sqrt(var), np.pi)


def yuan_lut(rng, step=0.05, reps=1024, n=YUAN_N):
    r"""Monte Carlo table of the standard deviation $\sigma_n$ of the closure with ``n`` independent
    looks, on a grid of the three coherences (sorted triples, filled by symmetry).

    Returns ``(grid, table)``: ``grid`` (G,) coherence values, ``table`` (G, G, G) $\sigma_n$ in
    radians (nan where the coherence matrix is not positive definite).
    """
    grid = np.arange(step / 2, 1.0, step)
    G = len(grid)
    table = np.full((G, G, G), np.nan)
    for i in range(G):
        for j in range(i, G):
            for k in range(j, G):
                a, b, c = grid[i], grid[j], grid[k]
                if 1 - a * a - b * b - c * c + 2 * a * b * c <= 1e-9:
                    continue
                S = np.array([[1, a, c], [a, 1, b], [c, b, 1]], dtype=np.complex128)
                Lc = np.linalg.cholesky(S)
                w = (rng.standard_normal((reps, 3, n)) + 1j * rng.standard_normal((reps, 3, n)))
                x = Lc @ (w / np.sqrt(2))
                s12 = (x[:, 0] * x[:, 1].conj()).sum(-1)
                s23 = (x[:, 1] * x[:, 2].conj()).sum(-1)
                s13 = (x[:, 0] * x[:, 2].conj()).sum(-1)
                sd = np.std(np.angle(s12 * s23 * s13.conj()))
                for p in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
                    table[p] = sd
    return grid, table


class YuanLUT:
    """Trilinear interpolation of :func:`yuan_lut` (nearest valid node outside the valid set)."""

    def __init__(self, grid, table):
        self.grid = np.asarray(grid)
        filled = np.asarray(table, dtype=np.float64).copy()
        bad = ~np.isfinite(filled)
        if bad.any():   # invalid nodes (not positive definite) take the nearest valid value
            idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
            filled = filled[tuple(idx)]
        self._f = RegularGridInterpolator((self.grid,) * 3, filled, bounds_error=False,
                                          fill_value=None)

    def __call__(self, g12, g23, g13):
        lo, hi = self.grid[0], self.grid[-1]
        pts = np.stack([np.clip(np.asarray(g, dtype=np.float64), lo, hi)
                        for g in (g12, g23, g13)], -1)
        return self._f(pts.reshape(-1, 3)).reshape(pts.shape[:-1])


def yuan_zone(phi, g12, g23, g13, enl, lut, alpha=None):
    r"""Significance ratio $|\hat\Phi|/\sigma$, $\sigma=\sigma_n(\hat\gamma)\sqrt{2048/N_l}$
    (compare with $z_{1-P_{fa}/2}$), and, with ``alpha``, the half-width $z_{1-\alpha/2}\sigma$."""
    sigma = lut(g12, g23, g13) * np.sqrt(YUAN_N / np.asarray(enl, dtype=np.float64))
    ratio = np.abs(phi) / sigma
    if alpha is None:
        return ratio
    return ratio, np.minimum(stats.norm.ppf(1 - alpha / 2) * sigma, np.pi)
