r"""Simulation of multilook cells of three acquisitions.

Snapshot model of a cell of $r\times c$ pixels, pixel $n$:

$$x_n=\sqrt{\tau_n}\,\boldsymbol L\,\boldsymbol s_n,\qquad \boldsymbol\Sigma=\boldsymbol L\boldsymbol L^{\mathsf H},$$

* $\boldsymbol s_n\in\mathbb C^3$: three independent unit-variance circular Gaussian fields
  (one per date), white or filtered by a 2-D kernel $k$ (spatial correlation of the looks,
  the same for every date: the separable model, which gives $c=1/A$);
* $\tau_n>0$: texture, common to the three dates, $\mathbb E\tau=1$ (compound-Gaussian
  clutter; $\tau\equiv1$ is Gaussian speckle);
* $\boldsymbol\Sigma$: coherence matrix of the cell; its closure is the true closure.

Data are generated in numpy (complex128), as in HDR: the caller moves them to a backend with
:func:`closure_glrt.backend.get_data_on_device`, which gives complex64 on torch-mps; casting
to complex64 elsewhere is done with :func:`closure_glrt.backend.to_dtype`.

First-order predictions for the scale of the test (relative to the nominal $N=rc$):
correlated looks $c=1/A$, $A=\frac1N\sum_{n,m}|\rho_{nm}|^2$; texture $c=N_{\mathrm{eff}}/N$
with $N_{\mathrm{eff}}=(\sum_n\tau_n)^2/\sum_{n,m}\tau_n\tau_m|\rho_{nm}|^2$ (to be checked).
"""

import numpy as np

__all__ = [
    "kernel_from_spectrum",
    "simulate_zones",
    "coherence_matrix",
    "moving_average_kernel",
    "correlation_area",
    "simulate_cells",
    "effective_looks",
]


def coherence_matrix(g12, g23, g13, closure=0.0):
    r"""Coherence matrices ``(..., 3, 3)`` with magnitudes $\gamma_{12},\gamma_{23},\gamma_{13}$
    and closure $\Phi$ put on the phase of entry (1, 2): $\arg(\Sigma_{12}\Sigma_{23}\Sigma_{31})=\Phi$."""
    g12, g23, g13, closure = np.broadcast_arrays(*(np.asarray(x, dtype=np.float64)
                                                    for x in (g12, g23, g13, closure)))
    S = np.zeros(g12.shape + (3, 3), dtype=np.complex128)
    S[..., 0, 0] = S[..., 1, 1] = S[..., 2, 2] = 1.0
    S[..., 0, 1] = g12 * np.exp(1j * closure)
    S[..., 1, 2] = g23
    S[..., 0, 2] = g13
    S[..., 1, 0] = np.conj(S[..., 0, 1])
    S[..., 2, 1] = S[..., 1, 2]
    S[..., 2, 0] = S[..., 0, 2]
    return S


def moving_average_kernel(s):
    """Separable s x s moving-average kernel (unnormalised)."""
    return np.ones((s, s))


def correlation_area(kernel, window):
    r"""$A=\frac1N\sum_{n,m\in W}|\rho(n-m)|^2$ for white noise filtered by ``kernel``."""
    k = np.asarray(kernel, dtype=np.complex128)
    acf = _autocorrelation(k)
    r, c = window
    ky, kx = k.shape
    total = 0.0
    for dy in range(-(r - 1), r):
        for dx in range(-(c - 1), c):
            if abs(dy) < ky and abs(dx) < kx:
                total += (r - abs(dy)) * (c - abs(dx)) * abs(acf[dy + ky - 1, dx + kx - 1]) ** 2
    return total / (r * c)


def _autocorrelation(k):
    """Normalised 2-D autocorrelation of a kernel, lags -(K-1)..(K-1)."""
    from scipy.signal import correlate

    a = correlate(k, k, mode="full")
    return a / a[k.shape[0] - 1, k.shape[1] - 1]


def _filtered_noise(rng, shape, kernel):
    """Unit-variance circular Gaussian fields of ``shape = (..., r, c)``, filtered by ``kernel``."""
    from scipy.signal import fftconvolve

    r, c = shape[-2:]
    if kernel is None:
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
    k = np.asarray(kernel, dtype=np.complex128)
    ky, kx = k.shape
    big = shape[:-2] + (r + ky - 1, c + kx - 1)
    z = (rng.standard_normal(big) + 1j * rng.standard_normal(big)) / np.sqrt(2)
    k = k / np.sqrt(np.sum(np.abs(k) ** 2))
    axes = (len(big) - 2, len(big) - 1)
    out = fftconvolve(z, k.reshape((1,) * (len(big) - 2) + k.shape), mode="valid", axes=axes)
    return out


def simulate_cells(Sigma, window, rng, kernel=None, texture=None):
    r"""Pixels of independent cells with coherence matrices ``Sigma`` of shape ``(..., 3, 3)``.

    Parameters
    ----------
    Sigma : array (..., 3, 3), Hermitian positive definite.
    window : (r, c) pixels per cell.
    rng : numpy Generator.
    kernel : 2-D array or None. Spatial filter of the white fields (same for every date).
    texture : None, float or callable. None: Gaussian speckle. A float ``nu``: Gamma texture of
        shape ``nu`` and mean 1 (K-distributed amplitude), drawn per pixel and common to the dates.
        A callable ``texture(rng, shape)`` returns $\tau$ of shape ``(..., r, c)``.

    Returns
    -------
    X : array (..., r*c, 3) complex128, looks of each cell (for :func:`sample_coherence_matrix`).
    tau : array (..., r*c) of the texture (ones for Gaussian speckle).
    """
    Sigma = np.asarray(Sigma, dtype=np.complex128)
    batch = Sigma.shape[:-2]
    r, c = window
    s = np.stack([_filtered_noise(rng, batch + (r, c), kernel) for _ in range(3)], axis=-1)
    s = s.reshape(batch + (r * c, 3))
    Lc = np.linalg.cholesky(Sigma)
    X = np.einsum("...ab,...nb->...na", Lc, s)
    tau = _draw_texture(rng, texture, batch + (r, c)).reshape(batch + (r * c,))
    return X * np.sqrt(tau)[..., None], tau


def _draw_texture(rng, texture, shape):
    """Texture tau of ``shape``: ones, Gamma(nu, 1/nu) for a float nu, or ``texture(rng, shape)``."""
    if texture is None:
        return np.ones(shape)
    if callable(texture):
        return np.asarray(texture(rng, shape), dtype=np.float64)
    nu = float(texture)
    return rng.gamma(nu, 1.0 / nu, size=shape)


def effective_looks(tau, rho=None):
    r"""$N_{\mathrm{eff}}=(\sum_n\tau_n)^2/\sum_{n,m}\tau_n\tau_m|\rho_{nm}|^2$ per cell.

    ``tau`` (..., N); ``rho`` (N, N) pixel correlation matrix or None (independent looks,
    $N_{\mathrm{eff}}=(\sum\tau)^2/\sum\tau^2$).
    """
    tau = np.asarray(tau, dtype=np.float64)
    if rho is None:
        return tau.sum(-1) ** 2 / (tau**2).sum(-1)
    R2 = np.abs(np.asarray(rho)) ** 2
    return tau.sum(-1) ** 2 / np.einsum("...n,nm,...m->...", tau, R2, tau)


def kernel_from_spectrum(s, taps: int):
    r"""Zero-phase complex kernel of $2\,$``taps``$+1$ samples with $|\mathrm{FFT}\,h|^2\propto s$, unit energy.

    ``s`` is a 1-D power spectrum (``s_az`` or ``s_rg`` of the spectrum store); filtering white
    noise along one axis with $h$ gives speckle with that spectrum along this axis.
    """
    h = np.fft.ifft(np.sqrt(np.clip(np.asarray(s, dtype=np.float64), 0, None)))
    h = np.roll(h, taps)[: 2 * taps + 1]
    return h / np.sqrt((np.abs(h) ** 2).sum())


def _separable_speckle(rng, n, shape, h_az, h_rg):
    """(n, 3, H, W) unit-variance speckle filtered along rows (az) by h_az and columns (rg) by h_rg."""
    ka, kr = len(h_az), len(h_rg)
    H, W = shape
    g = (rng.standard_normal((n, 3, H + ka - 1, W + kr - 1))
         + 1j * rng.standard_normal((n, 3, H + ka - 1, W + kr - 1))) / np.sqrt(2)
    x = sum(h_az[i] * g[:, :, i:i + H] for i in range(ka))
    return sum(h_rg[j] * x[:, :, :, j:j + W] for j in range(kr))


def simulate_zones(R, window, zone, rng, h_az=None, h_rg=None, closure=0.0):
    r"""Lagged cell covariances of whole zones of adjacent cells, as in the scan store.

    Parameters
    ----------
    R : array (B, L, 3, 3), real coherence matrices of the $L=z_az_b$ cells of each zone
        (e.g. built from the sample coherences of real cells with :func:`coherence_matrix`).
    window : (r, c) pixels per cell; zone : (z_a, z_b) cells per zone.
    rng : numpy Generator. h_az, h_rg : kernels of :func:`kernel_from_spectrum` (None: white).
    closure : true closure (rad) put on entry (1, 2) of every cell.

    One field per zone and date is filtered, then cut into the cells, so neighbouring cells
    share correlated pixels as on the image; the looks of each cell are mixed over the dates by
    the Cholesky factor of its matrix. Returns ``(B, L, 3, 3)`` complex128 with the scan layout
    ``[..., t, k]`` $=[\hat\Sigma]_{t,t+k}$ (``nan`` where $t+k>2$), ready for
    :func:`closure_glrt.closure.triplet_quantities`. Generated in numpy (HDR convention).
    """
    R = np.asarray(R, dtype=np.complex128).copy()
    B, L = R.shape[:2]
    r, c = window
    za, zb = zone
    h_az = np.ones(1) if h_az is None else np.asarray(h_az)
    h_rg = np.ones(1) if h_rg is None else np.asarray(h_rg)
    x = _separable_speckle(rng, B, (za * r, zb * c), h_az, h_rg)          # (B, 3, za r, zb c)
    x = x.reshape(B, 3, za, r, zb, c).transpose(0, 2, 4, 1, 3, 5).reshape(B, L, 3, r * c)
    R[..., 0, 1] *= np.exp(1j * closure)
    R[..., 1, 0] *= np.exp(-1j * closure)
    z = np.linalg.cholesky(R) @ x                                           # (B, L, 3, N)
    S = z @ np.swapaxes(z.conj(), -1, -2) / (r * c)
    out = np.full((B, L, 3, 3), np.nan + 0j)
    for t in range(3):
        out[..., t, 0] = S[..., t, t]
    out[..., 0, 1], out[..., 1, 1], out[..., 0, 2] = S[..., 0, 1], S[..., 1, 2], S[..., 0, 2]
    return out
