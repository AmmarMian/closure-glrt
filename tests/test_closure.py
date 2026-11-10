import numpy as np
import pytest
from conftest import BACKEND_DTYPE_PARAMS

from closure_glrt.backend import get_data_on_device, to_dtype, to_numpy
from closure_glrt.closure import (
    block_lag_covariance,
    block_sample_covariance,
    closure_phase,
    dezan_concentration,
    sample_coherence_matrix,
    triplet_quantities,
)
from closure_glrt.simulation import coherence_matrix, simulate_cells


def _on(x, backend_name, dtype):
    """numpy complex128 -> backend array of complex ``dtype`` (the HDR way: move, then cast)."""
    return to_dtype(get_data_on_device(x, backend_name), np.dtype(dtype), backend_name)


def _tol(dtype):
    return 1e-10 if dtype == "complex128" else 2e-5


def test_population_closure():
    S = coherence_matrix(0.8, 0.7, 0.6, closure=0.4)
    assert closure_phase(S) == pytest.approx(0.4)


@pytest.mark.parametrize("backend_name,dtype", BACKEND_DTYPE_PARAMS)
def test_closure_backend_and_gauge(backend_name, dtype):
    """Same closures as numpy complex128 on every backend; dtype follows the data; gauge invariant."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 50, 3)) + 1j * rng.standard_normal((200, 50, 3))
    ref = closure_phase(sample_coherence_matrix(X))
    Xb = _on(X, backend_name, dtype)
    G = sample_coherence_matrix(Xb, backend_name)
    assert str(G.dtype).endswith(dtype)
    phi = closure_phase(G, backend_name=backend_name)
    assert str(phi.dtype).endswith("float32" if dtype == "complex64" else "float64")
    assert np.allclose(to_numpy(phi), ref, atol=50 * _tol(dtype))
    gauge = _on(np.exp(1j * np.array([0.3, -1.2, 2.0])), backend_name, dtype)
    phig = closure_phase(sample_coherence_matrix(Xb * gauge, backend_name), backend_name=backend_name)
    assert np.allclose(to_numpy(phig), to_numpy(phi), atol=50 * _tol(dtype))


@pytest.mark.parametrize("g,N", [(0.8, 256), (0.6, 128)])
def test_dezan_variance_monte_carlo(g, N):
    """1/Var(closure) matches kappa0 within a few percent at high coherence and many looks."""
    rng = np.random.default_rng(1)
    S = np.broadcast_to(coherence_matrix(g, g, g), (20000, 3, 3))
    X, _ = simulate_cells(S, (N, 1), rng)
    phi = closure_phase(sample_coherence_matrix(X))
    assert 1 / phi.var() == pytest.approx(dezan_concentration(g, g, g, N), rel=0.05)


@pytest.mark.parametrize("backend_name,dtype", BACKEND_DTYPE_PARAMS)
def test_lag_covariance_and_triplets(backend_name, dtype):
    rng = np.random.default_rng(2)
    stack = rng.standard_normal((12, 10, 5)) + 1j * rng.standard_normal((12, 10, 5))
    full = block_sample_covariance(stack, (3, 2))
    lags_np = block_lag_covariance(stack, (3, 2), max_lag=2)
    for k in range(3):
        for t in range(5 - k):
            assert np.allclose(lags_np[..., t, k], full[..., t, t + k])
        assert np.all(np.isnan(lags_np[..., 5 - k:, k]))
    sb = _on(stack, backend_name, dtype)
    lags = block_lag_covariance(sb, (3, 2), max_lag=2, backend_name=backend_name)
    assert str(lags.dtype).endswith(dtype)
    assert np.allclose(to_numpy(lags), lags_np, atol=10 * _tol(dtype), equal_nan=True)
    assert np.allclose(to_numpy(block_sample_covariance(sb, (3, 2), backend_name)), full,
                       atol=10 * _tol(dtype))
    phi, kappa, _ = triplet_quantities(lags, 6, backend_name)
    phi_ref, kappa_ref, _ = triplet_quantities(lags_np, 6)
    assert np.allclose(to_numpy(phi), phi_ref, atol=100 * _tol(dtype))
    assert np.allclose(to_numpy(kappa), kappa_ref, rtol=1e3 * _tol(dtype))
    d = np.sqrt(np.einsum("...aa->...a", full).real)
    G = full / (d[..., :, None] * d[..., None, :])
    a = np.abs(G)
    assert np.allclose(phi_ref[..., 0], closure_phase(G, 0, 1, 2))
    assert np.allclose(kappa_ref[..., 1], dezan_concentration(a[..., 1, 2], a[..., 2, 3],
                                                              a[..., 1, 3], 6))
