import numpy as np
import pytest

from closure_glrt.closure import (
    block_lag_covariance,
    block_sample_covariance,
    closure_phase,
    dezan_concentration,
    sample_coherence_matrix,
    triplet_quantities,
)
from closure_glrt.simulation import coherence_matrix, simulate_cells


def test_closure_gauge_invariant():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3)) + 1j * rng.standard_normal((50, 3))
    G = sample_coherence_matrix(X)
    Xg = X * np.exp(1j * np.array([0.3, -1.2, 2.0]))
    assert closure_phase(sample_coherence_matrix(Xg)) == pytest.approx(closure_phase(G))
    assert np.allclose(np.diag(G), 1.0)


def test_population_closure():
    S = coherence_matrix(0.8, 0.7, 0.6, closure=0.4)
    assert closure_phase(S) == pytest.approx(0.4)


@pytest.mark.parametrize("g,N", [(0.8, 256), (0.6, 128)])
def test_dezan_variance_monte_carlo(g, N):
    """1/Var(closure) matches kappa0 within a few percent at high coherence and many looks."""
    rng = np.random.default_rng(1)
    S = np.broadcast_to(coherence_matrix(g, g, g), (20000, 3, 3))
    X, _ = simulate_cells(S, (N, 1), rng)
    phi = closure_phase(sample_coherence_matrix(X))
    k0 = dezan_concentration(g, g, g, N)
    assert 1 / phi.var() == pytest.approx(k0, rel=0.05)


def test_lag_covariance_matches_full():
    rng = np.random.default_rng(2)
    stack = rng.standard_normal((12, 10, 5)) + 1j * rng.standard_normal((12, 10, 5))
    full = block_sample_covariance(stack, (3, 2))
    lags = block_lag_covariance(stack, (3, 2), max_lag=2)
    for k in range(3):
        for t in range(5 - k):
            assert np.allclose(lags[..., t, k], full[..., t, t + k])
    phi, kappa, gmin = triplet_quantities(lags, 6)
    G = full / np.sqrt(np.einsum("...aa->...a", full).real[..., :, None]
                       * np.einsum("...aa->...a", full).real[..., None, :])
    assert np.allclose(phi[..., 0], closure_phase(G, 0, 1, 2))
    g = np.abs(G)
    assert np.allclose(kappa[..., 1], dezan_concentration(g[..., 1, 2], g[..., 2, 3], g[..., 1, 3], 6))
