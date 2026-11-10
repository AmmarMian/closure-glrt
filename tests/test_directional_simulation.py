import numpy as np
import pytest

from closure_glrt.closure import closure_phase, dezan_concentration, sample_coherence_matrix
from closure_glrt.directional import WrappedNormal, ks_uniform, leave_one_out_pit, vm_cdf
from closure_glrt.simulation import (
    coherence_matrix,
    correlation_area,
    effective_looks,
    moving_average_kernel,
    simulate_cells,
)


def test_vm_cdf_support():
    assert vm_cdf(-np.pi, 3.0) == pytest.approx(0.0, abs=1e-12)
    assert vm_cdf(np.pi, 3.0) == pytest.approx(1.0)
    assert vm_cdf(0.0, 3.0) == pytest.approx(0.5)


def test_wrapped_normal_cdf():
    wn = WrappedNormal(0.0, 1.5)
    assert wn.cdf(np.pi) == pytest.approx(1.0)
    assert wn.cdf(0.0) == pytest.approx(0.5)


def test_pit_uniform_under_model():
    rng = np.random.default_rng(0)
    w = rng.uniform(5, 50, (3000, 25))
    phi = rng.vonmises(0.3, 0.4 * w)
    u, _ = leave_one_out_pit(phi, w, "vm_c")
    # Not exactly uniform even when the law holds: mu_-l and c_-l are estimated on the other
    # cells (KS about 0.016 here). Real data are compared with this simulated baseline.
    assert ks_uniform(u) < 0.03
    u1, _ = leave_one_out_pit(phi, w, "vm_1")
    assert ks_uniform(u1) > 0.05  # c = 0.4, so the unscaled weights are wrong


def test_correlation_area_moving_average():
    """Infinite-window limit ((2s^2+1)/(3s))^2 is approached by a large window."""
    s = 3
    A = correlation_area(moving_average_kernel(s), (200, 200))
    assert A == pytest.approx(((2 * s**2 + 1) / (3 * s)) ** 2, rel=0.02)


def test_correlated_looks_scale():
    """Var(closure) * kappa0(nominal) ~ A for correlated looks (Prop. separable correlation)."""
    rng = np.random.default_rng(1)
    S = np.broadcast_to(coherence_matrix(0.9, 0.9, 0.81), (20000, 3, 3))
    window = (16, 16)
    k = moving_average_kernel(2)
    X, _ = simulate_cells(S, window, rng, kernel=k)
    phi = closure_phase(sample_coherence_matrix(X))
    A = correlation_area(k, window)
    ratio = phi.var() * dezan_concentration(0.9, 0.9, 0.81, 256)
    assert ratio == pytest.approx(A, rel=0.08)


def test_effective_looks_white():
    tau = np.ones((4, 10))
    assert np.allclose(effective_looks(tau), 10)
