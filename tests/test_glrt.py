import numpy as np
import pytest
from conftest import BACKEND_DTYPE_PARAMS
from scipy.special import i0e

from closure_glrt.backend import get_data_on_device, to_dtype, to_numpy
from closure_glrt.glrt import (
    ScaleFreeClosureGLRT,
    confidence_halfwidth,
    glrt,
    legendre,
    pvalue,
    scale_ml,
    threshold,
    weighted_resultant,
)


def _zones(rng, n, L, kbar, c=1.0, mu=0.0, spread=10.0):
    w = kbar * np.exp(rng.uniform(-0.5, 0.5, (n, L)) * np.log(spread))
    phi = rng.vonmises(mu, c * w)
    return phi, w


def _gstar_brute(X, w):
    c = np.geomspace(1e-6, 1e6, 200001)
    g = np.sum(np.log(i0e(np.outer(c, w))) + np.outer(c, w), axis=1)
    return max(0.0, np.max(c * X - g))


def test_legendre_matches_grid():
    rng = np.random.default_rng(0)
    w = rng.uniform(1, 20, 15)
    for X in (0.3 * w.sum(), 0.8 * w.sum(), 0.97 * w.sum()):
        assert legendre(X, w) == pytest.approx(_gstar_brute(X, w), rel=1e-5, abs=1e-8)
    assert legendre(-1.0, w) == 0.0


def test_scale_root_and_limits():
    w = np.array([1.0, 2.0, 5.0])
    c = scale_ml(np.array([4.0]), w[None])
    from scipy.special import i1e
    assert np.sum(w * i1e(c * w) / i0e(c * w)) == pytest.approx(4.0)
    assert scale_ml(np.array(0.0), w) == 0.0
    assert np.isinf(scale_ml(np.array(8.0), w))


def test_invariant_to_weight_scale():
    rng = np.random.default_rng(1)
    phi, w = _zones(rng, 200, 20, 30)
    lam = glrt(phi, w)
    for b in (1e-2, 7.3, 1e3):
        assert np.allclose(glrt(phi, b * w), lam, rtol=1e-6, atol=1e-8)


def test_glrt_equals_profile_likelihood():
    """Lambda equals twice the gap of the profiled likelihoods, by brute force over (mu, c)."""
    rng = np.random.default_rng(2)
    phi, w = _zones(rng, 1, 12, 20, c=0.5, mu=0.15)
    phi, w = phi[0], w[0]
    mus = np.linspace(-np.pi, np.pi, 4001)
    cs = np.geomspace(1e-3, 1e2, 3001)

    def ll(mu, c):
        x = np.multiply.outer(c, w)
        return (x * np.cos(phi - mu)).sum(-1) - (np.log(i0e(x)) + x).sum(-1)

    l1 = max(ll(m, cs).max() for m in mus)
    l0 = ll(0.0, cs).max()
    assert glrt(phi, w) == pytest.approx(2 * (l1 - l0), rel=1e-3)


@pytest.mark.parametrize("L", [10, 30])
@pytest.mark.parametrize("c", [0.25, 1.0])
def test_false_alarm_rate(L, c):
    """F threshold keeps 1 % at moderate concentration, for every scale c."""
    rng = np.random.default_rng(3)
    phi, w = _zones(rng, 20000, L, 20, c=c)
    rate = np.mean(pvalue(glrt(phi, w), L) < 0.01)
    assert abs(rate - 0.01) < 0.0025  # 2 sigma = 0.0014


def test_threshold_and_pvalue_agree():
    for L in (5, 25, 100):
        eta = threshold(L, 0.01)
        assert pvalue(np.array(eta), L) == pytest.approx(0.01)


def test_confidence_arc_matches_test_inversion():
    """The arc is the set of mu0 not rejected by the exact test of mu = mu0."""
    rng = np.random.default_rng(4)
    phi, w = _zones(rng, 20, 25, 15, c=0.5, mu=0.2)
    h = confidence_halfwidth(phi, w, alpha=0.05)
    _, mu, _, _ = weighted_resultant(phi, w)
    eta = threshold(25, 0.05)
    for z in range(20):
        for side in (-1, 1):
            inside = glrt(phi[z] - (mu[z] + side * 0.99 * h[z]), w[z])
            outside = glrt(phi[z] - (mu[z] + side * 1.01 * h[z]), w[z])
            if h[z] < np.pi:
                assert inside <= eta + 1e-6 and outside >= eta - 1e-6


def test_confidence_coverage():
    rng = np.random.default_rng(5)
    mu_true = 0.1
    phi, w = _zones(rng, 4000, 25, 30, c=0.4, mu=mu_true)
    h = confidence_halfwidth(phi, w, alpha=0.05)
    _, mu, _, _ = weighted_resultant(phi, w)
    cover = np.abs(np.angle(np.exp(1j * (mu - mu_true)))) <= h
    assert abs(cover.mean() - 0.95) < 0.015


@pytest.mark.parametrize("kbar", [2.0, 50.0, 2000.0])
@pytest.mark.parametrize("backend_name,dtype", BACKEND_DTYPE_PARAMS)
def test_backends_agree_with_numpy(backend_name, dtype, kbar):
    """Statistic, scales and arc agree with numpy float64 on every backend, also in float32."""
    real = "float32" if dtype == "complex64" else "float64"
    rng = np.random.default_rng(6)
    phi, w = _zones(rng, 500, 25, kbar, c=0.5, mu=0.02)
    ref = glrt(phi, w, details=True)
    h_ref = confidence_halfwidth(phi, w)
    P, Wt = (to_dtype(get_data_on_device(x, backend_name), np.dtype(real), backend_name)
             for x in (phi, w))
    out = glrt(P, Wt, backend_name, details=True)
    assert str(out["lam"].dtype).endswith(real)
    tol = 1e-6 if real == "float64" else 2e-3
    lam = to_numpy(out["lam"])
    assert np.allclose(lam, ref["lam"], rtol=tol, atol=10 * tol)
    assert np.allclose(to_numpy(out["c1"]), ref["c1"], rtol=10 * tol)
    agree = (pvalue(lam, 25) < 0.01) == (pvalue(ref["lam"], 25) < 0.01)
    assert agree.mean() > 0.995
    h = to_numpy(confidence_halfwidth(P, Wt, backend_name=backend_name))
    assert np.allclose(h, h_ref, rtol=10 * tol, atol=tol)


def test_detector_class():
    rng = np.random.default_rng(7)
    phi, w = _zones(rng, 10, 25, 20)
    det = ScaleFreeClosureGLRT(25)
    assert np.allclose(det.compute(phi, w), glrt(phi, w))
    assert det.get_threshold(0.01) == pytest.approx(float(threshold(25, 0.01)))
