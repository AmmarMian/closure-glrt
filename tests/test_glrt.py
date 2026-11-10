import numpy as np
import pytest
from scipy.special import i0e

from closure_glrt.glrt import (
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


def test_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(6)
    phi, w = _zones(rng, 300, 25, 10, c=0.5, mu=0.05)
    lam_np = glrt(phi, w)
    lam_t = glrt(torch.from_numpy(phi), torch.from_numpy(w)).numpy()
    assert np.allclose(lam_np, lam_t, rtol=1e-6, atol=1e-8)
    h_t = confidence_halfwidth(torch.from_numpy(phi), torch.from_numpy(w)).numpy()
    assert np.allclose(confidence_halfwidth(phi, w), h_t, rtol=1e-6, atol=1e-8)
