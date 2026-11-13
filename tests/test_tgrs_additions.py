import numpy as np
from scipy import stats

from closure_glrt.closure import triplet_quantities
from closure_glrt.glrt import glrt_known_scale, known_scale_halfwidth
from closure_glrt.simulation import coherence_matrix, simulate_zones


def test_known_scale_arc_matches_test_inversion():
    rng = np.random.default_rng(0)
    phi = rng.vonmises(0.1, 20.0, size=(50, 25))
    w = rng.uniform(5, 50, size=(50, 25))
    c, alpha = 0.3, 0.05
    h = known_scale_halfwidth(phi, w, c, alpha)
    re, im = (w * np.cos(phi)).sum(-1), (w * np.sin(phi)).sum(-1)
    mu = np.arctan2(im, re)
    q = stats.chi2.ppf(1 - alpha, 1)
    finite = h < np.pi
    for s in (1, -1):
        edge = glrt_known_scale(phi - (mu + s * h)[:, None], w, c)
        assert np.allclose(edge[finite], q, rtol=1e-6)


def _closure_of(lz, n_looks):
    return triplet_quantities(lz, n_looks)[0][..., 0]


def test_simulate_zones_closure_per_cell_and_options():
    rng = np.random.default_rng(1)
    B, zone, window = 400, (1, 2), (20, 20)
    R = coherence_matrix(np.full((B, 2), 0.9), np.full((B, 2), 0.9), np.full((B, 2), 0.9))
    cl = np.radians([5.0, -5.0])
    phi = _closure_of(simulate_zones(R, window, zone, rng, closure=cl), 400)
    assert np.allclose(np.degrees(np.median(phi, 0)), [5, -5], atol=1.0)
    # texture: same closure law centre, shapes preserved
    phi_t = _closure_of(simulate_zones(R, window, zone, rng, closure=cl, texture=1.0), 400)
    assert phi_t.shape == (B, 2)
    assert np.allclose(np.degrees(np.median(phi_t, 0)), [5, -5], atol=1.5)
    # second population with lower coherence: lower sample coherence, closure kept
    Ra = coherence_matrix(np.full((B, 2), 0.4), np.full((B, 2), 0.4), np.full((B, 2), 0.4))
    lz = simulate_zones(R, window, zone, rng, closure=cl, R_alt=Ra)
    g = np.abs(lz[..., 0, 1]) / np.sqrt(lz[..., 0, 0].real * lz[..., 1, 0].real)
    assert 0.5 < np.median(g) < 0.8
    assert np.allclose(np.degrees(np.median(_closure_of(lz, 400), 0)), [5, -5], atol=2.0)


def test_neighbor_concentration_excludes_the_cell():
    from closure_glrt.closure import dezan_concentration, neighbor_concentration
    rng = np.random.default_rng(2)
    B, zone, window = 50, (3, 3), (15, 5)
    g = np.full((B, 9), 0.9)
    g[:, 4] = 0.5                         # centre cell less coherent
    R = coherence_matrix(g, g, g * g)
    lz = simulate_zones(R, window, zone, rng)
    k = neighbor_concentration(lz, 75, zone)
    assert k.shape == (B, 9)
    # the centre cell sees only its coherent neighbours, corners see the incoherent centre
    k_hi = dezan_concentration(0.9, 0.9, 0.81, 75)
    assert np.median(k[:, 4]) > 0.5 * k_hi
    assert np.median(k[:, 4]) > np.median(k[:, 0])
