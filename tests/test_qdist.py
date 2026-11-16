import pytest; jax = pytest.importorskip("jax")
import numpy as np, jax.numpy as jnp
from qang.qdist import qg_moments, probs_from_moments, hellinger2_qg, haar_posterior_qz, shots_tv_bound

def test_walsh_roundtrip():
    p = jnp.array(np.random.default_rng(0).dirichlet(np.ones(8)))
    assert np.allclose(probs_from_moments(qg_moments(p, 3), 3), p, atol=1e-14)

def test_hellinger_geodesic():
    th, th2 = 0.3, 1.9
    assert abs(float(hellinger2_qg(np.cos(th), np.cos(th2))) - (1 - np.cos(abs(th - th2) / 2))) < 1e-14

def test_haar_posterior_equator():
    m, v = haar_posterior_qz(jnp.array(50.0), jnp.array(100.0))
    assert abs(float(m)) < 1e-14 and abs(float(v) - 1 / 103) < 1e-14

def test_shots_bound():
    assert abs(shots_tv_bound(8, 36, 0.05) - 3600.0) < 1e-6
