import pytest; jax = pytest.importorskip("jax")
import numpy as np, jax.numpy as jnp
from qang.qweightqnn import unary_loader, weight_qnn, weight_mask, qsp_readout

def test_loader_signs():
    x = jnp.array([0.5, -0.5, 0.5, -0.5]); psi = unary_loader(x, 4)
    assert np.allclose(np.asarray(jnp.real(psi[np.array([8, 4, 2, 1])])), np.asarray(x), atol=1e-13)

def test_weight_conserved():
    psi = weight_qnn(jnp.array([123., 456., 789.]), unary_loader(jnp.array([1., 0, 0, 0]), 4), 4, ((0, 1), (2, 3), (1, 2)))
    assert float(jnp.sum(jnp.abs(psi) ** 2 * (~weight_mask(4, 1)))) < 1e-14

def test_qsp_chebyshev():
    a = 0.3; assert abs(float(qsp_readout(jnp.zeros(4), a)) - np.cos(3 * np.arccos(2 * a * a - 1))) < 1e-12
