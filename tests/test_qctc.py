import pytest; jax = pytest.importorskip("jax")
import numpy as np, jax.numpy as jnp
from qang.qctc import pctc_map, dctc_map, bloch, state_from_bloch, CNOT, SWAP

def test_cnot_pctc_projects_dctc_decoheres():
    rho = state_from_bloch(jnp.array([0.4, 0.0, -0.6]))
    op, p = pctc_map(rho, CNOT); od, _, _ = dctc_map(rho, CNOT)
    assert abs(float(bloch(op)[2]) - 1) < 1e-13 and abs(float(p) - 0.2) < 1e-14
    assert abs(float(bloch(od)[2]) + 0.6) < 1e-13 and abs(float(bloch(od)[0])) < 1e-13

def test_swap_identity():
    rho = state_from_bloch(jnp.array([0.3, -0.4, 0.5])); op, p = pctc_map(rho, SWAP)
    assert np.allclose(np.asarray(bloch(op)), [0.3, -0.4, 0.5], atol=1e-14) and abs(float(p) - 0.25) < 1e-14
