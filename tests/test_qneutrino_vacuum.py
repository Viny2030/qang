import pytest; jax = pytest.importorskip("jax")
import numpy as np, jax.numpy as jnp
from qang.qneutrino import osc_phase_qga, survival_qz, pmns, p3_vacuum, sector_filter, embed_two_qubits

def test_one_turn_is_one_oscillation_length():
    dm2, E = 7.42e-5, 2.5; L = 4 * np.pi * E / (2 * 1.267 * 2 * dm2)
    assert abs(float(osc_phase_qga(dm2, L, E)) - 1000.0) < 1e-9

def test_unitarity_three_flavors():
    U = pmns(0.583, 0.1496, 0.855, 3.4); p = p3_vacuum(U, jnp.array([0.0, 7.42e-5, 2.51e-3]), 1300.0, 2.5, 1)
    assert abs(float(p.sum()) - 1) < 1e-12

def test_sector_filter_removes_11():
    q = sector_filter(jnp.array([0.4, 0.3, 0.2, 0.1])); assert float(q[3]) == 0.0 and abs(float(q.sum()) - 1) < 1e-15
