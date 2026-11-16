import pytest; jax = pytest.importorskip("jax")
import numpy as np, jax.numpy as jnp
from qang.qneutrino import pmns, p3_vacuum
from qang.qmajorana import pmns_majorana, p4_vacuum, u4_minimal, qz_spin_projection

def test_majorana_phases_invisible():
    m2 = jnp.array([0.0, 7.42e-5, 2.51e-3])
    a = p3_vacuum(pmns(0.583, 0.1496, 0.855, 3.4), m2, 1300.0, 2.5, 1)
    b = p3_vacuum(pmns_majorana(0.583, 0.1496, 0.855, 3.4, 1.0, 2.5), m2, 1300.0, 2.5, 1)
    assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-14)

def test_four_flavors_unitary():
    U4 = u4_minimal(pmns(0.583, 0.1496, 0.855, 3.4), 0.3)
    assert abs(float(p4_vacuum(U4, jnp.array([0.0, 7.42e-5, 2.51e-3, 1.0]), 0.5, 0.003, 0).sum()) - 1) < 1e-12

def test_spin_projection_grid():
    assert float(qz_spin_projection(1.0, 2.0)) == 0.5
