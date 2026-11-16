import pytest; jax = pytest.importorskip("jax")
import numpy as np, jax.numpy as jnp
from qang.qtele import werner, teleport, bloch, state_from_bloch, teleport_fidelity_werner_qzz, resource_correlators

def test_channel_contracts_bloch():
    r = jnp.array([0.3, -0.4, 0.5]); out = teleport(state_from_bloch(r), werner(0.7))
    assert np.allclose(np.asarray(bloch(out)), 0.7 * np.asarray(r), atol=1e-13)

def test_classical_threshold_333_mqg():
    assert abs(float(teleport_fidelity_werner_qzz(1 / 3)) - 2 / 3) < 1e-15

def test_werner_correlators():
    qxx, qyy, qzz = resource_correlators(werner(0.5))
    assert abs(float(qxx) - 0.5) < 1e-14 and abs(float(qyy) + 0.5) < 1e-14 and abs(float(qzz) - 0.5) < 1e-14
