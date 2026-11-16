import pytest; jax = pytest.importorskip("jax")
import jax.numpy as jnp
from qang.qan import qan_state
from qang.qfisher import fubini_study_metric, classical_fisher_z, psd_gap, circuits_for_metric, QFI_1Q_QGA

def test_single_qubit_fisher_constant():
    s = lambda t: qan_state(t, jnp.zeros(0), jnp.zeros(1), 1, ())
    for a in (10.0, 250.0, 480.0):
        assert abs(4 * float(fubini_study_metric(s, jnp.array([a]))[0, 0]) - QFI_1Q_QGA) < 1e-15
        assert abs(float(classical_fisher_z(s, jnp.array([a]))[0, 0]) - QFI_1Q_QGA) < 1e-12

def test_cfi_below_qfi():
    s = lambda t: qan_state(t[:2], t[2:3], t[3:], 2, ((0, 1),))
    t = jnp.array([123.0, 77.0, 210.0, 15.0, 399.0])
    assert psd_gap(4 * fubini_study_metric(s, t), classical_fisher_z(s, t)) > -1e-12

def test_circuit_counts():
    assert circuits_for_metric(8) == (44, 16)
