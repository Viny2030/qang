import pytest; jax = pytest.importorskip("jax")
import jax.numpy as jnp, numpy as np
from qang.qan import mutual_information_qg, joint_probs_qg, mi_shot_variance_qg, qan_state, qan_readout

def test_product_zero_mi():
    assert abs(float(mutual_information_qg(0.3, -0.6, -0.18))) < 1e-12

def test_joint_probs_sum_and_marginals():
    p = np.asarray(joint_probs_qg(0.2, -0.3, 0.5))
    assert abs(p.sum() - 1) < 1e-15 and abs((p[0]+p[1]) - 0.6) < 1e-15

def test_bell_one_bit():
    psi = qan_state(jnp.array([250., 250.]), jnp.array([250.]), jnp.array([250., -250.]), 2, ((0, 1),))
    qz, qzz = qan_readout(psi, 2)
    assert abs(float(mutual_information_qg(qz[0], qz[1], qzz[0, 1])) - 1) < 1e-10

def test_delta_variance_positive():
    assert float(mi_shot_variance_qg(0.2, -0.3, 0.5)) > 0
