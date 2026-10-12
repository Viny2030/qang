"""Tests for examples/qnn_ingate_dephasing_qg.py (§133): in-gate dephasing in qang's QNN simulator."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def _inputs(m, k=3, seed=5):
    rng = np.random.default_rng(seed)
    th = rng.uniform(-1, 1, m.n_theta)
    psi = np.zeros((k, 2 ** m.n))
    for s in range(k):
        v = rng.normal(size=m.n)
        v /= np.linalg.norm(v)
        for q in range(m.n):
            psi[s, 1 << (m.n - 1 - q)] = v[q]
    return th, psi


def test_zero_ingate_equals_section_132_model():
    import qnn_ingate_dephasing_qg as D
    import qnn_torino_residual_qg as T

    a, b = D.InGateQNN(D.N, 1), T.GateNoise1QNN(D.N, 1)
    for m in (a, b):
        m.gate_error, m.one_qubit_error = 0.02, 0.003
    a.ingate = np.zeros(D.N)
    th, psi = _inputs(a)
    assert np.allclose(a.probs(th, psi, gamma=0.01, dephasing=0.002), b.probs(th, psi, gamma=0.01, dephasing=0.002))


def test_ingate_on_one_qubit_only_lowers_kept_and_matches_uniform_case():
    import qnn_ingate_dephasing_qg as D
    import qnn_torino_residual_qg as T

    m = D.InGateQNN(D.N, 1)
    th, psi = _inputs(m)
    m.ingate = np.array([0.0, 0.0, 0.01, 0.0, 0.0])
    kept = m.probs(th, psi)[:, D.ONE].sum(1)
    assert np.all(kept < 1.0) and np.all(kept > 0.9)
    u, v = D.InGateQNN(D.N, 1), T.GateNoise1QNN(D.N, 1)
    u.ingate = np.full(D.N, 0.004)
    v.one_qubit_error = 0.004
    assert np.allclose(u.probs(th, psi), v.probs(th, psi))


def test_rbs_count():
    import qnn_ingate_dephasing_qg as D
    from qang.qml import WeightQNN

    m = WeightQNN(D.N, 1)
    c = D.rbs_per_qubit(m)
    assert c.sum() == 2 * (D.N - 1 + m.n_theta) and np.all(c >= 1)
