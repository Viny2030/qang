"""Tests for examples/qnn_torino_residual_qg.py (§132): the single-qubit channel added to qang's QNN simulator."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def _model(p2=0.0, p1=0.0):
    import qnn_torino_residual_qg as T

    m = T.GateNoise1QNN(T.N, 1)
    m.gate_error, m.one_qubit_error = p2, p1
    return m


def _inputs(m, k=4, seed=3):
    rng = np.random.default_rng(seed)
    th = rng.uniform(-1, 1, m.n_theta)
    psi = np.zeros((k, 2 ** m.n))
    for s in range(k):
        v = rng.normal(size=m.n)
        v /= np.linalg.norm(v)
        for q in range(m.n):
            psi[s, 1 << (m.n - 1 - q)] = v[q]
    return th, psi


def test_zero_one_qubit_error_equals_gate_model():
    import qnn_gate_noise_qg as G

    m1 = _model(0.02, 0.0)
    m0 = G.gate_model(0.02)
    th, psi = _inputs(m1)
    m0.params_ = m1.params_ = None
    assert np.allclose(m1.probs(th, psi, gamma=0.01, dephasing=0.003), m0.probs(th, psi, gamma=0.01, dephasing=0.003))


def test_depol_one_is_a_channel():
    m = _model(0.0, 0.0)
    m._tables()
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2, 32, 32)) + 1j * rng.normal(size=(2, 32, 32))
    rho = a @ a.conj().transpose(0, 2, 1)
    rho /= np.trace(rho, axis1=1, axis2=2)[:, None, None]
    out = m._depol_one(rho, 2)
    assert np.allclose(np.trace(out, axis1=1, axis2=2), 1)
    assert np.allclose(out, out.conj().transpose(0, 2, 1))
    assert np.all(np.linalg.eigvalsh(out) > -1e-12)
    # qubit 2 is maximally mixed and uncorrelated after full depolarizing
    b = m._b[2]
    assert np.allclose(np.einsum("sii->si", out)[:, b].sum(1), 0.5)


def test_one_qubit_error_lowers_kept_fraction():
    import qnn_torino_residual_qg as T

    th, psi = _inputs(_model())
    k0 = _model().probs(th, psi)[:, T.ONE].sum(1)
    k1 = _model(0.0, 0.01).probs(th, psi)[:, T.ONE].sum(1)
    assert np.allclose(k0, 1.0) and np.all(k1 < 1.0) and np.all(k1 > 0.8)


def test_aer_models_split_the_gate_errors():
    pytest.importorskip("qiskit_aer")
    import qnn_hardware_qg as H
    import qnn_torino_residual_qg as T

    ms = T.aer_models(H.get_backend("fake", "fake_torino"))
    assert set(ms) == set(T.CHANNELS)
    assert set(ms["2q gates"]._local_quantum_errors) == {"cz"}
    assert set(ms["1q gates"]._local_quantum_errors) >= {"sx", "x"}
    assert not ms["2q gates"]._local_readout_errors and not ms["1q gates"]._local_readout_errors
