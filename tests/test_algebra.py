"""Tests for qang.algebra (§166)."""

import math

import numpy as np
import pytest

from qang import algebra as AL


def test_u3_qang_equals_qiskit_u_gate():
    pytest.importorskip("qiskit")
    from qiskit.circuit.library import UGate
    from qiskit.quantum_info import Operator

    rng = np.random.default_rng(0)
    for _ in range(200):
        th, ph, la = rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
        assert np.allclose(AL.u3_qang(math.cos(th), ph, la), Operator(UGate(th, ph, la)).data, atol=1e-12)


def test_standard_gates_from_qg():
    X = np.array([[0, 1], [1, 0]])
    Z = np.diag([1, -1])
    assert np.allclose(AL.u3_qang(1, 0, 0), np.eye(2))
    assert np.allclose(AL.u3_qang(-1, 0, math.pi), X)
    assert np.allclose(AL.u3_qang(1, 0, math.pi), Z)
    assert np.allclose(AL.u3_qang(0, 0, math.pi), (X + Z) / math.sqrt(2))


def test_transition_matrix_is_phase_free_and_doubly_stochastic():
    for q in (-0.7, 0.0, 0.3, 0.95):
        T = AL.transition_matrix(q)
        assert np.allclose(np.abs(AL.u3_qang(q, 0.7, 1.9)) ** 2, T)
        assert np.allclose(T.sum(0), 1) and np.allclose(T.sum(1), 1)


@pytest.mark.parametrize("q", [-0.8, 0.0, 0.45, 0.99])
def test_householder_and_grover(q):
    psi = np.array([math.sqrt((1 + q) / 2), math.sqrt((1 - q) / 2)])
    H = AL.householder_qg(q)
    assert np.allclose(H, np.eye(2) - 2 * np.outer(psi, psi))
    assert np.allclose(H @ H, np.eye(2)) and np.allclose(H, H.conj().T)
    assert np.allclose(AL.grover_diffuser_qg(q), 2 * np.outer(psi, psi) - np.eye(2))


def test_transition_probability():
    rng = np.random.default_rng(1)
    for _ in range(200):
        t1, t2, p1, p2 = rng.uniform(0, math.pi), rng.uniform(0, math.pi), rng.uniform(0, 6.3), rng.uniform(0, 6.3)
        a = np.array([math.cos(t1 / 2), np.exp(1j * p1) * math.sin(t1 / 2)])
        b = np.array([math.cos(t2 / 2), np.exp(1j * p2) * math.sin(t2 / 2)])
        exact = abs(np.vdot(a, b)) ** 2
        assert AL.transition_probability(math.cos(t1), math.cos(t2), p1, p2) == pytest.approx(exact, abs=1e-12)
    q1, q2 = 0.3, -0.5
    expected = (1 + q1 * q2 + math.sqrt((1 - q1 ** 2) * (1 - q2 ** 2))) / 2
    assert AL.transition_probability(q1, q2) == pytest.approx(expected)


@pytest.mark.parametrize("d", [2, 3, 4, 6])
def test_cartan_generalization(d):
    G = AL.cartan_generators(d)
    assert np.allclose(G @ G.T, 2 * np.eye(d - 1))
    assert np.allclose(G.sum(axis=1), 0)
    p = np.random.default_rng(d).dirichlet(np.ones(d))
    assert np.allclose(AL.populations_from_cartan(AL.cartan_values(p)), p)
    if d == 2:
        assert AL.cartan_values(p)[0] == pytest.approx(p[0] - p[1])  # qg_Z
