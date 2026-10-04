"""
Tests for qang.sectors, qang.qiskit_gate.SignedRQangGate and
examples/sector_exposure_qg.py (RESEARCH_NOTES §68).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

from qang.sectors import filter_distribution, hamming_weights


def test_filter_distribution():
    p = np.zeros(8)
    p[[0b011, 0b101, 0b111]] = [0.3, 0.5, 0.2]
    out, kept = filter_distribution(p, 3, 2)
    assert kept == pytest.approx(0.8)
    assert out[0b011] == pytest.approx(0.375) and out[0b111] == 0
    assert list(hamming_weights(2)) == [0, 1, 1, 2]


pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Pauli, Statevector  # noqa: E402

from qang.qiskit_gate import FullRQangGate, SignedRQangGate  # noqa: E402
from qang.core import Qang  # noqa: E402
from qang.sectors import sector_exposure  # noqa: E402


@pytest.mark.parametrize("q", [0.9, 0.2, -0.6])
@pytest.mark.parametrize("s", [1, -1])
def test_signed_gate(q, s):
    qc = QuantumCircuit(1)
    qc.append(SignedRQangGate(q, s), [0])
    sv = Statevector(qc)
    assert sv.expectation_value(Pauli("Z")).real == pytest.approx(q, abs=1e-12)
    assert sv.expectation_value(Pauli("X")).real == pytest.approx(s * math.sqrt(1 - q * q), abs=1e-12)
    qf = QuantumCircuit(1)
    qf.append(FullRQangGate(Qang(q, phi=0.0 if s == 1 else math.pi)), [0])
    assert abs(np.vdot(Statevector(qf).data, sv.data)) == pytest.approx(1.0, abs=1e-12)


def test_exposure_zero_for_number_conserving_and_positive_for_cnot():
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.iswap(0, 1)
    assert sector_exposure(qc, 1)["mean"] == pytest.approx(0.0, abs=1e-12)
    qc2 = QuantumCircuit(2)
    qc2.x(0)
    qc2.h(1)
    qc2.cx(0, 1)
    assert sector_exposure(qc2, 1)["mean"] > 0.4


pytest.importorskip("qiskit_aer")
import sector_exposure_qg as SE  # noqa: E402


def test_exposure_ranks_compilations():
    rows = SE.xxz_points(dts=(0.25, 0.5))
    by = {(r["comp"], r["dt"]): r for r in rows}
    for dt in (0.25, 0.5):
        assert by[("number-conserving", dt)]["exposure"] < 1e-9
        assert by[("number-conserving", dt)]["leak"] < 0.02
        assert by[("3 CNOT", dt)]["exposure"] > by[("MS rotations", dt)]["exposure"]
        assert by[("3 CNOT", dt)]["leak"] > by[("MS rotations", dt)]["leak"]


def test_echo_transfer_matrix_and_unmix_sector():
    pytest.importorskip("scipy")
    from qang.sectors import echo_transfer_matrix, sector_states, unmix_sector

    n, k = 4, 2
    states = sector_states(n, k)
    rng = np.random.default_rng(3)
    m = len(states)
    A = 0.9 * np.eye(m) + rng.uniform(0, 0.1 / (m - 1), (m, m)) * (1 - np.eye(m))
    A /= A.sum(axis=0, keepdims=True)
    M = A @ A  # an echo: twice the forward error
    echo = []
    for j in range(m):
        p = np.zeros(2**n)
        p[states] = 0.8 * M[:, j]
        p[0] = 0.2  # decayed out of the sector
        echo.append(p)
    M_est = echo_transfer_matrix(echo, n, k)
    assert np.allclose(M_est, M, atol=1e-12)
    x = rng.dirichlet(np.ones(m))
    p = np.zeros(2**n)
    p[states] = 0.7 * (A @ x)
    p[1] = 0.3  # decayed to weight 1, outside the sector
    out = unmix_sector(p, M_est, n, k, power=0.5)
    assert np.allclose(out[states], x, atol=1e-6)
    assert np.isclose(out.sum(), 1.0) and np.allclose(np.delete(out, states), 0)
