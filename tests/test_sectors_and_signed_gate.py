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
