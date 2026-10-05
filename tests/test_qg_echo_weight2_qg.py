"""Tests for examples/qg_echo_weight2_qg.py (§107)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_echo_returns_state_and_inputs_stay_in_sector():
    pytest.importorskip("qiskit")
    pytest.importorskip("sklearn")
    from qiskit.quantum_info import Statevector

    import qg_echo_weight2_qg as W

    gates = W.block_gates()
    qc = W.echo_circuit((1, 3), gates)
    qc.remove_final_measurements()
    assert Statevector(qc).probabilities()[W.state_index((1, 3))] == pytest.approx(1.0)
    p = Statevector(W.input_circuit([0.3, -1.0, 2.0, 0.5], gates, measure=False)).probabilities()
    weights = np.array([bin(i).count("1") for i in range(32)])
    assert p[weights == 2].sum() == pytest.approx(1.0)


def test_verdict():
    pytest.importorskip("sklearn")
    import qg_echo_weight2_qg as W

    r = {"qg_Z error, without qang": 0.2, "qg_Z error, qang": 0.08, "qg_Z error, qang + echo": 0.04,
         "qg_ZZ error, without qang": 0.2, "qg_ZZ error, qang": 0.09, "qg_ZZ error, qang + echo": 0.05}
    assert W.verdict([r]) == {"W1": True, "W2": True, "W3": True, "W4": True}
