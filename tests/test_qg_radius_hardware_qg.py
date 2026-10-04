"""Tests for examples/qg_radius_hardware_qg.py (§100)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_deficits_and_verdict():
    pytest.importorskip("sklearn")
    import qg_radius_hardware_qg as R
    from qang.qml import WeightQNN

    m = WeightQNN(5, 1)
    p = np.zeros(32)
    p[16] = 0.9   # weight 1, qubit 0 excited
    p[0] = 0.1    # decayed to |00000>
    d = R.deficits(p, m.zsign, 1000)
    assert d["kept"] == pytest.approx(0.9)
    assert np.allclose(d["with qang"], 0.0)
    assert d["without qang"][0] > 0.3
    r = {"trained: deficit error with qang": 0.05, "trained: deficit error without qang": 0.2,
         "echo: false deficit of the excited qubit, with qang": [0.1] * 5,
         "echo: false deficit of the excited qubit, without qang": [0.6] * 5}
    assert R.verdict(r) == {"R1": True, "R2": True, "R3": True, "R4": True}


def test_echo_circuit_returns_the_basis_state():
    pytest.importorskip("qiskit")
    pytest.importorskip("sklearn")
    from qiskit.quantum_info import Statevector

    import qg_radius_hardware_qg as R
    import qnn_hardware_qg as H

    model, _, _ = H.trained_model(epochs=2)
    qc = R.echo_circuit(model, 2)
    qc.remove_final_measurements()
    p = Statevector(qc).probabilities()
    assert p[int("00100", 2)] == pytest.approx(1.0)
