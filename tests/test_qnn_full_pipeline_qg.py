"""Tests for examples/qnn_full_pipeline_qg.py (§119): calibration and verdict."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_verdict():
    import qnn_full_pipeline_qg as P

    p = {"noiseless correct": 0.97, "A raw": 0.93, "A qang": 0.95, "A qang + echo": 0.96, "C qang": 0.955,
         "C qang + echo": 0.965, "D raw": 0.94, "inputs": 189}
    assert P.verdict([{"pooled": p}] * 3) == {"K1": True, "K2": True, "K3": True, "K4": True}


def test_calibration_from_backend():
    pytest.importorskip("qiskit_ibm_runtime")
    pytest.importorskip("sklearn")
    import qnn_full_pipeline_qg as P
    import qnn_hardware_qg as H
    from qang.qml import WeightQNN

    backend = H.get_backend("fake", "fake_brisbane")
    Xtr, Xte, ytr, yte = P.data("iris")
    m = WeightQNN(5, 1).fit(Xtr, ytr, 1, seed=0)
    layout, cal = P.calibration(backend, H.build_circuit(m, Xte[0]))
    assert len(set(layout)) == 5 and cal["gamma"].shape == (5,)
    assert np.all((cal["gamma"] > 0) & (cal["gamma"] < 0.2)) and 0 <= cal["dephasing"] < 0.1
