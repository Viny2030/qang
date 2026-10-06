"""Tests for examples/qnn_device_calibration_qg.py (§120): the mixed readout and the verdict."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_mixed_readout_identity_and_mixing():
    import qnn_device_calibration_qg as C
    from qang.qml import WeightQNN

    rng = np.random.default_rng(0)
    m = WeightQNN(5, 1)
    th = rng.uniform(-np.pi, np.pi, m.n_theta)
    pr = m.probs(th, m.encode(rng.uniform(-1, 1, (3, 4))), 0.05)
    assert np.allclose(C.MixedQNN(np.eye(5), 5, 1).qg_z(pr), m.qg_z(pr))
    mix = np.full((5, 5), 0.2)
    z = C.MixedQNN(mix, 5, 1).qg_z(pr)
    assert np.allclose(z, z[:, :1])  # full mixing: every qubit equally excited


def test_verdict():
    import qnn_device_calibration_qg as C

    p = {"noiseless": 0.98, "A raw": 0.95, "A qang": 0.97, "A qang + echo": 0.97, "E": 0.975, "H": 0.976, "H echo": 0.977,
         "inputs": 189}
    assert C.verdict([{"pooled": p}] * 3) == {"G1": True, "G2": True, "G3": True, "G4": True, "G5": True}
