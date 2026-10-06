"""Tests for examples/qnn_gate_noise_qg.py (§124): the gate-noise simulator and the calibration."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_simulator_equals_weightqnn_without_gate_error():
    import qnn_gate_noise_qg as G
    from qang.qml import WeightQNN

    rng = np.random.default_rng(0)
    m, g = WeightQNN(5, 1), G.GateNoiseQNN(5, 1)
    th = rng.uniform(-3, 3, m.n_theta)
    psi = m.encode(rng.uniform(-1, 1, (4, 4)))
    gam = np.array([0.01, 0.05, 0.02, 0.08, 0.03])
    assert np.allclose(m.probs(th, psi, gam, 0.02), g.probs(th, psi, gam, 0.02), atol=1e-12)


def test_pair_depolarizing_is_identity_over_four_tensor_partial_trace():
    import qnn_gate_noise_qg as G

    rng = np.random.default_rng(1)
    g = G.GateNoiseQNN(5, 1)
    g._tables()
    r = rng.normal(size=(2, 32, 32))
    r = r @ r.transpose(0, 2, 1)
    out = g._depol_pair(r, 1, 3)
    t = r.reshape((2,) + (2,) * 10)
    red = np.einsum("zabcdeAbCdE->zaceACE", t)  # trace over qubits 1 and 3
    ref = np.einsum("zaceACE,bB,dD->zabcdeABCDE", red, np.eye(2) / 2, np.eye(2) / 2).reshape(2, 32, 32)
    assert np.allclose(out, ref, atol=1e-12)
    g.gate_error = 0.1
    p = g.probs(rng.uniform(-3, 3, g.n_theta), g.encode(rng.uniform(-1, 1, (3, 4))), 0.02, 0.01)
    assert np.allclose(p.sum(axis=1), 1) and np.all(p > -1e-12)


def test_gate_calibration_and_verdict():
    pytest.importorskip("qiskit_ibm_runtime")
    pytest.importorskip("sklearn")
    import qnn_full_pipeline_qg as P
    import qnn_gate_noise_qg as G
    import qnn_hardware_qg as H
    from qang.qml import WeightQNN

    backend = H.get_backend("fake", "fake_brisbane")
    Xtr, Xte, ytr, yte = P.data("iris")
    m = WeightQNN(5, 1).fit(Xtr, ytr, 1, seed=0)
    layout, _ = P.calibration(backend, H.build_circuit(m, Xte[0]))
    c = G.gate_calibration(backend, H.build_circuit(m, Xte[0]), layout)
    assert c["gate"] == "ecr" and 0 < c["mean error"] < 0.1 and c["per RBS"] > 0 and 0 < c["depolarizing"] < 1
    p = {"noiseless": 0.98, "A raw": 0.95, "A qang": 0.97, "C1 qang": 0.968, "C2 qang": 0.976, "D2 raw": 0.965,
         "A predicted qang": 0.97, "inputs": 189}
    res = [{"pooled": p, "kept": {"kept device": 0.7, "kept predicted": 0.72}}] * 3
    assert G.verdict(res) == {"R1": True, "R2": True, "R3": True, "R4": True, "R5": True}
