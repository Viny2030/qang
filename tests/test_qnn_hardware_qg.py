"""Tests for examples/qnn_hardware_qg.py (§85) and
examples/hardware_characterization_ibm.py (§86): circuits, bit order and
readout, without any account or hardware."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("sklearn")

import hardware_characterization_ibm as HI  # noqa: E402
import qnn_hardware_qg as QH  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402


def test_compiled_qnn_circuits_match_the_model():
    m = WeightQNN(5, 1)
    m.params_ = np.random.default_rng(0).uniform(-3, 3, m.n_theta + m.n_head + 1)
    X = np.random.default_rng(1).uniform(-1, 1, (5, 4))
    assert QH.check_circuits(m, X) < 1e-12


def test_counts_bit_order_and_evaluation():
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    m = WeightQNN(5, 1)
    m.params_ = np.random.default_rng(2).uniform(-3, 3, m.n_theta + m.n_head + 1)
    X = np.random.default_rng(3).uniform(-1, 1, (3, 4))
    sim = AerSimulator()
    res = sim.run(transpile([QH.build_circuit(m, x) for x in X], sim), shots=20000, seed_simulator=1).result()
    probs = [QH.counts_to_probs(res.get_counts(i)) for i in range(3)]
    ref = m.probs(m.params_[: m.n_theta], m.encode(X))
    assert np.abs(np.array(probs) - ref).max() < 0.02
    r = QH.evaluate(m, X, (m.decision(m.params_, X) > 0).astype(int), probs)
    assert r["kept fraction"] == pytest.approx(1.0)
    assert set(QH.verdict(r)) == {"K1", "K2", "K3"}


def test_herald_counts_order():
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    spec, circs = HI.circuits("qg")
    sim = AerSimulator()
    res = sim.run(transpile(circs, sim), shots=100, seed_simulator=1).result()
    d = {s: HI.to_array(res.get_counts(i), True) for i, s in enumerate(spec)}
    assert list(d[("I", 0.0)]) == [100, 0, 0, 0]
    assert list(d[("X", 0.0)]) == [0, 100, 0, 0]
    spec_s, circs_s = HI.circuits("standard")
    res = sim.run(transpile(circs_s, sim), shots=100, seed_simulator=1).result()
    assert list(HI.to_array(res.get_counts(1), False)) == [0, 100]
