"""Tests for qang.qiskit_ml (§140): qang's QNN inside qiskit-machine-learning."""

import numpy as np
import pytest

pytest.importorskip("qiskit_machine_learning")
from qiskit.primitives import StatevectorSampler  # noqa: E402

from qang import qiskit_ml as QM  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402


def _model(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (12, 4))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    return WeightQNN(5, 1).fit(X, y, epochs=40), X, y


def test_sampler_qnn_matches_weightqnn_probs():
    m, X, _ = _model()
    qnn, _ = QM.qang_sampler_qnn(m, sampler=StatevectorSampler(default_shots=200000, seed=1))
    P = qnn.forward(QM.loader_inputs(X[:4]), m.params_[: m.n_theta])
    ref = m.probs(m.params_[: m.n_theta], m.encode(X[:4]))
    assert P.shape == (4, 32) and np.allclose(P, ref, atol=0.01)


def test_qg_features_and_classifier():
    m, X, y = _model(1)
    ref = m.probs(m.params_[: m.n_theta], m.encode(X))
    raw, filt, kept = QM.qg_features(ref, 5, 1)
    assert np.allclose(raw, m.qg_z(ref, qang=False)) and np.allclose(filt, m.qg_z(ref, qang=True))
    assert np.allclose(kept, 1.0)
    c = QM.QangQNNClassifier(m, sampler=StatevectorSampler(default_shots=100000, seed=2))
    assert np.mean(np.sign(c.decision(X)) == np.sign(m.decision(m.params_, X))) >= 0.9


def test_backward_shapes():
    m, X, _ = _model(2)
    qnn, _ = QM.qang_sampler_qnn(m, sampler=StatevectorSampler(default_shots=2000, seed=3))
    _, gw = qnn.backward(QM.loader_inputs(X[:1]), m.params_[: m.n_theta])
    assert gw.shape == (1, 32, m.n_theta)


def test_device_pass_manager_reads_logical_bits():
    """Regression for the qiskit-machine-learning 0.9.1 layout read-out: a device-transpiled QNN must keep
    most of its shots in weight 1, not read every shot as 0."""
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime.executor_sampler import Sampler
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

    m, X, _ = _model(3)
    b = FakeSherbrooke()
    pm = generate_preset_pass_manager(backend=b, optimization_level=1, seed_transpiler=1)
    qnn, _ = QM.qang_sampler_qnn(m, sampler=Sampler(mode=b, options={"default_shots": 1000}), pass_manager=pm)
    P = qnn.forward(QM.loader_inputs(X[:3]), m.params_[: m.n_theta])
    assert P.shape == (3, 32) and np.all(QM.qg_features(P, 5, 1)[2] > 0.5)
