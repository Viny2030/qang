"""Tests for examples/qnn_shot_training_qg.py (§109)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_shot_training_runs_and_verdict():
    pytest.importorskip("sklearn")
    import qnn_shot_training_qg as T
    from qang.qml import WeightQNN

    m = WeightQNN(5, 1)
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (12, 4))
    y = (X[:, 0] > 0).astype(int)
    init = np.concatenate([rng.uniform(-3, 3, m.n_theta), rng.normal(0, 0.5, m.n_head), [0.0]])
    p = T.train_shots(m, X, y, 50, True, init, np.random.default_rng(1), epochs=3)
    assert p.shape == init.shape and not np.allclose(p, init)
    assert 0 <= T.test_accuracy(m, p, X, y, True, np.random.default_rng(2), shots=50) <= 1
    s = {"S=1000 qang-raw": (0.002,), "S=100 qang-raw": (-0.02,), "exact-best1000": (0.0,), "exact-qang1000": (0.01,)}
    assert T.verdict(s) == {"S1": True, "S2": True, "S3": True, "S4": True}
