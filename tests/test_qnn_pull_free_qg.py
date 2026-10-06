"""Tests for examples/qnn_pull_free_qg.py (§123): exact invariance of the pull-free readout under equal T1."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_pull_free_raw_logits_are_scaled_noiseless_logits():
    import qnn_pull_free_qg as PF
    from qang.qml import kept_fraction

    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, (7, 4))
    for ro in ("qubit", "head"):
        m = PF.PullFreeQNN(5, 4, readout=ro)
        m.params_ = np.concatenate([rng.uniform(-np.pi, np.pi, m.n_theta), rng.normal(0, 1, m.n_head)])
        K = kept_fraction(0.08, 1, m.depth)
        raw = m.logits(m.params_, X, gamma=0.08, qang=False)
        assert np.allclose(raw, K * m.logits(m.params_, X), atol=1e-12)
        assert np.array_equal(m.predict(X, gamma=0.08, qang=False), m.predict(X))


def test_pull_free_fit_runs_and_verdict():
    import qnn_pull_free_qg as PF

    rng = np.random.default_rng(0)
    X, y = rng.uniform(-1, 1, (20, 4)), np.arange(20) % 3
    m = PF.PullFreeQNN(5, 3, readout="head").fit(X, y, epochs=2, seed=0)
    assert m.params_.shape == (m.n_theta + m.n_head,)
    rows = []
    for ro in ("qubit", "head"):
        for k, raw in (("standard", 0.75), ("pull-free", 0.93)):
            rows.append({"seed": 0, "readout": ro, "model": k, "noiseless": 0.94, "T1 qang": 0.94, "T1 raw": raw,
                         "unequal qang": 0.935, "unequal raw": 0.93 if k == "pull-free" else 0.7,
                         "shots qang": 0.93, "shots raw": 0.925 if k == "pull-free" else 0.7})
    assert PF.verdict(rows) == {"P1": True, "P2": True, "P3": True, "P4": True, "P5": True}
