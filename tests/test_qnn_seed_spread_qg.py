"""Tests for examples/qnn_seed_spread_qg.py (§121): F7, the raw logits are K L + (1 - K) v."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_f7_raw_logits_are_a_blend():
    import qnn_seed_spread_qg as S
    from qang.qml import MultiClassQNN, kept_fraction

    rng = np.random.default_rng(4)
    X = rng.uniform(-1, 1, (6, 4))
    for ro in ("qubit", "head"):
        m = MultiClassQNN(5, 4, readout=ro)
        m.params_ = np.concatenate([rng.uniform(-np.pi, np.pi, m.n_theta), rng.normal(0, 1, m.n_head)])
        K = kept_fraction(0.08, 1, m.depth)
        raw = m.logits(m.params_, X, gamma=0.08, qang=False)
        blend = K * m.logits(m.params_, X) + (1 - K) * S.pull_vector(m)[None, :]
        assert np.allclose(raw, blend, atol=1e-12)


def test_verdict_and_spearman():
    import qnn_seed_spread_qg as S

    assert np.isclose(S.spearman([1, 2, 3], [10, 20, 30]), 1.0)
    rows = []
    for i in range(9):
        for ro in ("qubit", "head"):
            rows.append({"seed": i, "readout": ro, "noiseless": 0.95, "with qang": 0.95,
                         "without qang": 0.95 - 0.03 * i, "pull": float(i), "F7 agreement": 1.0})
    v, _ = S.verdict(rows)
    assert v == {"V1": True, "V2": True, "V3": True, "V4": True, "V5": True}
