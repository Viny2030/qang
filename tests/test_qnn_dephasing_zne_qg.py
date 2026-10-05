"""Tests for examples/qnn_dephasing_zne_qg.py (§118): F6, the noise scaling and the extrapolation."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from qang.qml import WeightQNN  # noqa: E402
from qang.sectors import filter_distribution  # noqa: E402


def test_filter_removes_equal_t1_under_dephasing():
    # F6: equal T1 + dephasing, filtered = dephasing alone, filtered; any weight
    rng = np.random.default_rng(0)
    for w, kw in ((1, {}), (2, {"encoding": "dual"})):
        m = WeightQNN(5, w, **kw)
        th = rng.uniform(-np.pi, np.pi, m.n_theta)
        psi = m.encode(rng.uniform(-1, 1, (3, 4)))
        a = [filter_distribution(p, 5, w)[0] for p in m.probs(th, psi, 0.08, 0.03)]
        b = [filter_distribution(p, 5, w)[0] for p in m.probs(th, psi, None, 0.03)]
        assert np.allclose(a, b, atol=1e-12)


def test_scaling_and_extrapolation():
    import qnn_dephasing_zne_qg as Z

    g, p = Z.scaled(0.08, 0.03, 1)
    assert np.isclose(g, 0.08) and np.isclose(p, 0.03)
    g2, p2 = Z.scaled(0.08, 0.03, 2)
    assert np.isclose(1 - g2, 0.92**2) and np.isclose(1 - 2 * p2, 0.94**2)
    f = [np.array([0.5]), np.array([0.4]), np.array([0.3])]
    assert np.isclose(Z.extrapolate(f, 1)[0], 0.6) and np.isclose(Z.extrapolate(f, 2)[0], 0.6)
    assert Z.extrapolate([np.array([0.9]), np.array([0.5]), np.array([0.2])], 1)[0] == 1.0  # clipped


def test_verdict_shape():
    import qnn_dephasing_zne_qg as Z

    row = {"weight": 1, "dataset": "x", "seed": 0, "noiseless": 0.95}
    for p in Z.DEPHASING:
        row[f"p={p} noise-aware filter"] = 0.94
        for lab in ("exact", "shots"):
            for k, (a, e) in zip(Z.READOUTS, ((0.8, 0.3), (0.90, 0.10), (0.9, 0.2), (0.945, 0.04), (0.94, 0.05))):
                row[f"p={p} {lab} {k} acc"], row[f"p={p} {lab} {k} err"] = a, e
    rows = [dict(row, weight=w) for w in (1, 2)]
    assert Z.verdict(rows) == {"Z1": True, "Z2": True, "Z3": True, "Z4": True, "Z5": True}
