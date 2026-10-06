"""Tests for examples/qnn_dephasing_zne_8q_qg.py (§122): F6 at 8 qubits and the verdict."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from qang.qml import WeightQNN  # noqa: E402
from qang.sectors import filter_distribution  # noqa: E402


def test_f6_at_8_qubits():
    rng = np.random.default_rng(1)
    for w, kw in ((1, {}), (2, {"encoding": "dual"})):
        m = WeightQNN(8, w, **kw)
        th = rng.uniform(-np.pi, np.pi, m.n_theta)
        psi = m.encode(rng.uniform(-1, 1, (2, 7)))
        a = [filter_distribution(p, 8, w)[0] for p in m.probs(th, psi, 0.08, 0.03)]
        b = [filter_distribution(p, 8, w)[0] for p in m.probs(th, psi, None, 0.03)]
        assert np.allclose(a, b, atol=1e-12)


def test_verdict():
    import qnn_dephasing_zne_8q_qg as Y

    row = {"weight": 1, "dataset": "x", "seed": 0, "noiseless": 0.95, "noise-aware filter": 0.945}
    vals = {"raw": (0.8, 0.3), "filter": (0.9, 0.2), "raw + ZNE": (0.85, 0.25), "filter + ZNE": (0.93, 0.1),
            "filter + ZNE (Richardson)": (0.93, 0.08)}
    for lab in ("exact", "shots"):
        for k, (a, e) in vals.items():
            row[f"{lab} {k} acc"], row[f"{lab} {k} err"] = a, e
    rows = [dict(row, weight=1), dict(row, weight=2, **{"shots filter + ZNE acc": 0.85})]
    assert Y.verdict(rows) == {"Y1": True, "Y2": True, "Y3": True, "Y4": True, "Y5": True}
