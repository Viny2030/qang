"""Tests for examples/qnn_unequal_t1_qg.py (§117): the ratio invariance of the
filtered readout under T1, the rates and the verdict."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from qang.qml import WeightQNN  # noqa: E402
from qang.sectors import filter_distribution  # noqa: E402


def test_filtered_readout_depends_only_on_ratios():
    # (1 - gamma'_q) = c (1 - gamma_q) for every q: same filtered distribution, any weight
    rng = np.random.default_rng(3)
    g = np.array([0.01, 0.05, 0.09, 0.12, 0.16])
    g2 = 1 - 0.8 * (1 - g)
    for w, kw in ((1, {}), (2, {"encoding": "dual"})):
        m = WeightQNN(5, w, **kw)
        th = rng.uniform(-np.pi, np.pi, m.n_theta)
        psi = m.encode(rng.uniform(-1, 1, (4, 4)))
        f1 = [filter_distribution(p, 5, w)[0] for p in m.probs(th, psi, g)]
        f2 = [filter_distribution(p, 5, w)[0] for p in m.probs(th, psi, g2)]
        assert np.allclose(f1, f2, atol=1e-12)
        r1, r2 = m.probs(th, psi, g), m.probs(th, psi, g2)
        assert not np.allclose(r1, r2, atol=1e-3)  # the raw distribution changes


def test_rates_and_verdict():
    import qnn_unequal_t1_qg as U

    true, cal, drift = U.rates(1170)
    assert np.allclose(true.mean(), 0.08) and true.min() == 0.0 and np.all(drift >= true)
    assert np.all(np.abs(cal - true) <= 0.5 * true + 1e-12)
    row = {"noiseless": 0.95, "A": 0.93, "B": 0.90, "C": 0.948, "C*": 0.95, "D": 0.945,
           "A drift": 0.92, "B drift": 0.85, "C drift": 0.94, "D drift": 0.90}
    rows = [dict(row, weight=w, dataset="x", seed=0) for w in (1, 2)]
    assert U.verdict(rows) == {"U1": True, "U2": True, "U3": True, "U4": True, "U5": True}
