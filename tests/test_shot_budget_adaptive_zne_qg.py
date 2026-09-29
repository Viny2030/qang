"""
Tests for examples/shot_budget_adaptive_zne_qg.py (RESEARCH_NOTES §67):
filter vs filter + ZNE across shot budgets, and the pilot-based switch.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit_aer")

import shot_budget_adaptive_zne_qg as A  # noqa: E402


def _rmse(d, S, reps=250, seed=5):
    rng = np.random.default_rng(seed)
    p1, p3, ideal = A.distributions(d)
    e = {"filter": [], "filter+ZNE": [], "ZNE": []}
    for _ in range(reps):
        for k, v in A.fixed(p1, p3, S, rng).items():
            e[k].append(v - ideal)
    return {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in e.items()}


def test_crossover_shallow():
    lo = _rmse(2, 500)
    hi = _rmse(2, 50000)
    assert lo["filter"] < lo["filter+ZNE"]
    assert hi["filter+ZNE"] < hi["filter"]


def test_deep_circuit_needs_zne():
    m = _rmse(6, 2000)
    assert m["filter+ZNE"] < 0.7 * m["filter"]
    assert m["ZNE"] > m["filter+ZNE"]
