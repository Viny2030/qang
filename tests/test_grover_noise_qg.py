"""
Tests for examples/grover_noise_qg.py (RESEARCH_NOTES §63): Grover with
noise; the per-qubit qg reading loses to the outcome histogram.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit_aer")

import grover_noise_qg as G  # noqa: E402


@pytest.mark.parametrize("n,k", [(4, 1), (4, 3), (5, 4)])
def test_noiseless_matches_formula(n, k):
    p = G.probabilities(n, k)
    assert p[G.MARKED[n]] == pytest.approx(G.ideal_p(n, k), abs=1e-9)
    # uniform unmarked outcomes: the qg formula is exact without noise
    s = np.array([1 - 2 * ((G.MARKED[n] >> i) & 1) for i in range(n)])
    P = p[G.MARKED[n]]
    assert np.allclose(G.qg_from_p(p, n), s * (2**n * P - 1) / (2**n - 1), atol=1e-9)


def test_noise_moves_best_k_and_breaks_uniformity():
    rows0 = G.study_a(5, 0.0)
    rows = G.study_a(5, 0.005)
    assert min(rows0, key=lambda r: r[3])[0] == 3
    assert min(rows, key=lambda r: r[3])[0] == 2
    p = G.probabilities(5, 3, 0.01)
    s = np.array([1 - 2 * ((G.MARKED[5] >> i) & 1) for i in range(5)])
    pred = s * (32 * p[G.MARKED[5]] - 1) / 31
    assert np.abs(G.qg_from_p(p, 5) - pred).max() > 0.03


def test_mode_beats_qg_signs():
    rng = np.random.default_rng(0)
    P, rows = G.study_c(5, 4, 0.01, (100,), 800, rng)
    assert rows[0][2] > rows[0][1] + 0.15
