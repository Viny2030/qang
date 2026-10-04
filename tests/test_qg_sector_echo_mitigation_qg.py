"""Tests for examples/qg_sector_echo_mitigation_qg.py (§105)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_unmix_recovers_a_known_mixing_and_sqrt_is_stochastic():
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    import qg_sector_echo_mitigation_qg as E

    rng = np.random.default_rng(0)
    M = np.eye(5) * 0.85 + rng.uniform(0, 0.15 / 4, (5, 5)) * (1 - np.eye(5))
    M /= M.sum(axis=0, keepdims=True)
    x = np.array([0.5, 0.2, 0.1, 0.15, 0.05])
    assert np.allclose(E.unmix(M, M @ x), x, atol=1e-9)
    S = E.matrix_sqrt(M)
    assert np.allclose(S.sum(axis=0), 1) and np.all(S >= 0)
    assert np.allclose(S @ S, M, atol=1e-3)


def test_sector_probs_positions():
    pytest.importorskip("sklearn")
    import qg_sector_echo_mitigation_qg as E

    p = np.zeros(32)
    p[16] = 0.6
    p[1] = 0.2
    p[0] = 0.2
    assert np.allclose(E.sector_probs(p), [0.75, 0, 0, 0, 0.25])
