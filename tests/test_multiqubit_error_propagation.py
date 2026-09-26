"""
Tests for the mixed-state and multi-qubit error propagation in
qang.statistics and examples/multiqubit_error_propagation_qg.py (§45).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from qang import statistics as S  # noqa: E402

import multiqubit_error_propagation_qg as M  # noqa: E402


@pytest.mark.parametrize("theta", (0.3, 1.0, math.pi / 2, 2.5))
def test_mixed_formula_reduces_to_section_5_for_pure_states(theta):
    assert S.propagated_theta_variance_mixed(theta, 1.0, 1000) == pytest.approx(
        S.propagated_theta_variance(theta, 1000), rel=1e-9)


@pytest.mark.parametrize("r", (0.5, 0.8, 0.95))
def test_mixed_formula_closed_forms(r):
    n = 500
    assert S.propagated_theta_variance_mixed(math.pi / 2, r, n) == pytest.approx(S.theta_qcrb_variance(r, n))
    th = 0.4
    v = S.propagated_theta_variance_mixed(th, r, n)
    assert v == pytest.approx((1 + (1 - r * r) / (r * r * math.sin(th) ** 2)) / n)
    # inverse of the Ramsey Fisher information of §36 with V = r, qg = r cos(theta)
    q = r * math.cos(th)
    assert 1 / (n * v) == pytest.approx((r * r - q * q) / (1 - q * q))
    assert S.propagated_theta_variance_mixed(0.0, r, n) == float("inf")


def test_mixed_formula_matches_sampling_away_from_the_pole():
    for th in (math.pi / 2, math.pi / 4, math.pi / 8):
        for r in (1.0, 0.95, 0.8):
            v, clipped = M.sampled_theta_variance(th, r, 10000, trials=3000)
            assert clipped == 0
            assert v == pytest.approx(S.propagated_theta_variance_mixed(th, r, 10000), rel=0.1)


def test_estimator_sticks_to_the_pole_for_mixed_states():
    v, clipped = M.sampled_theta_variance(math.pi / 64, 0.95, 10000, trials=3000)
    assert clipped > 0.3
    _, clipped_pure = M.sampled_theta_variance(math.pi / 64, 1.0, 10000, trials=3000)
    assert clipped_pure < 0.01  # only when no minority outcome at all (e^-6)


def test_covariance_matches_multinomial_sampling():
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(8))
    q, cov = S.qg_covariance(p, 3, 200)
    z = S._z_values(3)
    counts = rng.multinomial(200, p, size=20000)
    qs = counts @ z.T / 200
    assert np.allclose(qs.mean(axis=0), q, atol=0.01)
    assert np.allclose(np.cov(qs.T), cov, atol=4e-4)


def test_product_state_has_diagonal_covariance():
    p = M.product_state(math.pi / 3)
    q, cov = S.qg_covariance(p, 4, 1)
    assert np.allclose(q, 0.5)
    assert np.allclose(cov, np.diag(np.full(4, 0.75)))


def test_register_witness_extremes():
    n = 1000
    assert S.register_witness_variance(M.ghz(), 4, n) == pytest.approx(1 / n)
    assert S.register_witness_variance(M.ghz(), 4, n, independent=True) == pytest.approx(1 / (4 * n))
    for k in (1, 2, 3):
        assert S.register_witness_variance(M.dicke(k), 4, n) == pytest.approx(0.0, abs=1e-15)
    t = M.table_registers(trials=1500)
    for name, (true, naive, sampled) in t.items():
        assert sampled == pytest.approx(true, rel=0.08, abs=2e-4), name


def test_leak_detection_needs_ten_times_fewer_shots_with_correct_error_bars():
    shift, n_true, n_naive = M.shots_for_detection()
    assert shift == pytest.approx(0.02, abs=1e-9)
    assert 400 < n_true < 480
    assert 9 < n_naive / n_true < 11
