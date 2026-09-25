"""
Tests for examples/qec_syndrome_drift_tracking_qg.py: syndrome ancillas
as a continuous qg witness, and adaptive code switching under drift.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

import qec_syndrome_drift_tracking_qg as S  # noqa: E402


@pytest.mark.parametrize("g,p", [(0.005, 0.0), (0.01, 0.005), (0.02, 0.02), (0.05, 0.001)])
def test_syndrome_rates_are_powers_of_the_qg_x_witness(g, p):
    assert S.leung_syndrome_rates(g, p) == pytest.approx(S.leung_rates_closed_form(g, p), abs=1e-13)
    assert S.phase_rate(g, p) == pytest.approx(S.phase_rate_closed_form(g, p), abs=1e-13)
    qg_x = math.sqrt(1 - g) * (1 - 2 * p)
    assert 1 - 2 * S.leung_rates_closed_form(g, p)[1] == pytest.approx(qg_x**4)
    assert 1 - 2 * S.phase_rate_closed_form(g, p) == pytest.approx(qg_x**2)
    assert 1 - 2 * S.leung_rates_closed_form(g, p)[0] == pytest.approx(1 - 2 * g * (1 - g))


def test_estimators_invert_the_closed_forms():
    g, p = 0.012, 0.004
    rz, rx = S.leung_rates_closed_form(g, p)
    n = 10**9
    assert S.estimate_from_leung(rz * 2 * n, rx * n, n) == pytest.approx((g, p), rel=1e-6)
    assert S.estimate_p_from_phase(S.phase_rate_closed_form(g, p) * 2 * n, n, g) == pytest.approx(p, rel=1e-6)


def test_one_window_separates_the_three_regimes():
    rng = np.random.default_rng(0)
    for ratio, expected in zip(S.LEVELS, ("leung", "none", "phase")):
        g, p = S.rates_from_times(S.T_ROUND, ratio)
        rz, rx = S.leung_rates_closed_form(g, p)
        picks = [S._best(*S.estimate_from_leung(rng.binomial(2000, rz), rng.binomial(1000, rx), 1000))
                 for _ in range(40)]
        assert np.mean([c == expected for c in picks]) > 0.8


def test_gamma_pool_combines_probe_and_syndrome_information():
    pool = S.GammaPool()
    pool.add_probe(-1 + 2 * 0.01, 10**6)
    pool.add_zz(round(2e6 * 0.01 * 0.99), 2 * 10**6)
    assert pool.estimate() == pytest.approx(0.01, rel=1e-3)


def test_telegraph_series_visits_all_levels():
    s = S.telegraph_series(400, seed=1)
    assert set(s) == set(S.LEVELS)


def test_strategy_ranking():
    t = S.compare(n_windows=200, seeds=range(2))
    r = {k: v["regret"] for k, v in t.items()}
    assert r["oracle"] == 0
    assert r["hybrid"] < r["syndrome tracking"] < r["probe every K"] < r["fixed none"] < r["calibrate once"]
    assert r["probe every window"] < 0.15 and r["hybrid"] < 0.15
    assert t["syndrome tracking"]["extra_shots_per_window"] < 50
    assert t["hybrid"]["extra_shots_per_window"] < 0.5 * t["probe every window"]["extra_shots_per_window"]
    assert r["fixed leung"] > 1.0
