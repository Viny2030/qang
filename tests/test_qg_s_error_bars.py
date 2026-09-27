"""
Tests for the qg_S interval (qang.statistics.qg_s_estimate) and
examples/qg_s_error_bars.py (RESEARCH_NOTES §51).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

pytest.importorskip("scipy")

from qang import statistics as S  # noqa: E402

import qg_s_error_bars as E  # noqa: E402


def test_mapping_through_the_identity():
    assert S.qg_s_from_qg_z_interval(-0.2, 0.3) == (pytest.approx(E.h2v(np.array(0.65)).item()), 1.0)
    lo, hi = S.qg_s_from_qg_z_interval(0.5, 0.8)
    assert lo == pytest.approx(E.h2v(np.array(0.9)).item()) and hi == pytest.approx(E.h2v(np.array(0.75)).item())
    lo, hi = S.qg_s_from_qg_z_interval(-0.8, -0.5)
    assert lo < hi <= 1.0


def test_wald_collapses_at_the_equator_and_qg_wilson_does_not():
    _, lo, hi = S.qg_s_estimate(50, 100, "wald")
    assert hi - lo == 0.0
    _, lo, hi = S.qg_s_estimate(50, 100)
    assert hi == 1.0 and lo < 0.98


def test_miller_madow_point_estimate():
    est, _, _ = S.qg_s_estimate(30, 100)
    p = 0.3
    plug = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    assert est == pytest.approx(plug + 1 / (200 * math.log(2)))
    assert S.qg_s_estimate(100, 100)[0] == 0.0


def test_bayes_closed_form_matches_sampling():
    rng = np.random.default_rng(0)
    for k0, n in ((3, 20), (10, 20), (70, 100)):
        mc = E.h2v(rng.beta(k0 + 1, n - k0 + 1, size=200000)).mean()
        assert E.bayes_mean_single(k0, n) == pytest.approx(mc, abs=2e-3)
    c = np.array([5, 0, 3, 12])
    mc = E.entropy_bits(rng.dirichlet(c + 1.0, size=200000)).mean()
    assert E.bayes_mean_dirichlet(c) == pytest.approx(mc, abs=2e-3)


def test_qg_wilson_covers_where_wald_and_bootstrap_fail():
    r = E.single_qubit(0.99, 20, trials=600, boot=150, post=400)
    assert r["coverage"]["qg_wilson"] > 0.85
    assert r["coverage"]["wald"] < 0.3 and r["coverage"]["bootstrap"] < 0.3
    r0 = E.single_qubit(0.0, 1000, trials=400, boot=150, post=400)
    assert r0["coverage"]["qg_wilson"] > 0.9
    assert r0["coverage"]["bootstrap"] < 0.4 and r0["coverage"]["bayes_haar"] == 0.0


def test_register_wald_mm_works_and_haar_prior_fails_near_uniform():
    r = E.register(0.9, 100, trials=300, boot=150, post=300)
    assert r["coverage"]["wald_mm"] > 0.9
    assert r["coverage"]["bayes_haar"] < 0.05
    assert r["coverage"]["bootstrap_mm"] < 0.75
    r0 = E.register(0.0, 100, trials=300, boot=150, post=300)
    assert abs(r0["bias"]["bayes_haar"]) < abs(r0["bias"]["plug-in"])
