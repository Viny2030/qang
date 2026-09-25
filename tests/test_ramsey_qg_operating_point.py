"""
Tests for examples/ramsey_qg_operating_point.py: Ramsey Fisher information
in qg units and few-shot behaviour at the pole vs the mid-fringe.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("scipy")

import ramsey_qg_operating_point as R  # noqa: E402


def test_fisher_in_qg_units_matches_the_direct_formula():
    for V in (1.0, 0.9, 0.5):
        for x in np.linspace(0.1, 3.0, 12):
            q = V * math.cos(x)
            assert R.fisher(x, 0.0, V) == pytest.approx(R.fisher_from_qg(q, V), rel=1e-12)
    assert np.allclose(R.fisher_from_qg(np.linspace(-0.99, 0.99, 50), 1.0), 1.0)


def test_mid_fringe_is_optimal_without_readout_asymmetry():
    x, q, f, fq = R.optimal_operating_point(0.9, 0.0)
    assert q == pytest.approx(0.0, abs=1e-6) and f == pytest.approx(0.81, rel=1e-6)


def test_asymmetric_readout_moves_the_optimum_but_barely_helps():
    x, q, f, fq = R.optimal_operating_point(0.9, 0.025, 0.945)
    assert q > 0.05 and f / fq - 1 < 0.005
    x, q, f, fq = R.optimal_operating_point(0.9, 0.09, 0.89)
    assert f / fq - 1 < 0.03


def test_few_shots_pole_vs_mid_fringe():
    r1 = R.few_shot(10, 1.0, reps=2000)
    assert r1[("pole", "all same")] > 0.7 and r1[("mid-fringe", "all same")] < 0.05
    assert r1[("pole", "Bayes")] < 0.6 * r1[("pole", "plug-in")]
    big = R.few_shot(1000, 0.9, reps=2000)
    assert big[("pole", "plug-in")] > 2 * big[("mid-fringe", "plug-in")]
    assert big[("mid-fringe", "plug-in")] == pytest.approx(big[("mid-fringe", "CRB")], rel=0.1)
