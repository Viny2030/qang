"""Tests for qang.proportions (§125): the qang angle as the arcsine transformation."""

import math
import os
import sys

import pytest

from qang import proportions as PR
from qang.core import Qang

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_angle_is_pi_minus_classical_arcsine_and_matches_born_rule():
    for p in (0.0, 0.03, 0.25, 0.5, 0.81, 1.0):
        th = PR.theta_from_proportion(p)
        assert math.isclose(th, math.pi - PR.arcsine(p), abs_tol=1e-12)
        assert math.isclose(PR.proportion_from_theta(th), p, abs_tol=1e-12)
        assert math.isclose(math.cos(th / 2) ** 2, p, abs_tol=1e-12)  # p0 = cos^2(theta/2)


def test_cohens_h_is_a_difference_of_qang_angles():
    p1, p2 = 0.834, 0.811
    assert math.isclose(PR.cohens_h(p1, p2),
                        PR.theta_from_proportion(p2) - PR.theta_from_proportion(p1), abs_tol=1e-12)
    assert PR.cohens_h(0.5, 0.5) == 0.0


def test_interval_contains_estimate_and_stays_in_unit_interval():
    for k, n in ((0, 20), (1, 50), (25, 50), (49, 50), (20, 20), (568, 681)):
        p, lo, hi = PR.proportion_interval(k, n)
        assert 0.0 <= lo <= p <= hi <= 1.0
    p, lo, hi = PR.proportion_interval(0, 50)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi > 0.0  # does not collapse at k = 0, unlike Wald


def test_compare_proportions_is_antisymmetric():
    a = PR.compare_proportions(568, 681, 421, 519)
    b = PR.compare_proportions(421, 519, 568, 681)
    assert math.isclose(a["h"], -b["h"]) and math.isclose(a["p_value"], b["p_value"])
    assert a["h_low"] < a["h"] < a["h_high"]


def test_z_value_and_variance():
    assert PR.z_value(0.95) == pytest.approx(1.959964, abs=1e-5)
    assert PR.theta_variance(100) == pytest.approx(1 / 100.5)
    assert PR.theta_variance(100, anscombe=False) == pytest.approx(0.01)


def test_bad_inputs():
    with pytest.raises(ValueError):
        PR.theta_from_proportion(1.2)
    with pytest.raises(ValueError):
        PR.proportion_interval(5, 3)
    with pytest.raises(ValueError):
        PR.z_value(1.0)


def test_example_verdict_and_data_file():
    import proportions_angular_qg as E

    good = {"S1": {0.5: 1.0}, "S2": {(30, 0.5): 0.95}, "S3": {0.02: (0.9, 0.3)}, "S4": {(100, 0.5): 1.02}}
    assert E.verdict(good) == {"S1": True, "S2": True, "S3": True, "S4": True}
    rows = E.comprar_illustration()
    assert len(rows) == 25 and all(0 <= k <= n for _, n, k, *_ in rows)


def test_qang_core_agrees():
    # the same angle as the core unit: qg_Z = cos(theta) = 2 p0 - 1
    p = 0.3
    th = PR.theta_from_proportion(p)
    assert math.isclose(math.cos(th), 2 * p - 1, abs_tol=1e-12)
    assert Qang is not None
