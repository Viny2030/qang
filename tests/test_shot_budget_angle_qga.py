"""Tests for examples/shot_budget_angle_qga.py (P1: shot budget vs angle in qga)."""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("qiskit")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shot_budget_angle_qga import (  # noqa: E402
    empirical_angle_std_qga,
    run_demo,
    shot_budget_qg,
    shot_budget_qga,
)


def test_shot_budget_qga_is_state_independent_constant():
    n = shot_budget_qga(5.0)
    assert np.isclose(n, (159.15494309189535 / 5.0) ** 2)


def test_shot_budget_qg_depends_on_angle():
    # Near pole, qg_Z variance is tiny; at equator it is maximal.
    n_pole = shot_budget_qg(0.02, 10.0)
    n_equator = shot_budget_qg(0.02, 250.0)
    assert n_pole < n_equator
    assert n_equator == 2500


def test_empirical_angle_std_matches_delta_method_away_from_poles():
    # At equator, delta method should be accurate.
    n = int(np.ceil(shot_budget_qga(5.0)))
    emp = empirical_angle_std_qga(250.0, n, n_trials=5000, seed=3)
    assert abs(emp - 5.0) < 0.35


def test_empirical_angle_std_breaks_down_near_pole():
    # Fixed budget: delta method is inaccurate near poles due to clipping/rare events.
    n = int(np.ceil(shot_budget_qga(5.0)))
    emp_pole = empirical_angle_std_qga(10.0, n, n_trials=8000, seed=3)
    assert abs(emp_pole - 5.0) > 0.5


def test_run_demo_structure():
    demo = run_demo()
    assert demo["target_qga"] == 5.0
    assert len(demo["rows"]) >= 5
    for row in demo["rows"]:
        assert row["shot_budget_qga"] == demo["rows"][0]["shot_budget_qga"]
        assert row["delta_method_angle_std_qga"] > 0
