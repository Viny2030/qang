"""
Tests for examples/multi_parameter_pole_damped_vqe.py: the vector-of
-parameters demonstration built on qang.gradients.multi_param_gradient_descent.

Checked:
  1. run_scenario reproduces the exact deterministic four-parameter finding
     already covered (in general form) by tests/test_gradients.py's
     multi-parameter tests, using this example's own specific scenario.
  2. At an aggressive learning rate, plain gradient descent fails on both
     near-pole parameters (param0, param1) while leaving the two pole-free
     parameters (param2, param3) exactly at their own optimum.
  3. theta_pole_damped rescues both near-pole parameters at that same
     learning rate, without in any way disturbing the two pole-free ones.
  4. The honestly-scoped limitation: at a sufficiently aggressive learning
     rate (lr=5.0), the theta=pi pole parameter's damped result, while much
     closer to its true minimum than plain's, is not claimed to be exact --
     this is the same small pole-trapping failure mode already documented
     in examples/pole_damped_gradient_descent_robustness.py, now confirmed
     to also occur in the multi-parameter setting.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import math

import pytest

from multi_parameter_pole_damped_vqe import H_X, H_Z, THETA0, TRUE_MINS, run_scenario


def test_true_mins_match_the_amplitude_formula():
    for hz, hx, true_min in zip(H_Z, H_X, TRUE_MINS):
        assert true_min == pytest.approx(-math.hypot(hz, hx), abs=1e-12)


@pytest.mark.parametrize("lr", [0.3, 1.0])
def test_both_spaces_succeed_at_a_safe_learning_rate(lr):
    plain_energies, damped_energies = run_scenario(lr)
    for i in range(4):
        assert plain_energies[i] == pytest.approx(TRUE_MINS[i], abs=1e-6)
        assert damped_energies[i] == pytest.approx(TRUE_MINS[i], abs=1e-6)


@pytest.mark.parametrize("lr", [2.0, 3.0])
def test_plain_fails_on_near_pole_parameters_only(lr):
    plain_energies, _ = run_scenario(lr)
    # param0, param1: near a pole -- plain fails to reach the true minimum
    assert abs(plain_energies[0] - TRUE_MINS[0]) > 1e-3
    assert abs(plain_energies[1] - TRUE_MINS[1]) > 1e-3
    # param2, param3: pole-free, already at their optimum -- plain leaves
    # them there untouched
    assert plain_energies[2] == pytest.approx(TRUE_MINS[2], abs=1e-9)
    assert plain_energies[3] == pytest.approx(TRUE_MINS[3], abs=1e-9)


@pytest.mark.parametrize("lr", [2.0, 3.0])
def test_damped_rescues_near_pole_parameters_without_disturbing_the_others(lr):
    _, damped_energies = run_scenario(lr)
    assert damped_energies[0] == pytest.approx(TRUE_MINS[0], abs=1e-6)
    assert damped_energies[1] == pytest.approx(TRUE_MINS[1], abs=1e-6)
    assert damped_energies[2] == pytest.approx(TRUE_MINS[2], abs=1e-9)
    assert damped_energies[3] == pytest.approx(TRUE_MINS[3], abs=1e-9)


def test_pole_free_parameters_are_undisturbed_even_at_a_very_aggressive_lr():
    """Regardless of how badly the near-pole parameters behave, a
    parameter's own damping factor depends only on its own theta -- never
    on any other parameter -- so param2 and param3 must stay exactly at
    their own optimum even at lr=5.0."""
    plain_energies, damped_energies = run_scenario(5.0)
    assert plain_energies[2] == pytest.approx(TRUE_MINS[2], abs=1e-9)
    assert plain_energies[3] == pytest.approx(TRUE_MINS[3], abs=1e-9)
    assert damped_energies[2] == pytest.approx(TRUE_MINS[2], abs=1e-9)
    assert damped_energies[3] == pytest.approx(TRUE_MINS[3], abs=1e-9)


def test_honest_limitation_pole_trapping_can_still_occur_at_extreme_lr():
    """At lr=5.0, param1 (near the theta=pi pole) is not guaranteed an
    exact recovery -- this checks the honestly-scoped claim: damped stays
    much closer to the true minimum than plain, without asserting the
    damped result is numerically exact."""
    plain_energies, damped_energies = run_scenario(5.0)
    true_min = TRUE_MINS[1]
    plain_error = abs(plain_energies[1] - true_min)
    damped_error = abs(damped_energies[1] - true_min)
    assert damped_error < 0.01  # close, though not necessarily exact
    assert damped_error < plain_error / 10  # still a dramatic improvement


if __name__ == "__main__":
    print("Run via `pytest tests/test_multi_parameter_pole_damped_vqe.py -v`.")
    plain_energies, damped_energies = run_scenario(3.0)
    assert damped_energies[0] == pytest.approx(TRUE_MINS[0], abs=1e-6)
    print("Smoke check passed.")
