"""
Tests for examples/pole_damped_gradient_descent_robustness.py: the
statistical study behind qang.gradients' ``theta_pole_damped`` space.

Checked:
  1. run_optimizer reproduces the exact deterministic finding already
     covered in tests/test_gradients.py, using this file's own (slightly
     more general) implementation.
  2. success_and_trapped_rates, at a safe learning rate, gives plain and
     damped success rates that are both high and close to each other.
  3. success_and_trapped_rates, at a badly-tuned (aggressive) learning
     rate, gives damped a success rate several times higher than plain's
     -- reproduced across three different random seeds, so this is not a
     seed-specific artifact.
  4. The "trapped" rate is honestly nonzero at aggressive learning rates
     (this technique is not claimed to be failure-free) but stays a small
     minority of trials.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import math

import pytest

from pole_damped_gradient_descent_robustness import (
    random_near_pole_landscape,
    run_optimizer,
    success_and_trapped_rates,
)
from qang.gradients import toy_vqe_energy


def test_run_optimizer_damped_survives_aggressive_lr_where_plain_fails():
    h_z, h_x = 1.0, -0.3
    true_min = -math.hypot(h_z, h_x)
    theta0 = 0.02

    final_plain = run_optimizer(theta0, lr=3.0, n_iters=300, damped=False, h_z=h_z, h_x=h_x)
    final_damped = run_optimizer(theta0, lr=3.0, n_iters=300, damped=True, h_z=h_z, h_x=h_x)

    assert final_plain is not None
    assert abs(toy_vqe_energy(final_plain, h_z, h_x) - true_min) > 1e-3
    assert final_damped is not None
    assert toy_vqe_energy(final_damped, h_z, h_x) == pytest.approx(true_min, abs=1e-6)


def test_random_near_pole_landscape_starts_close_to_a_pole():
    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(50):
        h_z, h_x, true_min, theta0 = random_near_pole_landscape(rng)
        assert true_min == pytest.approx(-math.hypot(h_z, h_x), abs=1e-12)
        dist_to_pole = min(abs(theta0 - 0.0), abs(theta0 - math.pi))
        assert dist_to_pole < 0.06


def test_success_rates_are_close_at_a_safe_learning_rate():
    rates = success_and_trapped_rates(lr=0.3, n_iters=150, seed=42, n_trials=300)
    assert rates["plain"] > 0.95
    assert rates["damped"] > 0.9
    assert abs(rates["plain"] - rates["damped"]) < 0.1


@pytest.mark.parametrize("seed", [42, 7, 123])
def test_damped_success_rate_beats_plain_at_an_aggressive_learning_rate(seed):
    rates = success_and_trapped_rates(lr=3.0, n_iters=150, seed=seed, n_trials=300)
    assert rates["plain"] < 0.2
    assert rates["damped"] > 2.0 * rates["plain"]


@pytest.mark.parametrize("lr", [1.0, 2.0, 3.0, 5.0, 10.0])
def test_damped_success_rate_exceeds_plain_across_the_aggressive_regime(lr):
    rates = success_and_trapped_rates(lr=lr, n_iters=150, seed=42, n_trials=300)
    assert rates["damped"] > rates["plain"]


def test_trapped_rate_is_honestly_nonzero_but_a_small_minority_at_aggressive_lr():
    """The pole-trapping failure mode is real and should not be hidden --
    but it should stay a small minority of trials, not dominate."""
    rates = success_and_trapped_rates(lr=10.0, n_iters=150, seed=42, n_trials=300)
    assert rates["trapped"] > 0.0
    assert rates["trapped"] < 0.2


def test_trapped_rate_is_near_zero_at_a_safe_learning_rate():
    rates = success_and_trapped_rates(lr=1.0, n_iters=150, seed=42, n_trials=300)
    assert rates["trapped"] < 0.05


if __name__ == "__main__":
    print("Run via `pytest tests/test_pole_damped_gradient_descent_robustness.py -v`.")
    rates = success_and_trapped_rates(lr=3.0, n_iters=150, seed=42, n_trials=300)
    assert rates["damped"] > 2.0 * rates["plain"]
    print("Smoke check passed.")
