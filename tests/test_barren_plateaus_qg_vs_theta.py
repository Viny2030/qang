"""
Tests for examples/barren_plateaus_qg_vs_theta.py: connecting the
barren-plateau phenomenon (theta-space) to the Section 4.1 coordinate
singularity (qg-space).

Checked:
  1. The standard cost-function-dependent barren plateau reproduces in
     theta-space for qang.ansatze.hardware_efficient_ansatz against a
     global Z^n cost: gradient variance decreases sharply and
     monotonically as n_qubits grows, with a clearly negative
     log-linear fit slope, at a fixed seed (deterministic, not flaky).
  2. Regularized qg-space conversions (clipped, Tikhonov) do not erase
     that decay -- it survives (checked qualitatively: variance still
     drops by more than an order of magnitude from n=4 to n=10) because
     away from the poles the conversion factor is a generic O(1) number.
  3. The deterministic near-pole demonstration: fixing one parameter at
     progressively smaller values approaching a pole (theta = 1e-4,
     then 1e-6) while holding the theta-space gradient itself roughly
     constant (parameter-shift evaluates away from the pole, at
     theta +/- pi/2), the raw qg-space gradient grows in proportion to
     1/theta (unboundedly), the clipped conversion stays essentially
     constant (correctly bounded by the eps floor), and the Tikhonov
     conversion shrinks towards zero -- exactly the three documented
     behaviors of qang.gradients' inverse-Jacobian variants, now shown
     to persist unchanged inside a many-qubit barren-plateau ansatz.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from barren_plateaus_qg_vs_theta import (
    gradient_variance_by_qubit_count,
    near_pole_gradient_blowup,
    parameter_shift_grad,
)


N_QUBITS_LIST = [4, 6, 8, 10]


def test_theta_space_gradient_variance_decreases_sharply_with_n_qubits():
    variances = gradient_variance_by_qubit_count(
        N_QUBITS_LIST, reps=1, n_samples=100, seed=0, space="theta"
    )
    values = [variances[n] for n in N_QUBITS_LIST]

    # strictly decreasing at this fixed seed
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1))

    # a real, substantial effect, not numerical noise
    assert values[0] / values[-1] > 5.0

    log_var = np.log(values)
    slope, _ = np.polyfit(N_QUBITS_LIST, log_var, 1)
    assert slope < -0.15


@pytest.mark.parametrize("space", ["clipped", "tikhonov"])
def test_regularized_qg_space_still_shows_exponential_decay(space):
    """Away from the poles, the regularized conversion factor is a
    generic O(1) number, so the barren-plateau decay survives -- it is
    NOT erased by reparameterizing into (regularized) qg-space."""
    variances = gradient_variance_by_qubit_count(
        N_QUBITS_LIST, reps=1, n_samples=100, seed=0, space=space
    )
    values = [variances[n] for n in N_QUBITS_LIST]
    # still drops by more than an order of magnitude from n=4 to n=10
    assert values[0] / values[-1] > 10.0


def test_near_pole_raw_gradient_grows_like_inverse_theta():
    near = near_pole_gradient_blowup(6, reps=1, seed=0, theta_pole_value=1e-4)
    far = near_pole_gradient_blowup(6, reps=1, seed=0, theta_pole_value=1e-6)

    # the underlying theta-space gradient itself barely changes (evaluated
    # at theta +/- pi/2, away from the pole either way)
    assert near["theta"] == pytest.approx(far["theta"], rel=0.05)

    # raw qg-space grows by very close to the same 100x factor as
    # theta_pole_value shrank by (1e-4 -> 1e-6), i.e. it scales like 1/theta
    ratio = abs(far["qg_raw"]) / abs(near["qg_raw"])
    assert ratio == pytest.approx(100.0, rel=0.05)
    assert abs(far["qg_raw"]) > 1000.0 * abs(far["theta"])  # genuinely unbounded blow-up


def test_near_pole_clipped_gradient_stays_bounded():
    near = near_pole_gradient_blowup(6, reps=1, seed=0, theta_pole_value=1e-4, eps=0.05)
    far = near_pole_gradient_blowup(6, reps=1, seed=0, theta_pole_value=1e-6, eps=0.05)
    # essentially unchanged: both are floored at the same eps
    assert abs(near["qg_clipped"]) == pytest.approx(abs(far["qg_clipped"]), rel=1e-3)
    # and bounded by (roughly) 1/eps times the theta-space gradient magnitude
    assert abs(near["qg_clipped"]) < (1.0 / 0.05 + 1.0) * abs(near["theta"])


def test_near_pole_tikhonov_gradient_shrinks_toward_zero():
    near = near_pole_gradient_blowup(6, reps=1, seed=0, theta_pole_value=1e-4, eps=0.05)
    far = near_pole_gradient_blowup(6, reps=1, seed=0, theta_pole_value=1e-6, eps=0.05)
    assert abs(far["qg_tikhonov"]) < abs(near["qg_tikhonov"])
    assert abs(far["qg_tikhonov"]) < 1e-3


def test_parameter_shift_grad_matches_finite_difference():
    rng = np.random.default_rng(0)
    n_qubits, reps = 4, 1
    params = rng.uniform(0.0, 2.0 * np.pi, size=n_qubits * (reps + 1))
    from barren_plateaus_qg_vs_theta import global_z_cost

    exact = parameter_shift_grad(n_qubits, reps, params, param_index=0)
    h = 1e-6
    shift = np.zeros_like(params)
    shift[0] = h
    finite_diff = (
        global_z_cost(n_qubits, reps, params + shift) - global_z_cost(n_qubits, reps, params - shift)
    ) / (2 * h)
    assert exact == pytest.approx(finite_diff, abs=1e-5)


if __name__ == "__main__":
    print("Run via `pytest tests/test_barren_plateaus_qg_vs_theta.py -v` for full coverage.")
    variances = gradient_variance_by_qubit_count(N_QUBITS_LIST, reps=1, n_samples=50, seed=0, space="theta")
    assert variances[4] > variances[10]
    print("Smoke check passed.")
