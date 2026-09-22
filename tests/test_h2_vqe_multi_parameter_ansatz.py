"""
Tests for examples/h2_vqe_multi_parameter_ansatz.py: a genuinely
two-parameter, entangled H2 VQE ansatz (as opposed to the single
parameter in tests/test_vqe_h2.py or the separable toy landscape in
tests/test_multi_parameter_pole_damped_vqe.py).

Checked:
  1. The two-parameter ansatz reduces exactly to the original
     single-parameter one (examples/vqe_h2_qg_vs_theta.py) when theta0 is
     fixed at pi, since Ry(pi)|0> = X|0> exactly.
  2. The parameter-shift-rule gradient for each parameter matches finite
     differences.
  3. Finding A (an honest cost): at a safe learning rate, theta_pole_damped
     needs more iterations than plain theta, because theta0's own true
     minimum sits exactly at the pole theta=pi.
  4. Finding B (the established benefit, now on a real, non-separable
     two-parameter molecular Hamiltonian): at a badly-tuned, aggressive
     learning rate, plain theta overshoots and fails while
     theta_pole_damped still reaches the exact FCI energy.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import math

import pytest

qiskit = pytest.importorskip("qiskit")

from h2_vqe_multi_parameter_ansatz import (
    CHEMICAL_ACCURACY,
    EXACT_FCI_ENERGY,
    h2_energy_grad_multi,
    h2_energy_multi,
    run_vqe_multi,
    steps_to_chemical_accuracy,
)

# The original single-parameter ansatz/energy, for the reduction check.
from vqe_h2_qg_vs_theta import h2_energy as h2_energy_single


@pytest.mark.parametrize("theta1", [0.1, 0.5, 1.0, 2.0, 2.5])
def test_theta0_at_pi_reduces_to_the_original_single_parameter_ansatz(theta1):
    """Ry(pi)|0> = X|0> exactly, so fixing theta0 = pi must reproduce the
    original single-parameter ansatz's energy bit for bit."""
    assert h2_energy_multi(math.pi, theta1) == pytest.approx(h2_energy_single(theta1), abs=1e-12)


@pytest.mark.parametrize("theta0,theta1", [(0.3, 0.7), (1.5, 2.3), (2.0, -0.5)])
def test_parameter_shift_gradient_matches_finite_difference(theta0, theta1):
    g0, g1 = h2_energy_grad_multi(theta0, theta1)
    h = 1e-6
    fd0 = (h2_energy_multi(theta0 + h, theta1) - h2_energy_multi(theta0 - h, theta1)) / (2 * h)
    fd1 = (h2_energy_multi(theta0, theta1 + h) - h2_energy_multi(theta0, theta1 - h)) / (2 * h)
    assert g0 == pytest.approx(fd0, abs=1e-6)
    assert g1 == pytest.approx(fd1, abs=1e-6)


def test_both_spaces_reach_fci_energy_at_a_moderate_learning_rate():
    """Sanity check: away from any aggressive-LR risk, both spaces
    eventually reach the exact FCI energy from this starting point."""
    theta0_0, theta1_0 = 1.0, 1e-6
    for space in ("theta", "theta_pole_damped"):
        energies = run_vqe_multi(space, theta0_0, theta1_0, lr=1.0, steps=300)
        assert energies[-1] == pytest.approx(EXACT_FCI_ENERGY, abs=1e-6)


@pytest.mark.parametrize("lr,plain_steps,damped_steps", [(0.05, 196, 603), (0.1, 97, 301), (0.3, 32, 99)])
def test_finding_a_theta_pole_damped_needs_more_iterations_at_safe_lr(lr, plain_steps, damped_steps):
    """Finding A: theta0's own optimum sits exactly at the pole theta=pi,
    so damping -- which shrinks theta0's step as it nears that target --
    costs real iterations here, more visibly than in the single-parameter
    study. Exact step counts are pinned down (not just "damped > plain")
    since this is a fully deterministic, non-random experiment."""
    theta0_0, theta1_0 = 1.0, 1e-6
    e_plain = run_vqe_multi("theta", theta0_0, theta1_0, lr=lr, steps=700)
    e_damped = run_vqe_multi("theta_pole_damped", theta0_0, theta1_0, lr=lr, steps=700)

    steps_plain = steps_to_chemical_accuracy(e_plain)
    steps_damped = steps_to_chemical_accuracy(e_damped)

    assert steps_plain == plain_steps
    assert steps_damped == damped_steps
    assert steps_damped > steps_plain  # the honest cost, stated plainly


def test_theta_pole_damped_eventually_converges_despite_the_extra_iterations():
    """The cost in test_finding_a is iterations, not failure: given enough
    steps, theta_pole_damped still reaches the exact FCI energy at a safe
    learning rate."""
    theta0_0, theta1_0 = 1.0, 1e-6
    energies = run_vqe_multi("theta_pole_damped", theta0_0, theta1_0, lr=0.05, steps=700)
    assert abs(energies[-1] - EXACT_FCI_ENERGY) < CHEMICAL_ACCURACY


@pytest.mark.parametrize("lr", [2.5, 3.0, 4.0])
def test_finding_b_plain_theta_fails_at_an_aggressive_learning_rate(lr):
    theta0_0, theta1_0 = 1.0, 1e-6
    energies = run_vqe_multi("theta", theta0_0, theta1_0, lr=lr, steps=500)
    assert abs(energies[-1] - EXACT_FCI_ENERGY) > CHEMICAL_ACCURACY


@pytest.mark.parametrize("lr", [2.5, 3.0, 4.0, 5.0])
def test_finding_b_theta_pole_damped_survives_the_same_aggressive_learning_rates(lr):
    """Finding B, confirmed on a real, non-separable two-parameter
    molecular Hamiltonian (not just the single-parameter case or the
    separable toy landscape): theta_pole_damped -- started from the exact
    same point where plain theta fails -- still reaches the exact FCI
    energy, and does so in a handful of steps."""
    theta0_0, theta1_0 = 1.0, 1e-6
    energies = run_vqe_multi("theta_pole_damped", theta0_0, theta1_0, lr=lr, steps=500)
    assert energies[-1] == pytest.approx(EXACT_FCI_ENERGY, abs=1e-6)
    steps = steps_to_chemical_accuracy(energies)
    assert steps is not None
    assert steps < 20


def test_run_vqe_multi_rejects_unknown_space():
    with pytest.raises(ValueError):
        run_vqe_multi("qg_raw", 1.0, 1e-6, lr=0.1, steps=5)


if __name__ == "__main__":
    print("Run via `pytest tests/test_h2_vqe_multi_parameter_ansatz.py -v`.")
    energies = run_vqe_multi("theta_pole_damped", 1.0, 1e-6, lr=3.0, steps=500)
    assert energies[-1] == pytest.approx(EXACT_FCI_ENERGY, abs=1e-6)
    print("Smoke check passed.")
