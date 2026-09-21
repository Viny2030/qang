"""
Tests for the H2 VQE example (examples/vqe_h2_qg_vs_theta.py), extending
Future Research Direction #3 (qang.gradients' toy benchmark) to a real
molecule.

Five things are checked:

  1. The minimal H2 ansatz, optimized in theta-space, reaches the exact
     FCI ground-state energy (independently cross-checked against
     PySCF + qiskit-nature in the docstring/derivation, not re-derived
     here to keep this test's dependencies light -- see the example
     file's module docstring for how H2_ELECTRONIC was obtained).
  2. The exact parameter-shift-rule gradient matches finite differences.
  3. Headline finding #1: starting from the physical Hartree-Fock
     starting point (theta ~= 0, the natural start of any real VQE run),
     every qg-space variant (raw, clipped, Tikhonov) is trapped exactly
     at the Hartree-Fock energy and recovers zero correlation energy --
     not a slow convergence, a hard zero -- because arccos()'s range
     [0, pi] cannot reach the true minimum, which lies outside it. This
     is a second, distinct limitation from the Jacobian singularity
     already documented in Section 4.1, and it is demonstrated here on
     a real chemistry problem rather than a constructed toy case.
  4. The fifth space, theta_pole_damped, escapes finding #3 entirely
     (it never leaves theta-space) and, from the same Hartree-Fock start,
     reaches the exact FCI energy at a safe learning rate.
  5. Headline finding #2: at a badly-tuned, too-aggressive learning rate,
     plain theta-space optimization diverges and never reaches chemical
     accuracy, while theta_pole_damped -- started from the exact same
     point -- still reaches the exact FCI energy.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import math

import pytest

qiskit = pytest.importorskip("qiskit")

from vqe_h2_qg_vs_theta import (
    CHEMICAL_ACCURACY,
    EXACT_FCI_ENERGY,
    h2_energy,
    h2_energy_grad,
    run_vqe,
    steps_to_chemical_accuracy,
)


def test_theta_space_reaches_exact_fci_energy_from_hf_start():
    energies = run_vqe("theta", theta0=1e-6, lr=0.3, steps=300)
    assert energies[-1] == pytest.approx(EXACT_FCI_ENERGY, abs=1e-6)


def test_theta_space_reaches_chemical_accuracy_quickly():
    energies = run_vqe("theta", theta0=1e-6, lr=0.3, steps=300)
    steps = steps_to_chemical_accuracy(energies)
    assert steps is not None
    assert steps < 20  # reaches it within a handful of steps in practice


@pytest.mark.parametrize("theta", [0.1, 0.7, 1.5, 2.3, 3.0])
def test_parameter_shift_gradient_matches_finite_difference(theta):
    exact = h2_energy_grad(theta)
    h = 1e-6
    finite_diff = (h2_energy(theta + h) - h2_energy(theta - h)) / (2 * h)
    assert exact == pytest.approx(finite_diff, abs=1e-6)


@pytest.mark.parametrize("space", ["qg_raw", "qg_clipped", "qg_tikhonov"])
def test_qg_space_trapped_at_hartree_fock_energy_from_hf_start(space):
    """The headline finding: all three qg-space variants recover exactly
    zero correlation energy from the physical HF starting point, because
    the true minimum lies outside arccos()'s reachable range [0, pi]."""
    hf_energy = h2_energy(0.0)
    energies = run_vqe(space, theta0=1e-6, lr=0.3, steps=300)
    final_energy = energies[-1]

    # trapped: final energy matches the HF energy, not the FCI energy
    assert final_energy == pytest.approx(hf_energy, abs=1e-6)
    assert abs(final_energy - EXACT_FCI_ENERGY) > CHEMICAL_ACCURACY

    # never even transiently improves beyond the HF energy from this start
    assert all(e >= hf_energy - 1e-9 for e in energies)


def _best_energy_in_0_pi(grid_points: int = 20000) -> float:
    """Best (lowest) energy over theta in [0, pi], via a fine grid scan --
    deliberately dependency-free (no scipy) so this test only needs
    qiskit, matching the rest of the suite."""
    best = h2_energy(0.0)
    for i in range(grid_points + 1):
        theta = math.pi * i / grid_points
        e = h2_energy(theta)
        if e < best:
            best = e
    return best


def test_theta_pole_damped_reaches_exact_fci_energy_at_a_safe_learning_rate():
    """theta_pole_damped never leaves theta-space, so it does not suffer
    the arccos-range trapping checked above -- it should behave like
    plain theta-space optimization at a safe learning rate."""
    energies = run_vqe("theta_pole_damped", theta0=1e-6, lr=0.3, steps=300)
    assert energies[-1] == pytest.approx(EXACT_FCI_ENERGY, abs=1e-6)


@pytest.mark.parametrize("lr", [3.0, 5.0])
def test_plain_theta_space_diverges_at_a_badly_tuned_learning_rate(lr):
    """Headline finding #2, half 1: plain theta-space gradient descent,
    started from the exact same Hartree-Fock point that converges cleanly
    at lr=0.3, overshoots and never reaches chemical accuracy once the
    learning rate is badly tuned (too aggressive)."""
    energies = run_vqe("theta", theta0=1e-6, lr=lr, steps=300)
    assert abs(energies[-1] - EXACT_FCI_ENERGY) > CHEMICAL_ACCURACY


@pytest.mark.parametrize("lr", [0.3, 1.0, 2.0, 3.0, 5.0])
def test_theta_pole_damped_reaches_fci_energy_across_safe_and_aggressive_lr(lr):
    """Headline finding #2, half 2: theta_pole_damped, started from the
    same Hartree-Fock point, reaches the exact FCI energy at every one of
    these learning rates -- including the two (3.0, 5.0) where plain
    theta-space diverges (checked above)."""
    energies = run_vqe("theta_pole_damped", theta0=1e-6, lr=lr, steps=300)
    assert energies[-1] == pytest.approx(EXACT_FCI_ENERGY, abs=1e-6)


def test_true_minimum_lies_outside_the_arccos_reachable_branch():
    """Sanity check underlying the headline finding: the best energy
    reachable from anywhere inside theta in [0, pi] (arccos's range) is
    the Hartree-Fock energy itself -- the entire correlation energy of
    H2 lives in the other half of the circle."""
    hf_energy = h2_energy(0.0)
    best_in_branch = _best_energy_in_0_pi()
    assert best_in_branch == pytest.approx(hf_energy, abs=1e-4)
    assert abs(best_in_branch - EXACT_FCI_ENERGY) > CHEMICAL_ACCURACY


if __name__ == "__main__":
    print("Run via `pytest tests/test_vqe_h2.py -v` for full parametrized coverage.")
    energies = run_vqe("theta", theta0=1e-6, steps=300)
    assert energies[-1] == pytest.approx(EXACT_FCI_ENERGY, abs=1e-6)
    print("Smoke check passed.")
