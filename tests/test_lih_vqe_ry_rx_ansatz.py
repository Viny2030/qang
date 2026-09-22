bash

cat /tmp/claude-0/-home-claude/0f095fba-ee27-5ef0-bc6c-bd81ba55699a/scratchpad/qang_circuits/tests/test_lih_vqe_ry_rx_ansatz.py
Salida

"""
Tests for examples/lih_vqe_ry_rx_ansatz.py: a mixed Ry/Rx, four-parameter,
4-qubit LiH VQE ansatz -- extending
tests/test_h2_vqe_multi_parameter_ansatz.py (Ry-only, 2 qubits) to a
bigger real molecule and a second rotation axis.

Checked:
  1. The ansatz reduces exactly to qiskit-nature's own Hartree-Fock
     reference at (theta0, theta1, theta2, theta3) = (pi, pi, 0, 0):
     same energy as HF_ENERGY, and the HF/FCI gap is the expected,
     physically genuine correlation energy (not a bug).
  2. The parameter-shift-rule gradient matches finite differences for
     all four parameters, including theta3 -- the Rx-parametrized one --
     which is the concrete evidence that qang's parameter-shift and
     pole-damping machinery generalizes beyond Ry.
  3. Rx and Ry rotations from |0> give IDENTICAL single-qubit Z-basis
     populations (so qg_Z(theta), and therefore pole_damping_factor, are
     identical functions of theta for either axis) but produce DIFFERENT
     whole-register energies once entangled via CX -- the axis choice is
     a genuine physical degree of freedom, not a relabeling.
  4. Finding A: at safe learning rates, theta_pole_damped needs many more
     iterations than plain theta, because all four parameters' true
     optima sit at or very near a pole.
  5. Finding B: at aggressive learning rates, plain theta fails to
     converge while theta_pole_damped succeeds every time.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import math

import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qang.gradients import pole_damping_factor
from lih_vqe_ry_rx_ansatz import (
    ANSATZ_OPTIMUM,
    CHEMICAL_ACCURACY,
    EXACT_GROUND_STATE,
    HF_ENERGY,
    lih_ansatz,
    lih_energy,
    lih_energy_grad,
    run_vqe_lih,
    steps_to_sustained_convergence,
)


# --------------------------------------------------------------------- #
# 1. Hartree-Fock reduction and the correlation-energy gap
# --------------------------------------------------------------------- #
def test_hf_point_matches_the_official_hartree_fock_energy():
    e = lih_energy([math.pi, math.pi, 0.0, 0.0])
    assert e == pytest.approx(HF_ENERGY, abs=1e-9)


def test_hf_point_reduces_to_the_bare_computational_basis_state():
    """Ry(pi)|0> = X|0> exactly and theta2 = theta3 = 0 leaves qubits 2, 3
    at |0>, so the whole ansatz must collapse to the single computational
    basis state |0011> (qubits 0, 1 = |1>, qubits 2, 3 = |0>), matching
    qiskit-nature's own Hartree-Fock reference state for this active
    space bit for bit."""
    sv = Statevector.from_instruction(lih_ansatz([math.pi, math.pi, 0.0, 0.0]))
    probs = sv.probabilities_dict()
    assert probs.get("0011", 0.0) == pytest.approx(1.0, abs=1e-9)
    assert sum(p for bitstring, p in probs.items() if bitstring != "0011") == pytest.approx(0.0, abs=1e-9)


def test_hf_to_fci_gap_is_the_expected_correlation_energy_not_a_bug():
    """The gap between HF_ENERGY and EXACT_GROUND_STATE is small, positive
    (HF is a variational upper bound on the exact ground state), and
    below conventional chemical accuracy for this particular active
    space -- an honest, checked property of this specific reduction, not
    an assumption."""
    gap = HF_ENERGY - EXACT_GROUND_STATE
    assert gap > 0.0
    assert gap < CHEMICAL_ACCURACY


# --------------------------------------------------------------------- #
# 2. Parameter-shift gradient vs. finite differences, all four
#    parameters, including theta3 (Rx)
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "thetas",
    [
        [math.pi, math.pi, 0.0, 0.0],
        [2.5, 1.8, 0.7, 0.3],
        [0.5, 0.5, 0.5, 0.5],
        [1.0, 2.0, 3.0, 0.2],
    ],
)
def test_parameter_shift_gradient_matches_finite_difference_for_every_parameter(thetas):
    grads = lih_energy_grad(thetas)
    h = 1e-6
    for i in range(4):
        plus = list(thetas)
        plus[i] += h
        minus = list(thetas)
        minus[i] -= h
        fd = (lih_energy(plus) - lih_energy(minus)) / (2 * h)
        assert grads[i] == pytest.approx(fd, abs=1e-6)


def test_parameter_shift_gradient_matches_finite_difference_specifically_for_the_rx_parameter():
    """Isolates theta3 (the Rx-parametrized qubit) as the concrete,
    tested evidence that the parameter-shift rule is not specific to
    Ry -- it holds identically for any single-qubit Pauli rotation."""
    for theta3 in [0.1, 1.0, 2.0, 2.9, 4.5]:
        thetas = [1.2, 0.4, 0.9, theta3]
        h = 1e-6
        plus = list(thetas)
        plus[3] += h
        minus = list(thetas)
        minus[3] -= h
        fd = (lih_energy(plus) - lih_energy(minus)) / (2 * h)
        ps = lih_energy_grad(thetas)[3]
        assert ps == pytest.approx(fd, abs=1e-6)


# --------------------------------------------------------------------- #
# 3. Rx vs. Ry: identical populations, different entangled energies
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("theta3", [0.1, 0.3, 1.0, 2.0, 2.9])
def test_rx_and_ry_give_identical_single_qubit_populations_from_ground_state(theta3):
    """Before any entangling gate, Rx(theta)|0> and Ry(theta)|0> have the
    exact same measurement-basis populations -- they differ only in
    relative phase. This is exactly why pole_damping_factor(theta),
    derived from qg_Z(theta) = cos(theta), needs no modification at all
    to apply to an Rx-parametrized qubit."""
    qc_rx = QuantumCircuit(1)
    qc_rx.rx(theta3, 0)
    qc_ry = QuantumCircuit(1)
    qc_ry.ry(theta3, 0)
    p0_rx = abs(Statevector.from_instruction(qc_rx).data[0]) ** 2
    p0_ry = abs(Statevector.from_instruction(qc_ry).data[0]) ** 2
    assert p0_rx == pytest.approx(p0_ry, abs=1e-12)
    assert p0_rx == pytest.approx((1.0 + math.cos(theta3)) / 2.0, abs=1e-12)


@pytest.mark.parametrize(
    "thetas",
    [
        [1.0, 0.5, 0.2, 0.8],
        [2.5, 1.0, 0.3, 1.5],
        [math.pi, math.pi, 0.1, 0.4],
    ],
)
def test_rx_and_ry_give_different_whole_register_energies_once_entangled(thetas):
    """The relative phase Rx leaves behind IS physically meaningful once
    qubit 3 is entangled via CX -- the axis choice is a genuine degree of
    freedom, not a relabeling of the same physics."""
    e_rx = lih_energy(thetas, axis3="rx")
    e_ry = lih_energy(thetas, axis3="ry")
    assert e_rx != pytest.approx(e_ry, abs=1e-6)


def test_lih_ansatz_rejects_unknown_axis3():
    with pytest.raises(ValueError):
        lih_ansatz([0.1, 0.2, 0.3, 0.4], axis3="rz")


# --------------------------------------------------------------------- #
# pole_damping_factor applies unmodified to the Rx-parametrized theta3
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("theta3", [0.0, 0.3, math.pi / 2, math.pi, 2 * math.pi - 0.1])
def test_pole_damping_factor_is_a_function_of_theta_alone_regardless_of_axis(theta3):
    """pole_damping_factor takes only theta -- it has no notion of which
    gate produced it -- so it is, by construction, identical whether
    theta3 belongs to an Rx or an Ry gate. This test pins that identity
    down explicitly for the theta3 values this ansatz actually visits at
    its own optimum and at the poles."""
    assert pole_damping_factor(theta3) == pytest.approx(max(abs(math.sin(theta3)), 0.05))


# --------------------------------------------------------------------- #
# 4/5. Findings A and B on the real four-parameter molecular ansatz
# --------------------------------------------------------------------- #
def test_both_spaces_reach_the_ansatz_optimum_at_a_moderate_learning_rate():
    theta0 = [math.pi, math.pi, 1e-6, 1e-6]
    for space in ("theta", "theta_pole_damped"):
        energies = run_vqe_lih(space, theta0, lr=1.0, steps=400)
        assert energies[-1] == pytest.approx(ANSATZ_OPTIMUM, abs=1e-6)


@pytest.mark.parametrize("lr,plain_steps,damped_steps", [(0.3, 17, 355), (1.0, 5, 106)])
def test_finding_a_theta_pole_damped_needs_many_more_iterations_at_safe_lr(lr, plain_steps, damped_steps):
    """Finding A: ALL FOUR of this ansatz's own optimum parameter values
    sit at or within ~0.04 rad of a pole, so damping every one of them
    compounds into a much larger extra-iterations cost than the H2
    example's single near-pole parameter. Exact step counts are pinned
    down since this is a fully deterministic experiment."""
    theta0 = [math.pi, math.pi, 1e-6, 1e-6]
    e_plain = run_vqe_lih("theta", theta0, lr=lr, steps=400)
    e_damped = run_vqe_lih("theta_pole_damped", theta0, lr=lr, steps=400)

    steps_plain = steps_to_sustained_convergence(e_plain)
    steps_damped = steps_to_sustained_convergence(e_damped)

    assert steps_plain == plain_steps
    assert steps_damped == damped_steps
    assert steps_damped > steps_plain  # the honest cost, stated plainly


def test_theta_pole_damped_eventually_converges_despite_the_extra_iterations():
    theta0 = [math.pi, math.pi, 1e-6, 1e-6]
    energies = run_vqe_lih("theta_pole_damped", theta0, lr=1.0, steps=400)
    assert abs(energies[-1] - ANSATZ_OPTIMUM) < CHEMICAL_ACCURACY


@pytest.mark.parametrize("lr", [2.0, 2.5, 4.0, 7.0, 10.0])
def test_finding_b_plain_theta_fails_at_every_aggressive_learning_rate(lr):
    theta0 = [math.pi, math.pi, 1e-6, 1e-6]
    energies = run_vqe_lih("theta", theta0, lr=lr, steps=300)
    assert steps_to_sustained_convergence(energies) is None


@pytest.mark.parametrize(
    "lr,damped_steps",
    [(2.0, 53), (2.5, 42), (4.0, 26), (7.0, 15), (10.0, 10)],
)
def test_finding_b_theta_pole_damped_survives_every_aggressive_learning_rate(lr, damped_steps):
    """Finding B, confirmed on a real, mixed-axis, four-parameter
    molecular ansatz: theta_pole_damped -- started from the exact same
    point where plain theta fails at every one of these learning rates --
    still converges, and does so in FEWER steps as lr grows (the early,
    most-damped steps do most of the work of escaping the near-pole
    starting region)."""
    theta0 = [math.pi, math.pi, 1e-6, 1e-6]
    energies = run_vqe_lih("theta_pole_damped", theta0, lr=lr, steps=300)
    steps = steps_to_sustained_convergence(energies)
    assert steps == damped_steps
    assert energies[-1] == pytest.approx(ANSATZ_OPTIMUM, abs=1e-6)


def test_finding_b_theta_pole_damped_converges_faster_as_lr_grows_more_aggressive():
    theta0 = [math.pi, math.pi, 1e-6, 1e-6]
    lrs = [2.0, 2.5, 4.0, 7.0, 10.0]
    steps = [
        steps_to_sustained_convergence(run_vqe_lih("theta_pole_damped", theta0, lr=lr, steps=300))
        for lr in lrs
    ]
    assert all(s is not None for s in steps)
    assert steps == sorted(steps, reverse=True)


def test_run_vqe_lih_rejects_unknown_space():
    with pytest.raises(ValueError):
        run_vqe_lih("qg_raw", [math.pi, math.pi, 1e-6, 1e-6], lr=0.1, steps=5)


if __name__ == "__main__":
    print("Run via `pytest tests/test_lih_vqe_ry_rx_ansatz.py -v`.")
    theta0 = [math.pi, math.pi, 1e-6, 1e-6]
    energies = run_vqe_lih("theta_pole_damped", theta0, lr=4.0, steps=300)
    assert energies[-1] == pytest.approx(ANSATZ_OPTIMUM, abs=1e-6)
    print("Smoke check passed.")

