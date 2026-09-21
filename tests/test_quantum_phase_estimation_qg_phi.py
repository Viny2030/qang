"""
Tests for examples/quantum_phase_estimation_qg_phi.py: closing the loop
between qang.phase.QangPhi (a closed-form unit) and Quantum Phase
Estimation (an actual multi-qubit algorithm).

Checked:
  1. QPE run on each non-trivial qang.phase.ANCHOR_POINTS gate (T, S, Z,
     S-dagger) recovers its exact anchor phi value with probability 1,
     for a counting register with enough qubits to represent it exactly
     in binary (1/8, 1/4, 1/2, 3/4 all terminate at <= 3 bits).
  2. The QangPhi built from that recovered phi_estimate reproduces the
     gate's own eigenvalue e^{i*2*pi*phi} to floating-point precision --
     QPE (an algorithm) and QangPhi.value (a closed-form formula) agree
     exactly.
  3. For a generic, non-exactly-representable phase, the QPE estimation
     error is bounded by 1/2^n_counting (the standard textbook bound),
     and shrinks as n_counting grows.
  4. qpe_circuit rejects an invalid n_counting.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import cmath

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qang.phase import ANCHOR_POINTS, QangPhi
from quantum_phase_estimation_qg_phi import (
    qpe_circuit,
    qpe_estimate,
    qpe_recovers_qang_phi,
)


@pytest.mark.parametrize("name", ["t_gate", "s_gate", "z_gate", "s_dagger"])
def test_qpe_recovers_anchor_phase_exactly(name):
    phi_true = ANCHOR_POINTS[name]
    n_counting = 4  # enough bits for 1/8, 1/4, 1/2, 3/4 all exactly
    phi_estimate, probability = qpe_estimate(n_counting, phi_true)
    assert phi_estimate == pytest.approx(phi_true, abs=1e-12)
    assert probability == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("name", ["t_gate", "s_gate", "z_gate", "s_dagger"])
def test_qpe_recovered_qang_phi_matches_true_eigenvalue_exactly(name):
    phi_true = ANCHOR_POINTS[name]
    qg_phi, true_eigenvalue = qpe_recovers_qang_phi(4, phi_true)
    assert isinstance(qg_phi, QangPhi)
    assert abs(qg_phi.value - true_eigenvalue) < 1e-9
    # and matches the closed-form QangPhi built directly from the anchor
    assert abs(qg_phi.value - QangPhi(phi_true).value) < 1e-9


def test_identity_anchor_needs_no_counting_qubits_argument_but_is_trivial():
    # phi = 0 (identity): QPE should also recover it exactly
    phi_estimate, probability = qpe_estimate(3, ANCHOR_POINTS["identity"])
    assert phi_estimate == pytest.approx(0.0, abs=1e-12)
    assert probability == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("n_counting", [4, 6, 8, 10])
def test_generic_phase_estimation_error_bounded_by_inverse_power_of_two(n_counting):
    phase = 0.3
    phi_estimate, _probability = qpe_estimate(n_counting, phase)
    bound = 1.0 / (2 ** n_counting)
    assert abs(phi_estimate - phase) <= bound + 1e-12


def test_generic_phase_estimation_error_shrinks_with_more_counting_qubits():
    phase = 0.3
    errors = []
    for n_counting in [4, 6, 8, 10]:
        phi_estimate, _ = qpe_estimate(n_counting, phase)
        errors.append(abs(phi_estimate - phase))
    assert all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))
    assert errors[-1] < errors[0]


def test_qpe_circuit_rejects_invalid_n_counting():
    with pytest.raises(ValueError):
        qpe_circuit(0, 0.25)


def test_qpe_circuit_has_expected_qubit_count():
    qc = qpe_circuit(5, 0.25)
    assert qc.num_qubits == 6  # 5 counting + 1 target


if __name__ == "__main__":
    print("Run via `pytest tests/test_quantum_phase_estimation_qg_phi.py -v` for full coverage.")
    phi_estimate, probability = qpe_estimate(4, ANCHOR_POINTS["t_gate"])
    assert phi_estimate == pytest.approx(1.0 / 8.0, abs=1e-12)
    print("Smoke check passed.")
