"""
Tests for qang.circuits: standard state circuits (Bell, GHZ, W), the
promoted circuit-to-qg-profile bridge, and their closed-form qg
predictions.

Checked:
  1. bell_circuit() matches qang.multiqubit.bell_state() exactly, for all
     four Bell states, up to Qiskit's little-endian qubit-order fix.
  2. ghz_circuit() and w_circuit() prepare the exact analytic statevectors
     they're named for, for several register sizes.
  3. qg_z_profile_of_circuit() / joint_qg_s_of_circuit() on these circuits
     match the closed-form predictions (ghz_qg_z_profile, ghz_joint_qg_s,
     w_qg_z_profile) -- the circuit-level and analytic answers agree.
  4. n=2 special cases: ghz_circuit(2) is bell_circuit('phi_plus');
     w_circuit(2) is bell_circuit('psi_plus') up to global phase.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit.quantum_info import Statevector

from qang.multiqubit import bell_state, joint_qg_s, per_qubit_qg_z
from qang.circuits import (
    bell_circuit,
    ghz_circuit,
    ghz_joint_qg_s,
    ghz_qg_z_profile,
    joint_qg_s_of_circuit,
    qg_z_profile_of_circuit,
    w_circuit,
    w_qg_z_profile,
)


@pytest.mark.parametrize("kind", ["phi_plus", "phi_minus", "psi_plus", "psi_minus"])
def test_bell_circuit_matches_closed_form_bell_state(kind):
    qc = bell_circuit(kind)
    sv = Statevector.from_instruction(qc).data
    expected = bell_state(kind)
    # global phase is not physically meaningful; compare up to it
    phase = None
    for a, b in zip(sv, expected):
        if abs(b) > 1e-9:
            phase = a / b
            break
    assert phase is not None
    assert np.allclose(sv, expected * phase, atol=1e-9)


def _ghz_statevector(n_qubits: int) -> np.ndarray:
    dim = 2 ** n_qubits
    sv = np.zeros(dim, dtype=complex)
    s = 1.0 / math.sqrt(2.0)
    sv[0] = s
    sv[dim - 1] = s
    return sv


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
def test_ghz_circuit_matches_closed_form(n_qubits):
    qc = ghz_circuit(n_qubits)
    sv = Statevector.from_instruction(qc).data
    if n_qubits == 1:
        # H|0> = (|0> + |1>)/sqrt(2), the n=1 "GHZ" is just a Hadamard state
        expected = np.array([1.0, 1.0]) / math.sqrt(2.0)
    else:
        expected = _ghz_statevector(n_qubits)
    assert np.allclose(sv, expected, atol=1e-9)


def _w_statevector(n_qubits: int) -> np.ndarray:
    """Closed-form W state in Qiskit's own (little-endian) qubit order:
    equal superposition of every basis state with exactly one '1' bit."""
    dim = 2 ** n_qubits
    sv = np.zeros(dim, dtype=complex)
    amp = 1.0 / math.sqrt(n_qubits)
    for i in range(n_qubits):
        sv[1 << i] = amp
    return sv


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6])
def test_w_circuit_matches_closed_form(n_qubits):
    qc = w_circuit(n_qubits)
    sv = Statevector.from_instruction(qc).data
    expected = _w_statevector(n_qubits)
    assert np.allclose(sv, expected, atol=1e-9)


def test_w_circuit_single_qubit_is_x():
    qc = w_circuit(1)
    sv = Statevector.from_instruction(qc).data
    assert np.allclose(sv, [0.0, 1.0], atol=1e-9)


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
def test_ghz_qg_z_profile_matches_circuit(n_qubits):
    qc = ghz_circuit(n_qubits)
    profile = qg_z_profile_of_circuit(qc, n_qubits)
    expected = ghz_qg_z_profile(n_qubits)
    assert np.allclose(profile, expected, atol=1e-9)


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
def test_ghz_joint_qg_s_matches_circuit(n_qubits):
    qc = ghz_circuit(n_qubits)
    value = joint_qg_s_of_circuit(qc, n_qubits, normalize=True)
    assert value == pytest.approx(ghz_joint_qg_s(n_qubits, normalize=True), abs=1e-9)


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6])
def test_w_qg_z_profile_matches_circuit(n_qubits):
    qc = w_circuit(n_qubits)
    profile = qg_z_profile_of_circuit(qc, n_qubits)
    expected = w_qg_z_profile(n_qubits)
    assert np.allclose(profile, expected, atol=1e-9)


def test_ghz_2_qubit_is_bell_phi_plus():
    qc_ghz = ghz_circuit(2)
    qc_bell = bell_circuit("phi_plus")
    sv_ghz = Statevector.from_instruction(qc_ghz).data
    sv_bell = Statevector.from_instruction(qc_bell).data
    assert np.allclose(sv_ghz, sv_bell, atol=1e-9)


def test_w_2_qubit_matches_psi_plus_up_to_phase():
    qc_w = w_circuit(2)
    qc_bell = bell_circuit("psi_plus")
    sv_w = Statevector.from_instruction(qc_w).data
    sv_bell = Statevector.from_instruction(qc_bell).data
    phase = None
    for a, b in zip(sv_w, sv_bell):
        if abs(b) > 1e-9:
            phase = a / b
            break
    assert phase is not None
    assert np.allclose(sv_w, sv_bell * phase, atol=1e-9)


def test_qg_z_profile_of_circuit_matches_manual_computation():
    """Sanity-checks the promoted helper against a manual, from-scratch
    per_qubit_qg_z call, to guard against a silent regression in the
    reverse_qargs() fix it wraps."""
    qc = bell_circuit("phi_plus")
    manual_sv = Statevector.from_instruction(qc).reverse_qargs().data
    manual = per_qubit_qg_z(manual_sv, n_qubits=2)
    assert np.allclose(qg_z_profile_of_circuit(qc, 2), manual, atol=1e-12)


def test_joint_qg_s_of_circuit_matches_manual_computation():
    qc = ghz_circuit(3)
    manual_sv = Statevector.from_instruction(qc).data
    manual = joint_qg_s(manual_sv, n_qubits=3, normalize=True)
    assert joint_qg_s_of_circuit(qc, 3, normalize=True) == pytest.approx(manual, abs=1e-12)


if __name__ == "__main__":
    print("Run via `pytest tests/test_circuits.py -v` for full parametrized coverage.")
    qc = ghz_circuit(3)
    profile = qg_z_profile_of_circuit(qc, 3)
    assert np.allclose(profile, [0.0, 0.0, 0.0], atol=1e-9)
    print("Smoke check passed.")
