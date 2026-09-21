"""
Tests for qang.qec: the 3-qubit bit-flip code's encode / extract_syndrome
/ syndrome_from_qg_z / decode_syndrome / apply_correction / decode_bit_flip
functions, promoted from tests/test_qec_syndrome_extraction.py into
reusable library code.

Checked:
  1. The full encode -> (optional error) -> extract syndrome -> read via
     qg_Z -> decode -> correct -> decode_bit_flip cycle recovers the
     original logical state with fidelity 1.0, for all 4 error cases
     (none, and each of the 3 data qubits) and several logical
     amplitudes theta -- the same claim tests/test_qec_syndrome_extraction.py
     already validated, now exercised through the public library
     functions instead of inlined test-only code.
  2. The syndrome identified via qg_Z never depends on the encoded
     logical amplitude theta (a stabilizer measurement carries
     information about the error only).
  3. syndrome_from_qg_z raises if handed a non-pole ancilla qg_Z value
     (i.e. it is only meaningful right after extract_syndrome() on
     ancillas that started in |0>).
  4. decode_syndrome / SYNDROME_TABLE reject an invalid (s1, s2) pair.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity

from qang.circuits import qg_z_profile_of_circuit
from qang.qec import (
    SYNDROME_TABLE,
    apply_correction,
    decode_bit_flip,
    decode_syndrome,
    encode_bit_flip,
    extract_syndrome,
    syndrome_from_qg_z,
)

DATA_QUBITS = (0, 1, 2)
ANCILLA_QUBITS = (3, 4)
N_QUBITS = 5


def _prepared_circuit(theta: float, error_qubit) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)
    qc.ry(theta, 0)
    encode_bit_flip(qc, DATA_QUBITS)
    if error_qubit is not None:
        qc.x(error_qubit)
    extract_syndrome(qc, DATA_QUBITS, ANCILLA_QUBITS)
    return qc


@pytest.mark.parametrize("error_qubit", [None, 0, 1, 2])
@pytest.mark.parametrize("theta", [0.3, 1.0, 2.0, math.pi / 2, 2.9])
def test_syndrome_identifies_error_independent_of_logical_amplitude(theta, error_qubit):
    qc = _prepared_circuit(theta, error_qubit)
    profile = qg_z_profile_of_circuit(qc, N_QUBITS)
    s1, s2 = syndrome_from_qg_z(profile, ANCILLA_QUBITS)
    assert decode_syndrome(s1, s2) == error_qubit


@pytest.mark.parametrize("error_qubit", [None, 0, 1, 2])
@pytest.mark.parametrize("theta", [0.3, 1.2, 2.5])
def test_full_detect_correct_decode_cycle_recovers_logical_qubit_exactly(theta, error_qubit):
    qc = _prepared_circuit(theta, error_qubit)
    profile = qg_z_profile_of_circuit(qc, N_QUBITS)
    s1, s2 = syndrome_from_qg_z(profile, ANCILLA_QUBITS)
    identified = decode_syndrome(s1, s2)
    assert identified == error_qubit

    apply_correction(qc, identified)
    decode_bit_flip(qc, DATA_QUBITS)

    reference = QuantumCircuit(1)
    reference.ry(theta, 0)
    reference_sv = Statevector.from_instruction(reference)

    full_sv = Statevector.from_instruction(qc)
    recovered_rho = partial_trace(full_sv, [1, 2, 3, 4])  # keep qubit 0 only
    fidelity = state_fidelity(recovered_rho, reference_sv)
    assert fidelity == pytest.approx(1.0, abs=1e-9)


def test_syndrome_from_qg_z_rejects_non_pole_values():
    with pytest.raises(ValueError):
        syndrome_from_qg_z([0.0, 0.0, 0.0, 0.3, -1.0], ANCILLA_QUBITS)


def test_decode_syndrome_rejects_invalid_bits():
    with pytest.raises(ValueError):
        decode_syndrome(2, 0)


def test_syndrome_table_covers_all_four_cases():
    assert set(SYNDROME_TABLE.keys()) == {(0, 0), (1, 0), (1, 1), (0, 1)}
    assert SYNDROME_TABLE[(0, 0)] is None
    assert {SYNDROME_TABLE[k] for k in [(1, 0), (1, 1), (0, 1)]} == {0, 1, 2}


def test_encode_and_extract_syndrome_reject_wrong_length_qubit_lists():
    qc = QuantumCircuit(N_QUBITS)
    with pytest.raises(ValueError):
        encode_bit_flip(qc, (0, 1))
    with pytest.raises(ValueError):
        extract_syndrome(qc, (0, 1, 2), (3,))


if __name__ == "__main__":
    print("Run via `pytest tests/test_qec.py -v` for full parametrized coverage.")
    qc = _prepared_circuit(1.0, 1)
    profile = qg_z_profile_of_circuit(qc, N_QUBITS)
    s1, s2 = syndrome_from_qg_z(profile, ANCILLA_QUBITS)
    assert decode_syndrome(s1, s2) == 1
    print("Smoke check passed.")
