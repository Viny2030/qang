"""
Tests for stabilizer-code syndrome extraction, reformulated in terms of
qg_Z (Future Research Direction, Section 6.2 of the paper; GitHub issue:
"Explore qg_Z reformulation of stabilizer-code syndrome extraction (QEC)").

Uses the 3-qubit bit-flip code (Gottesman, 1997) as the minimal worked
example. Qubits 0, 1, 2 hold the encoded logical qubit
alpha|000> + beta|111>; qubits 3, 4 are syndrome ancillas measuring the
stabilizer generators Z0 Z1 and Z1 Z2 via CNOTs.

The point of this file: syndrome measurements are +-1-valued parity
checks, formally identical in structure to the qg_Z-valued measurements
already treated in Section 4.4 of the paper for T1/randomized
benchmarking. Concretely, qang.multiqubit.per_qubit_qg_z of each ancilla
qubit gives the syndrome bit directly (qg_Z = +1 -> syndrome bit 0,
qg_Z = -1 -> syndrome bit 1), and -- the actual content of a stabilizer
code, not just a relabeling -- this qg_Z value is exactly the same
regardless of the logical amplitudes (alpha, beta): the syndrome carries
information about the error only, never about the encoded data. That
independence is checked explicitly below by sweeping theta.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity

from qang.multiqubit import per_qubit_qg_z

# qubits: 0, 1, 2 = data (logical); 3, 4 = syndrome ancillas
DATA_QUBITS = (0, 1, 2)
ANCILLA_QUBITS = (3, 4)
N_QUBITS = 5

# standard bit-flip code syndrome lookup table: (s1, s2) -> which data
# qubit had an X error, or None for no error.
SYNDROME_TABLE = {
    (0, 0): None,
    (1, 0): 0,
    (1, 1): 1,
    (0, 1): 2,
}


def _qg_z_profile(qc: QuantumCircuit, n_qubits: int = N_QUBITS):
    """qg_Z of every qubit, indexed to match Qiskit's own qubit numbering
    (see tests/test_deutsch_jozsa_bv.py for why reverse_qargs() is needed)."""
    sv = Statevector.from_instruction(qc).reverse_qargs().data
    return per_qubit_qg_z(sv, n_qubits=n_qubits)


def _bit_flip_syndrome_circuit(theta: float, error_qubit) -> QuantumCircuit:
    """Encode alpha|000> + beta|111> (alpha=cos(theta/2), beta=sin(theta/2)),
    optionally apply a single X error, then extract the syndrome."""
    qc = QuantumCircuit(N_QUBITS)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    if error_qubit is not None:
        qc.x(error_qubit)
    qc.cx(0, 3)
    qc.cx(1, 3)  # ancilla 3 = parity(q0, q1) = Z0 Z1 stabilizer
    qc.cx(1, 4)
    qc.cx(2, 4)  # ancilla 4 = parity(q1, q2) = Z1 Z2 stabilizer
    return qc


def _syndrome_bits(qc: QuantumCircuit):
    profile = _qg_z_profile(qc)
    a1, a2 = profile[3], profile[4]
    assert abs(abs(a1) - 1.0) < 1e-9  # ancillas are always exact poles
    assert abs(abs(a2) - 1.0) < 1e-9
    return (0 if a1 > 0 else 1), (0 if a2 > 0 else 1)


@pytest.mark.parametrize("error_qubit", [None, 0, 1, 2])
@pytest.mark.parametrize("theta", [0.3, 1.0, 2.0, math.pi / 2, 2.9])
def test_syndrome_qg_z_identifies_error_independent_of_logical_amplitude(theta, error_qubit):
    """The syndrome (read out via qg_Z of the ancillas) must correctly
    identify the error location and must NOT depend on theta -- a
    stabilizer measurement carries no information about the encoded
    logical amplitudes, only about the error."""
    qc = _bit_flip_syndrome_circuit(theta, error_qubit)
    s1, s2 = _syndrome_bits(qc)
    assert SYNDROME_TABLE[(s1, s2)] == error_qubit


@pytest.mark.parametrize("error_qubit", [None, 0, 1, 2])
@pytest.mark.parametrize("theta", [0.3, 1.2, 2.5])
def test_detect_and_correct_cycle_recovers_logical_qubit_exactly(theta, error_qubit):
    """Full cycle: extract syndrome via qg_Z, decode which qubit (if any)
    to flip back, apply the correction, then undo the encoding CNOTs.
    The recovered logical qubit must match the originally prepared state
    with fidelity 1, regardless of which of the 4 error cases occurred."""
    qc = _bit_flip_syndrome_circuit(theta, error_qubit)
    s1, s2 = _syndrome_bits(qc)
    identified = SYNDROME_TABLE[(s1, s2)]
    assert identified == error_qubit

    if identified is not None:
        qc.x(identified)  # correction
    qc.cx(0, 2)
    qc.cx(0, 1)  # undo the encoding

    reference = QuantumCircuit(1)
    reference.ry(theta, 0)
    reference_sv = Statevector.from_instruction(reference)

    full_sv = Statevector.from_instruction(qc)
    recovered_rho = partial_trace(full_sv, [1, 2, 3, 4])  # keep qubit 0 only
    fidelity = state_fidelity(recovered_rho, reference_sv)
    assert fidelity == pytest.approx(1.0, abs=1e-9)


if __name__ == "__main__":
    print("Run via `pytest tests/test_qec_syndrome_extraction.py -v` for full parametrized coverage.")
    qc = _bit_flip_syndrome_circuit(1.0, 1)
    s1, s2 = _syndrome_bits(qc)
    assert SYNDROME_TABLE[(s1, s2)] == 1
    print("Smoke check passed.")
