"""
qang.qec -- the 3-qubit bit-flip code's encode / syndrome-extraction /
decode-and-correct cycle, promoted from tests/test_qec_syndrome_extraction.py
into reusable library functions (Future Research Direction, Section 6.2 of
the paper: "Explore qg_Z reformulation of stabilizer-code syndrome
extraction (QEC)").

The 3-qubit bit-flip code (Gottesman, 1997) encodes one logical qubit
alpha|0> + beta|1> as alpha|000> + beta|111> across three data qubits, and
uses two ancilla qubits to measure the stabilizer generators Z0 Z1 and
Z1 Z2 via CNOTs. The key point this module makes reusable: those
+-1-valued parity checks are read out via qg_Z of the ancilla qubits
(qang.multiqubit.per_qubit_qg_z), exactly the same measurement quantity
already used for T1 / randomized-benchmarking in Section 4.4 of the
paper -- syndrome extraction is not a new kind of measurement, just a
new place to apply the one already defined. This was validated
numerically (independent of the encoded logical amplitude, across the
full detect-correct-decode cycle with fidelity 1.0) in
tests/test_qec_syndrome_extraction.py; this module turns that validated
logic into composable functions instead of leaving it inlined in tests.

This module is optional: importing ``qang`` itself never requires Qiskit.
Only importing *this* module does, and it raises a clear, actionable error
if Qiskit is not installed (same pattern as qang.qiskit_gate).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

try:
    from qiskit import QuantumCircuit

    _QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when qiskit is absent
    _QISKIT_AVAILABLE = False


def _require_qiskit():
    if not _QISKIT_AVAILABLE:
        raise ImportError(
            "qang.qec requires Qiskit. Install it with "
            "`pip install qiskit` (and, to run circuits, `pip install qiskit-aer`)."
        )


# Standard 3-qubit bit-flip code syndrome lookup table: (s1, s2) -> which
# data qubit had an X error, or None for no error. s1 = parity(data[0],
# data[1]) = Z0 Z1 stabilizer; s2 = parity(data[1], data[2]) = Z1 Z2
# stabilizer.
SYNDROME_TABLE = {
    (0, 0): None,
    (1, 0): 0,
    (1, 1): 1,
    (0, 1): 2,
}


def encode_bit_flip(
    qc: "QuantumCircuit", data_qubits: Sequence[int] = (0, 1, 2)
) -> "QuantumCircuit":
    """
    Encode whatever single-qubit state already sits on ``data_qubits[0]``
    (e.g. alpha|0> + beta|1> from a prior RY) into the 3-qubit bit-flip
    code alpha|000> + beta|111>, via two CNOTs from the first data qubit
    to the other two. Mutates and returns ``qc``.
    """
    _require_qiskit()
    if len(data_qubits) != 3:
        raise ValueError(f"data_qubits must have exactly 3 entries, got {len(data_qubits)}.")
    d0, d1, d2 = data_qubits
    qc.cx(d0, d1)
    qc.cx(d0, d2)
    return qc


def extract_syndrome(
    qc: "QuantumCircuit", data_qubits: Sequence[int] = (0, 1, 2), ancilla_qubits: Sequence[int] = (3, 4)
) -> "QuantumCircuit":
    """
    Append the standard syndrome-extraction network: ancilla_qubits[0]
    accumulates the parity of data_qubits[0], data_qubits[1] (the Z0 Z1
    stabilizer); ancilla_qubits[1] accumulates the parity of
    data_qubits[1], data_qubits[2] (the Z1 Z2 stabilizer). Mutates and
    returns ``qc``. Ancillas must start in |0>.
    """
    _require_qiskit()
    if len(data_qubits) != 3:
        raise ValueError(f"data_qubits must have exactly 3 entries, got {len(data_qubits)}.")
    if len(ancilla_qubits) != 2:
        raise ValueError(f"ancilla_qubits must have exactly 2 entries, got {len(ancilla_qubits)}.")
    d0, d1, d2 = data_qubits
    a0, a1 = ancilla_qubits
    qc.cx(d0, a0)
    qc.cx(d1, a0)  # a0 = parity(d0, d1) = Z0 Z1 stabilizer
    qc.cx(d1, a1)
    qc.cx(d2, a1)  # a1 = parity(d1, d2) = Z1 Z2 stabilizer
    return qc


def syndrome_from_qg_z(qg_z_profile: Sequence[float], ancilla_qubits: Sequence[int] = (3, 4)) -> Tuple[int, int]:
    """
    Convert the ancilla qubits' qg_Z readings (from
    qang.circuits.qg_z_profile_of_circuit, or any other source of a
    qg_Z-per-qubit profile) into the two classical syndrome bits: an
    ancilla always sits at an exact qg_Z pole (+1 or -1) after a parity
    (CX-chain) measurement, with qg_Z = +1 decoding to syndrome bit 0 and
    qg_Z = -1 decoding to syndrome bit 1.
    """
    a0, a1 = ancilla_qubits
    v0, v1 = qg_z_profile[a0], qg_z_profile[a1]
    if abs(abs(v0) - 1.0) > 1e-6 or abs(abs(v1) - 1.0) > 1e-6:
        raise ValueError(
            f"ancilla qg_Z values must sit at an exact pole (+-1), got {v0!r}, {v1!r}. "
            "Did extract_syndrome() run on ancillas starting in |0>?"
        )
    return (0 if v0 > 0 else 1), (0 if v1 > 0 else 1)


def decode_syndrome(s1: int, s2: int) -> Optional[int]:
    """Which data qubit (0, 1, or 2) the syndrome (s1, s2) identifies as
    having an X error, or None if the syndrome indicates no error."""
    key = (s1, s2)
    if key not in SYNDROME_TABLE:
        raise ValueError(f"syndrome bits must each be 0 or 1, got {key}.")
    return SYNDROME_TABLE[key]


def apply_correction(qc: "QuantumCircuit", identified_qubit: Optional[int]) -> "QuantumCircuit":
    """Apply the X correction identified by decode_syndrome(), or do
    nothing if identified_qubit is None. Mutates and returns ``qc``."""
    _require_qiskit()
    if identified_qubit is not None:
        qc.x(identified_qubit)
    return qc


def decode_bit_flip(
    qc: "QuantumCircuit", data_qubits: Sequence[int] = (0, 1, 2)
) -> "QuantumCircuit":
    """Undo encode_bit_flip()'s two CNOTs, returning the logical qubit's
    state to data_qubits[0] alone (assuming any error has already been
    corrected via apply_correction()). Mutates and returns ``qc``."""
    _require_qiskit()
    if len(data_qubits) != 3:
        raise ValueError(f"data_qubits must have exactly 3 entries, got {len(data_qubits)}.")
    d0, d1, d2 = data_qubits
    qc.cx(d0, d2)
    qc.cx(d0, d1)
    return qc
