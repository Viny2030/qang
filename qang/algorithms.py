"""
qang.algorithms -- textbook multi-qubit quantum algorithms built with
Qiskit, using the same "everything unitary, verify by exact statevector"
discipline already used throughout qang (qang.qec's deferred-measurement
error correction; qang.circuits' state constructors).

Three algorithms live here:

  1. Quantum teleportation, using the deferred-measurement principle:
     the textbook version measures the Bell pair and applies classically
     -controlled X/Z corrections, but any measure-then-classically-
     -control circuit has an exactly equivalent fully unitary circuit
     where the "would-be" classical controls are replaced by ordinary
     quantum-controlled gates (Nielsen & Chuang, Section 4.4). That lets
     correctness be checked by exact statevector fidelity instead of by
     sampling -- the same technique qang.qec already relies on.

  2. Superdense coding -- the "reverse" protocol, sending two classical
     bits over one qubit given a pre-shared Bell pair. Empirically
     important note (verified numerically, not assumed): with the gate
     ordering used here, the decoded bit on qubit 0 recovers the
     original *bit_z* (the Z-correction bit) and the decoded bit on
     qubit 1 recovers the original *bit_x* (the X-correction bit) -- the
     roles are swapped relative to a same-index guess, which is why the
     variable names below say ``bit_z``/``bit_x`` rather than ``b0``/``b1``.

  3. Grover's algorithm, with an oracle built from an X-sandwich plus a
     phase-kickback multi-controlled-Z (via an H-sandwiched MCXGate) and
     the standard diffusion operator, run for the optimal number of
     iterations (round(pi/4 * sqrt(2**n_qubits))).

This module is optional: importing ``qang`` itself never requires Qiskit.
Only importing *this* module does, and it raises a clear, actionable error
if Qiskit is not installed (same pattern as qang.qiskit_gate / qang.circuits).
"""

from __future__ import annotations

import math
from typing import Tuple

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import MCXGate
    from qiskit.quantum_info import Statevector, partial_trace, state_fidelity

    _QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when qiskit is absent
    _QISKIT_AVAILABLE = False


def _require_qiskit():
    if not _QISKIT_AVAILABLE:
        raise ImportError(
            "qang.algorithms requires Qiskit. Install it with "
            "`pip install qiskit` (and, to run circuits, `pip install qiskit-aer`)."
        )


# --------------------------------------------------------------------- #
# 1. Quantum teleportation (deferred-measurement / fully unitary form)
# --------------------------------------------------------------------- #
def teleportation_circuit(state_prep_angles: Tuple[float, float] = (0.0, 0.0)) -> "QuantumCircuit":
    """
    3-qubit teleportation circuit using the deferred-measurement
    principle (no mid-circuit measurement or classical control at all):

      qubit 0: Alice's message qubit, prepared via RY(theta) then RZ(phi)
               from ``state_prep_angles`` = (theta, phi).
      qubit 1: Alice's half of a shared Bell pair.
      qubit 2: Bob's half of the shared Bell pair -- ends up holding an
               exact copy of qubit 0's original state.

    Bell-pair prep (H + CX on 1,2), "Bell measurement" on qubits 0,1
    replaced by CX(0,1) + H(0) (unitary, undoes the measurement basis
    change instead of measuring in it), and the classically-controlled
    corrections replaced by quantum-controlled gates: CX(1, 2) and
    CZ(0, 2). Because every step is unitary, the output state's reduced
    density matrix on qubit 2 is *exactly* equal to the input state on
    qubit 0 (verified via partial_trace + state_fidelity in
    tests/test_algorithms.py, not by sampling).
    """
    _require_qiskit()
    theta, phi = state_prep_angles
    qc = QuantumCircuit(3, name="teleportation")
    qc.ry(theta, 0)
    qc.rz(phi, 0)

    qc.h(1)
    qc.cx(1, 2)

    qc.cx(0, 1)
    qc.h(0)

    qc.cx(1, 2)
    qc.cz(0, 2)
    return qc


def teleportation_output_fidelity(state_prep_angles: Tuple[float, float] = (0.0, 0.0)) -> float:
    """
    Build the teleportation circuit, and return the exact state fidelity
    between qubit 2's final reduced state and qubit 0's originally
    prepared (RY, RZ) single-qubit state. Should be 1.0 (up to floating
    -point error) for any angles.
    """
    _require_qiskit()
    theta, phi = state_prep_angles
    original = QuantumCircuit(1)
    original.ry(theta, 0)
    original.rz(phi, 0)
    original_state = Statevector.from_instruction(original)

    qc = teleportation_circuit(state_prep_angles)
    full_state = Statevector.from_instruction(qc)
    # qubit 2 is Qiskit index 2; trace out qubits 0 and 1.
    bob_state = partial_trace(full_state, [0, 1])
    return float(state_fidelity(original_state, bob_state))


# --------------------------------------------------------------------- #
# 2. Superdense coding
# --------------------------------------------------------------------- #
def superdense_coding_circuit(bit_z: int, bit_x: int) -> "QuantumCircuit":
    """
    2-qubit superdense coding circuit encoding two classical bits
    (``bit_z``, ``bit_x``) into one shared Bell pair using only local
    operations on qubit 0, then decoding via CX(0,1) + H(0).

    Gate sequence: H(0); CX(0,1) [shared Bell pair]; Z(0) if bit_z;
    X(0) if bit_x; CX(0,1); H(0).

    IMPORTANT (empirically verified, not a naming guess): after
    decoding, qubit 0's computational-basis value equals the original
    ``bit_z``, and qubit 1's value equals the original ``bit_x`` -- the
    roles are swapped relative to same-index qubit/bit correspondence.
    See ``decode_superdense_coding`` and tests/test_algorithms.py, which
    check exactly this mapping for all four two-bit messages.
    """
    _require_qiskit()
    if bit_z not in (0, 1) or bit_x not in (0, 1):
        raise ValueError(f"bit_z and bit_x must each be 0 or 1, got bit_z={bit_z}, bit_x={bit_x}.")

    qc = QuantumCircuit(2, name=f"superdense_{bit_z}{bit_x}")
    qc.h(0)
    qc.cx(0, 1)

    if bit_z == 1:
        qc.z(0)
    if bit_x == 1:
        qc.x(0)

    qc.cx(0, 1)
    qc.h(0)
    return qc


def decode_superdense_coding(qc: "QuantumCircuit") -> Tuple[int, int]:
    """
    Deterministically decode a superdense-coding circuit's output
    (exact statevector, no sampling): returns (bit_z, bit_x) as read off
    qubit 0 and qubit 1's individual (deterministic, 0/1-probability)
    marginals respectively.
    """
    _require_qiskit()
    sv = Statevector.from_instruction(qc)
    probs_q0 = sv.probabilities(qargs=[0])
    probs_q1 = sv.probabilities(qargs=[1])
    bit_z = int(round(probs_q0[1]))
    bit_x = int(round(probs_q1[1]))
    return bit_z, bit_x


# --------------------------------------------------------------------- #
# 3. Grover's algorithm
# --------------------------------------------------------------------- #
def grover_oracle(n_qubits: int, marked_state: int) -> "QuantumCircuit":
    """
    Phase oracle that flips the sign of exactly |marked_state> (an
    integer in [0, 2**n_qubits)), via the standard X-sandwich + phase
    -kickback multi-controlled-Z trick: flip every qubit whose bit in
    ``marked_state`` is 0, apply a multi-controlled Z (built from an
    H-sandwiched MCXGate on an ancilla-free all-controls-are-qubits
    version using phase kickback on the last qubit), then undo the
    X-sandwich.
    """
    _require_qiskit()
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}.")
    if not (0 <= marked_state < 2 ** n_qubits):
        raise ValueError(f"marked_state must be in [0, 2**n_qubits), got {marked_state}.")

    qc = QuantumCircuit(n_qubits, name=f"oracle_{marked_state}")
    bits = [(marked_state >> i) & 1 for i in range(n_qubits)]  # bits[i] = qubit i's target bit

    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)

    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)

    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    return qc


def grover_diffusion(n_qubits: int) -> "QuantumCircuit":
    """
    Standard Grover diffusion operator (inversion about the mean):
    H^{\\otimes n}; X^{\\otimes n}; multi-controlled-Z (phase kickback,
    same construction as grover_oracle); X^{\\otimes n}; H^{\\otimes n}.
    """
    _require_qiskit()
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}.")

    qc = QuantumCircuit(n_qubits, name="diffusion")
    for q in range(n_qubits):
        qc.h(q)
    for q in range(n_qubits):
        qc.x(q)

    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)

    for q in range(n_qubits):
        qc.x(q)
    for q in range(n_qubits):
        qc.h(q)
    return qc


def grover_optimal_iterations(n_qubits: int) -> int:
    """round(pi/4 * sqrt(2**n_qubits)), the standard optimal number of
    Grover iterations for a single marked item among 2**n_qubits."""
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}.")
    return round((math.pi / 4.0) * math.sqrt(2 ** n_qubits))


def grover_circuit(n_qubits: int, marked_state: int, n_iterations: int = None) -> "QuantumCircuit":
    """
    Full Grover search circuit: uniform superposition (H^{\\otimes n}),
    then ``n_iterations`` repetitions of (oracle, diffusion).
    ``n_iterations`` defaults to ``grover_optimal_iterations(n_qubits)``.
    """
    _require_qiskit()
    if n_iterations is None:
        n_iterations = grover_optimal_iterations(n_qubits)
    oracle = grover_oracle(n_qubits, marked_state)
    diffusion = grover_diffusion(n_qubits)

    qc = QuantumCircuit(n_qubits, name=f"grover_{marked_state}")
    for q in range(n_qubits):
        qc.h(q)
    for _ in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)
    return qc


def grover_success_probability(n_qubits: int, marked_state: int, n_iterations: int = None) -> float:
    """Exact probability (statevector, no sampling) that measuring the
    Grover circuit's output yields ``marked_state``."""
    _require_qiskit()
    qc = grover_circuit(n_qubits, marked_state, n_iterations)
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()
    return float(probs[marked_state])


if __name__ == "__main__":
    print("Teleportation fidelity for a few (theta, phi) message states:")
    for theta, phi in [(0.3, 0.0), (1.0, 0.7), (2.5, 1.9)]:
        fid = teleportation_output_fidelity((theta, phi))
        print(f"  theta={theta:.2f} phi={phi:.2f}  fidelity={fid:.12f}")
    print()

    print("Superdense coding, all four two-bit messages:")
    for bit_z in (0, 1):
        for bit_x in (0, 1):
            qc = superdense_coding_circuit(bit_z, bit_x)
            decoded_z, decoded_x = decode_superdense_coding(qc)
            print(
                f"  sent (bit_z={bit_z}, bit_x={bit_x})  "
                f"-> decoded (bit_z={decoded_z}, bit_x={decoded_x})  "
                f"{'OK' if (decoded_z, decoded_x) == (bit_z, bit_x) else 'MISMATCH'}"
            )
    print()

    print("Grover's algorithm, boosted probability of the marked state:")
    for n_qubits in [3, 4, 5]:
        marked = (2 ** n_qubits) // 3
        n_iter = grover_optimal_iterations(n_qubits)
        prob = grover_success_probability(n_qubits, marked, n_iter)
        uniform = 1.0 / (2 ** n_qubits)
        print(
            f"  n_qubits={n_qubits}  marked={marked}  iterations={n_iter}  "
            f"P(marked)={prob:.6f}  (uniform baseline={uniform:.6f})"
        )
