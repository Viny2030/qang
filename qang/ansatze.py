"""
qang.ansatze -- a small reusable library of variational ansatz
constructors, including a genuinely qg-native ansatz whose free
parameters are qg_Z values directly rather than raw angles.

Two families:

  1. Standard building blocks, generalized out of the one-off ansatz
     written for examples/vqe_h2_qg_vs_theta.py so future VQE work in
     qang doesn't have to redefine them:

       - ``single_excitation_ansatz``   the minimal 1-parameter ansatz
         used for the H2 example (X on the occupied qubit, RY(theta) on
         the virtual qubit, then a CX back to the occupied qubit),
         generalized to an arbitrary qubit pair on an arbitrary register
         size. ``single_excitation_ansatz(2, 0, 1, theta)`` reproduces
         examples/vqe_h2_qg_vs_theta.py's ``h2_ansatz(theta)`` exactly.

       - ``hardware_efficient_ansatz``  the standard "HEA" pattern used
         across the VQE literature: layers of single-qubit RY rotations
         alternating with a ladder of CX entanglers.

  2. The qg-native versions of both: ``qg_ry_layer``, and
     ``hardware_efficient_ansatz_qg``, whose rotation parameters are
     supplied directly as qg_Z values (qang.core.Qang) rather than raw
     theta angles. This extends Future Research Direction #1's "construct
     circuit parameters natively in qg" (already implemented at the
     single-qubit-gate level in qang.qiskit_gate) up to the ansatz level:
     a full variational layer can now be parameterized entirely in
     physically meaningful qg units.

This module is optional: importing ``qang`` itself never requires Qiskit.
Only importing *this* module does, and it raises a clear, actionable error
if Qiskit is not installed (same pattern as qang.qiskit_gate).
"""

from __future__ import annotations

from typing import Sequence, Union

from .core import Qang

try:
    from qiskit import QuantumCircuit

    _QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when qiskit is absent
    _QISKIT_AVAILABLE = False


def _require_qiskit():
    if not _QISKIT_AVAILABLE:
        raise ImportError(
            "qang.ansatze requires Qiskit. Install it with "
            "`pip install qiskit` (and, to run circuits, `pip install qiskit-aer`)."
        )


# --------------------------------------------------------------------- #
# theta-space ansatze
# --------------------------------------------------------------------- #
def single_excitation_ansatz(
    n_qubits: int, occupied_qubit: int, virtual_qubit: int, theta: float
) -> "QuantumCircuit":
    """
    Minimal single-parameter excitation ansatz: X on ``occupied_qubit``
    (the Hartree-Fock reference), RY(theta) on ``virtual_qubit``, then a
    CX from ``virtual_qubit`` back to ``occupied_qubit`` to inject the
    excitation. ``single_excitation_ansatz(2, 0, 1, theta)`` reproduces
    examples/vqe_h2_qg_vs_theta.py's ``h2_ansatz`` exactly.
    """
    _require_qiskit()
    if occupied_qubit == virtual_qubit:
        raise ValueError("occupied_qubit and virtual_qubit must differ.")
    if not (0 <= occupied_qubit < n_qubits and 0 <= virtual_qubit < n_qubits):
        raise ValueError(f"qubit indices must lie in [0, {n_qubits - 1}].")

    qc = QuantumCircuit(n_qubits, name="single_excitation")
    qc.x(occupied_qubit)
    qc.ry(theta, virtual_qubit)
    qc.cx(virtual_qubit, occupied_qubit)
    return qc


def hardware_efficient_ansatz(
    n_qubits: int, reps: int, params: Sequence[float]
) -> "QuantumCircuit":
    """
    Standard hardware-efficient ansatz: ``reps + 1`` layers of RY
    rotations (one angle per qubit per layer), each pair of consecutive
    layers separated by a ladder of CX entanglers (qubit i -> i+1).

    ``params`` must have exactly ``n_qubits * (reps + 1)`` entries, laid
    out layer-by-layer (params[0:n_qubits] is the first rotation layer,
    and so on).
    """
    _require_qiskit()
    if reps < 0:
        raise ValueError(f"reps must be >= 0, got {reps}.")
    expected = n_qubits * (reps + 1)
    params = list(params)
    if len(params) != expected:
        raise ValueError(f"expected {expected} params (n_qubits * (reps + 1)), got {len(params)}.")

    qc = QuantumCircuit(n_qubits, name="hea")
    idx = 0
    for layer in range(reps + 1):
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        if layer < reps:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
    return qc


# --------------------------------------------------------------------- #
# qg-native ansatze: parameters are qg_Z values, not raw angles
# --------------------------------------------------------------------- #
def qg_ry_layer(
    qc: "QuantumCircuit", qubits: Sequence[int], qg_values: Sequence[Union[float, Qang]]
) -> "QuantumCircuit":
    """
    Append one RY rotation per qubit in ``qubits``, with each angle
    derived from a qg_Z value via ``Qang(qg_value).to_theta()`` rather
    than supplied directly as a raw angle -- the ansatz-layer analogue of
    qang.qiskit_gate.RQangGate, applied across a whole register at once.
    """
    _require_qiskit()
    if len(qubits) != len(qg_values):
        raise ValueError(
            f"qubits and qg_values must have the same length, got {len(qubits)} and {len(qg_values)}."
        )
    for q, v in zip(qubits, qg_values):
        qg = v if isinstance(v, Qang) else Qang(v, mode="polar")
        qc.ry(qg.to_theta(), q)
    return qc


def hardware_efficient_ansatz_qg(
    n_qubits: int, reps: int, qg_params: Sequence[Union[float, Qang]]
) -> "QuantumCircuit":
    """
    Exactly ``hardware_efficient_ansatz``, except every rotation angle is
    supplied as a qg_Z value (qang.core.Qang or a bare float in
    [-1, 1]) instead of a raw theta -- so the whole variational ansatz can
    be tuned directly in the physically meaningful qg unit end to end,
    extending Future Research Direction #1 from a single gate to a full
    multi-layer ansatz.
    """
    _require_qiskit()
    if reps < 0:
        raise ValueError(f"reps must be >= 0, got {reps}.")
    expected = n_qubits * (reps + 1)
    qg_params = list(qg_params)
    if len(qg_params) != expected:
        raise ValueError(
            f"expected {expected} qg_params (n_qubits * (reps + 1)), got {len(qg_params)}."
        )

    qc = QuantumCircuit(n_qubits, name="hea_qg")
    idx = 0
    for layer in range(reps + 1):
        layer_qubits = list(range(n_qubits))
        layer_qg = qg_params[idx: idx + n_qubits]
        qg_ry_layer(qc, layer_qubits, layer_qg)
        idx += n_qubits
        if layer < reps:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
    return qc
