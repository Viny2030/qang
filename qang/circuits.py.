"""
qang.circuits -- reusable Qiskit circuit constructors for standard
multi-qubit states, plus a direct bridge from "a circuit" to "its qg
profile" (qang.multiqubit), promoted out of duplicated test-only helpers
into public library API.

Three things live here:

  1. Standard state circuits (Bell, GHZ, W) -- the canonical worked
     examples already used throughout the paper's multi-qubit discussion
     (Section 2.2's Bell-state entanglement witness; qang.multiqubit's own
     module docstring), now available as actual QuantumCircuit builders
     rather than only as closed-form statevectors
     (qang.multiqubit.bell_state).

  2. ``qg_z_profile_of_circuit`` / ``joint_qg_s_of_circuit`` -- the exact
     "simulate, fix Qiskit's little-endian qubit order, then hand the
     statevector to qang.multiqubit" pattern that was previously
     duplicated as a private helper in three separate test files
     (tests/test_deutsch_jozsa_bv.py, tests/test_qec_syndrome_extraction.py,
     tests/test_phase.py's Hadamard-test helper). It is promoted here once,
     publicly, so any future example or test can go directly from a
     QuantumCircuit to its qg_Z / qg_S profile in one call.

  3. Closed-form qg-profile predictions for each standard state, so the
     circuit-level and analytic answers can be cross-checked against each
     other (see tests/test_circuits.py) rather than trusting either alone.

This module is optional: importing ``qang`` itself never requires Qiskit.
Only importing *this* module does, and it raises a clear, actionable error
if Qiskit is not installed (same pattern as qang.qiskit_gate).
"""

from __future__ import annotations

from typing import List, Optional

from .multiqubit import joint_qg_s, per_qubit_qg_z

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    _QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when qiskit is absent
    _QISKIT_AVAILABLE = False


def _require_qiskit():
    if not _QISKIT_AVAILABLE:
        raise ImportError(
            "qang.circuits requires Qiskit. Install it with "
            "`pip install qiskit` (and, to run circuits, `pip install qiskit-aer`)."
        )


# --------------------------------------------------------------------- #
# circuit -> qg profile bridge
# --------------------------------------------------------------------- #
def qg_z_profile_of_circuit(qc: "QuantumCircuit", n_qubits: Optional[int] = None) -> List[float]:
    """
    Per-qubit qg_Z of ``qc``'s output state, indexed to match Qiskit's own
    qubit numbering.

    Qiskit's Statevector is little-endian (qubit 0 is the *least*
    significant tensor factor), while qang.multiqubit.per_qubit_qg_z
    expects qubit 0 to be the *leftmost* (most significant) tensor factor.
    ``reverse_qargs()`` reconciles the two conventions -- see
    tests/test_deutsch_jozsa_bv.py for the original numerical verification
    of this fix (an X-gate test case).
    """
    _require_qiskit()
    if n_qubits is None:
        n_qubits = qc.num_qubits
    sv = Statevector.from_instruction(qc).reverse_qargs().data
    return per_qubit_qg_z(sv, n_qubits=n_qubits)


def joint_qg_s_of_circuit(
    qc: "QuantumCircuit", n_qubits: Optional[int] = None, normalize: bool = True
) -> float:
    """Joint qg_S (Shannon entropy of the full measurement-outcome
    distribution) of ``qc``'s output state. Reversal doesn't matter here
    (the outcome distribution over |x> is the same set of probabilities
    regardless of qubit-label order), but n_qubits still needs to match."""
    _require_qiskit()
    if n_qubits is None:
        n_qubits = qc.num_qubits
    sv = Statevector.from_instruction(qc).data
    return joint_qg_s(sv, n_qubits=n_qubits, normalize=normalize)


# --------------------------------------------------------------------- #
# standard state circuits
# --------------------------------------------------------------------- #
def bell_circuit(kind: str = "phi_plus") -> "QuantumCircuit":
    """
    2-qubit circuit preparing one of the four Bell states (Qiskit's own
    little-endian convention), matching qang.multiqubit.bell_state(kind)
    exactly -- see tests/test_circuits.py for the cross-check.
    """
    _require_qiskit()
    kinds = {"phi_plus", "phi_minus", "psi_plus", "psi_minus"}
    if kind not in kinds:
        raise ValueError(f"kind must be one of {sorted(kinds)}, got {kind!r}.")

    qc = QuantumCircuit(2, name=f"bell_{kind}")
    qc.h(0)
    qc.cx(0, 1)
    if kind in ("phi_minus", "psi_minus"):
        qc.z(0)
    if kind in ("psi_plus", "psi_minus"):
        qc.x(0)
    return qc


def ghz_circuit(n_qubits: int) -> "QuantumCircuit":
    """
    n-qubit GHZ state (|00...0> + |11...1>) / sqrt(2): H on qubit 0
    followed by a CX ladder -- the standard textbook construction, and the
    natural n-qubit generalization of the Bell state above (n=2 reduces
    to bell_circuit('phi_plus')).
    """
    _require_qiskit()
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}.")
    qc = QuantumCircuit(n_qubits, name=f"ghz_{n_qubits}")
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def w_circuit(n_qubits: int) -> "QuantumCircuit":
    """
    n-qubit W state (equal superposition of every single-excitation basis
    state, e.g. n=3: (|100> + |010> + |001>) / sqrt(3)), via the standard
    recursive controlled-rotation construction: start with a single |1>
    on the last qubit, then repeatedly "share" the excitation with the
    next qubit down via a controlled-RY followed by a CX.

    Correctness (exact analytic amplitudes, not just "looks right") is
    checked in tests/test_circuits.py against the closed-form W-state
    definition for n = 2 through 6.
    """
    _require_qiskit()
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}.")
    import math

    qc = QuantumCircuit(n_qubits, name=f"w_{n_qubits}")
    if n_qubits == 1:
        qc.x(0)
        return qc

    qc.x(n_qubits - 1)
    for k in range(n_qubits - 1):
        control = n_qubits - 1 - k
        target = n_qubits - 2 - k
        remaining = n_qubits - k
        theta = 2.0 * math.acos(math.sqrt(1.0 / remaining))
        qc.cry(theta, control, target)
        qc.cx(target, control)
    return qc


# --------------------------------------------------------------------- #
# closed-form qg-profile predictions, for cross-checking the circuits above
# --------------------------------------------------------------------- #
def ghz_qg_z_profile(n_qubits: int) -> List[float]:
    """Every qubit of a GHZ state is maximally mixed on its own (qg_Z=0),
    the same entanglement witness already used for the Bell state in
    qang.multiqubit's module docstring, generalized to n qubits."""
    return [0.0] * n_qubits


def ghz_joint_qg_s(n_qubits: int, normalize: bool = True) -> float:
    """A GHZ state's measurement outcomes are 50/50 on |00...0> and
    |11...1> regardless of n, i.e. exactly 1 bit of Shannon entropy;
    normalized by n_qubits per qang.multiqubit.joint_qg_s's convention."""
    if n_qubits <= 1:
        return 1.0
    return 1.0 / n_qubits if normalize else 1.0


def w_qg_z_profile(n_qubits: int) -> List[float]:
    """Each qubit of a W state has P(|1>) = 1/n_qubits by symmetry, so
    qg_Z = P(0) - P(1) = (1 - 2/n_qubits), identical for every qubit."""
    return [1.0 - 2.0 / n_qubits] * n_qubits
