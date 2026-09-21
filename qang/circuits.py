"""
qang.circuits -- reusable Qiskit circuit constructors for standard
multi-qubit states, plus a direct bridge from "a circuit" to "its qg
profile" (qang.multiqubit), promoted out of duplicated test-only helpers
into public library API.

Four things live here:

  1. Standard state circuits (Bell, GHZ, W, plus graph states, cluster
     states, and generalized Dicke states) -- the canonical worked
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

  4. An honest boundary case, made explicit by the graph/cluster-state
     addition: every graph state has qg_Z = 0 on every qubit AND joint
     qg_S = 1.0 (maximal), *regardless of which graph it is built from*.
     This is not a bug -- H^{\\otimes n} already puts every qubit's
     Z-basis marginal at 50/50, and every CZ gate is diagonal in the
     computational basis, so it can only add phases, never change any
     |amplitude|^2. A graph state's entanglement (confirmed independently
     below via its stabilizer group, e.g. X_i * prod_{j in N(i)} Z_j = +1)
     therefore lives entirely in phase relationships between computational-
     basis components -- structure that qg_Z and qg_S, both defined purely
     from Z-basis (diagonal) statistics, cannot see at all. Unlike GHZ or W
     (whose entanglement happens to coincide with a non-trivial qg_S), a
     graph state is a clean counterexample showing qg_S is not a universal
     entanglement measure -- a limitation worth stating as plainly as the
     paper's own Section 4.1 coordinate singularity.

This module is optional: importing ``qang`` itself never requires Qiskit.
Only importing *this* module does, and it raises a clear, actionable error
if Qiskit is not installed (same pattern as qang.qiskit_gate).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .multiqubit import joint_qg_s, per_qubit_qg_z

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UCRYGate
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


def graph_state_circuit(n_qubits: int, edges: Sequence[Tuple[int, int]]) -> "QuantumCircuit":
    """
    Graph state for an arbitrary graph on ``n_qubits`` vertices and the
    given ``edges`` (each a (i, j) pair of qubit indices): H on every
    qubit, then a CZ for every edge. This is the standard graph-state
    construction (Hein, Eisert & Briegel, 2004); ``linear_cluster_state_circuit``
    below is the special case where the graph is a path.

    Correctness is checked in tests/test_circuits.py via the stabilizer
    condition X_i * prod_{j in neighbors(i)} Z_j |psi> = +|psi> for every
    vertex i, rather than by comparing raw amplitudes.
    """
    _require_qiskit()
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}.")
    qc = QuantumCircuit(n_qubits, name=f"graph_state_{n_qubits}")
    for q in range(n_qubits):
        qc.h(q)
    for (i, j) in edges:
        if not (0 <= i < n_qubits and 0 <= j < n_qubits) or i == j:
            raise ValueError(f"invalid edge {(i, j)!r} for n_qubits={n_qubits}.")
        qc.cz(i, j)
    return qc


def linear_cluster_state_circuit(n_qubits: int) -> "QuantumCircuit":
    """
    1-D cluster state on a path graph 0-1-2-...-(n_qubits-1): the
    canonical resource state for measurement-based quantum computing on a
    line. Just ``graph_state_circuit`` specialized to path edges.
    """
    _require_qiskit()
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}.")
    edges = [(i, i + 1) for i in range(n_qubits - 1)]
    qc = graph_state_circuit(n_qubits, edges)
    qc.name = f"linear_cluster_{n_qubits}"
    return qc


def dicke_state_circuit(n_qubits: int, k: int) -> "QuantumCircuit":
    """
    Generalized Dicke state |D(n_qubits, k)>: the equal superposition of
    every n_qubits-bit basis state with exactly k ones (k=1 is exactly
    the W state above, and dicke_state_circuit(n, 1) reproduces
    w_circuit(n) up to global phase).

    Built via a recursive hypergeometric-conditional-rotation scheme: for
    qubit i (processed in order 0, 1, ..., n_qubits-1), conditioned on
    the number of ones already placed among qubits 0..i-1 (w = popcount
    of the control register's computational-basis index), qubit i is
    rotated so that P(qubit i = 1) equals the hypergeometric probability
    of still being able to place exactly k ones total:

        remaining_qubits = n_qubits - i
        ones_needed = k - w
        ones_needed <= 0            -> theta = 0       (force |0>)
        ones_needed >= remaining_qubits -> theta = pi   (force |1>)
        otherwise                   -> theta = 2*acos(sqrt(1 - ones_needed/remaining_qubits))

    Only popcount(j) (not the individual control bits) determines the
    angle, so this is correct regardless of Qiskit's internal bit-to-qubit
    ordering for UCRYGate's controls -- verified exactly (np.allclose,
    atol=1e-9) against the closed-form Dicke amplitudes for n_qubits in
    2..6 and every k in 0..n_qubits (see tests/test_circuits.py).
    """
    _require_qiskit()
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}.")
    if not (0 <= k <= n_qubits):
        raise ValueError(f"k must satisfy 0 <= k <= n_qubits, got k={k}, n_qubits={n_qubits}.")

    qc = QuantumCircuit(n_qubits, name=f"dicke_{n_qubits}_{k}")
    if k == 0:
        return qc
    if k == n_qubits:
        for q in range(n_qubits):
            qc.x(q)
        return qc

    for i in range(n_qubits):
        remaining_qubits = n_qubits - i
        n_controls = i
        angle_list = []
        for j in range(2 ** n_controls):
            w = bin(j).count("1")
            ones_needed = k - w
            if ones_needed <= 0:
                theta = 0.0
            elif ones_needed >= remaining_qubits:
                theta = math.pi
            else:
                p1 = ones_needed / remaining_qubits
                theta = 2.0 * math.acos(math.sqrt(1.0 - p1))
            angle_list.append(theta)

        if n_controls == 0:
            qc.ry(angle_list[0], 0)
        else:
            qc.append(UCRYGate(angle_list), [i] + list(range(n_controls)))
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


def graph_state_qg_z_profile(n_qubits: int) -> List[float]:
    """
    Every graph state (any graph, any number of edges) has qg_Z = 0 on
    every qubit: H^{\\otimes n} already puts each qubit's own Z-basis
    marginal at exactly 50/50, and every subsequent CZ is diagonal in the
    computational basis, so it can only add phases -- it never changes
    any |amplitude|^2, hence never changes any marginal. See this
    module's docstring (point 4) for the full discussion of why this
    makes qg_Z/qg_S blind to graph-state entanglement.
    """
    return [0.0] * n_qubits


def graph_state_joint_qg_s(n_qubits: int, normalize: bool = True) -> float:
    """
    Every graph state's measurement-outcome distribution is exactly
    uniform over all 2**n_qubits computational basis states (same
    reasoning as graph_state_qg_z_profile: H^{\\otimes n} gives uniform
    |amplitude|^2 = 1/2**n_qubits everywhere, and CZ cannot change that),
    i.e. exactly n_qubits bits of Shannon entropy -- the maximum
    possible, and (per qang.multiqubit.joint_qg_s's normalize convention,
    dividing by n_qubits) normalized joint qg_S = 1.0 regardless of the
    graph's edge structure.
    """
    return 1.0 if normalize else float(n_qubits)


def dicke_qg_z_profile(n_qubits: int, k: int) -> List[float]:
    """
    Each qubit of the generalized Dicke state |D(n_qubits, k)> has
    P(|1>) = k / n_qubits by symmetry (every weight-k basis state is
    equally likely, and every qubit position is equally likely to be one
    of the k excited ones), so qg_Z = 1 - 2*k/n_qubits for every qubit --
    this generalizes w_qg_z_profile, which is exactly the k=1 case.
    """
    if not (0 <= k <= n_qubits):
        raise ValueError(f"k must satisfy 0 <= k <= n_qubits, got k={k}, n_qubits={n_qubits}.")
    return [1.0 - 2.0 * k / n_qubits] * n_qubits
