"""
Tests for qang.circuits: standard state circuits (Bell, GHZ, W, graph
states, cluster states, generalized Dicke states), the promoted
circuit-to-qg-profile bridge, and their closed-form qg predictions.

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
  5. graph_state_circuit() / linear_cluster_state_circuit() satisfy the
     stabilizer condition X_i * prod_{j in N(i)} Z_j = +1 for every
     vertex i, for several graph topologies -- and, regardless of graph
     structure, always have qg_Z = 0 on every qubit and joint qg_S = 1.0
     (the "qg_S is blind to graph-state entanglement" boundary case
     documented in qang.circuits' module docstring).
  6. dicke_state_circuit(n, k) matches the closed-form Dicke amplitudes
     exactly for every n in 2..6 and every k in 0..n, its qg_Z profile
     matches dicke_qg_z_profile, and dicke_state_circuit(n, 1) equals
     w_circuit(n) up to global phase (k=1 is the W state).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit.quantum_info import Pauli, Statevector

from qang.multiqubit import bell_state, joint_qg_s, per_qubit_qg_z
from qang.circuits import (
    bell_circuit,
    dicke_qg_z_profile,
    dicke_state_circuit,
    ghz_circuit,
    ghz_joint_qg_s,
    ghz_qg_z_profile,
    graph_state_circuit,
    graph_state_joint_qg_s,
    graph_state_qg_z_profile,
    joint_qg_s_of_circuit,
    linear_cluster_state_circuit,
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


def _stabilizer_expectation(qc, vertex, neighbors, n_qubits):
    """X on ``vertex``, Z on each of ``neighbors``, I elsewhere -- built as
    a Qiskit Pauli label with index 0 as the *rightmost* character
    (Qiskit's own convention), then evaluated exactly via Statevector."""
    label = ["I"] * n_qubits
    label[vertex] = "X"
    for j in neighbors:
        label[j] = "Z"
    label_str = "".join(reversed(label))
    sv = Statevector.from_instruction(qc)
    return sv.expectation_value(Pauli(label_str)).real


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
def test_star_graph_state_satisfies_stabilizer_condition(n_qubits):
    edges = [(0, i) for i in range(1, n_qubits)]
    qc = graph_state_circuit(n_qubits, edges)
    neighbors = {0: list(range(1, n_qubits))}
    for i in range(1, n_qubits):
        neighbors[i] = [0]
    for vertex in range(n_qubits):
        val = _stabilizer_expectation(qc, vertex, neighbors[vertex], n_qubits)
        assert val == pytest.approx(1.0, abs=1e-9)


def test_linear_cluster_state_satisfies_stabilizer_condition():
    n_qubits = 4
    qc = linear_cluster_state_circuit(n_qubits)
    neighbors = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    for vertex in range(n_qubits):
        val = _stabilizer_expectation(qc, vertex, neighbors[vertex], n_qubits)
        assert val == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
def test_graph_state_qg_z_is_always_zero_regardless_of_graph(n_qubits):
    """The honest boundary case documented in qang.circuits' module
    docstring: H^n gives every qubit a 50/50 marginal, and CZ is
    diagonal (phase-only), so qg_Z = 0 on every qubit no matter which
    edges are present -- checked here for a star graph."""
    edges = [(0, i) for i in range(1, n_qubits)]
    qc = graph_state_circuit(n_qubits, edges)
    numeric = qg_z_profile_of_circuit(qc, n_qubits)
    assert np.allclose(numeric, graph_state_qg_z_profile(n_qubits), atol=1e-9)
    assert np.allclose(numeric, [0.0] * n_qubits, atol=1e-9)


def test_graph_state_joint_qg_s_is_always_maximal_regardless_of_graph():
    """Same boundary case for joint qg_S: every graph state's outcome
    distribution is exactly uniform over all 2**n basis states, so
    normalized joint qg_S = 1.0 (maximal) regardless of the graph."""
    n_qubits = 4
    # path plus one isolated vertex (vertex 3 has no edges at all)
    edges = [(0, 1), (1, 2)]
    qc = graph_state_circuit(n_qubits, edges)
    numeric = joint_qg_s_of_circuit(qc, n_qubits, normalize=True)
    assert numeric == pytest.approx(graph_state_joint_qg_s(n_qubits, normalize=True), abs=1e-9)
    assert numeric == pytest.approx(1.0, abs=1e-9)
    # and the isolated vertex still has qg_Z = 0, same as every other qubit
    profile = qg_z_profile_of_circuit(qc, n_qubits)
    assert np.allclose(profile, [0.0] * n_qubits, atol=1e-9)


def test_graph_state_circuit_rejects_invalid_edge():
    with pytest.raises(ValueError):
        graph_state_circuit(3, [(0, 3)])  # qubit 3 doesn't exist
    with pytest.raises(ValueError):
        graph_state_circuit(3, [(1, 1)])  # self-loop


def _analytic_dicke_statevector(n_qubits, k):
    from math import comb

    dim = 2 ** n_qubits
    sv = np.zeros(dim, dtype=complex)
    if comb(n_qubits, k) == 0:
        return sv
    norm = 1.0 / math.sqrt(comb(n_qubits, k))
    for x in range(dim):
        bits = [(x >> i) & 1 for i in range(n_qubits)]  # Qiskit little-endian: bit i = qubit i
        if sum(bits) == k:
            sv[x] = norm
    return sv


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6])
def test_dicke_state_circuit_matches_analytic_amplitudes_for_every_k(n_qubits):
    for k in range(n_qubits + 1):
        qc = dicke_state_circuit(n_qubits, k)
        sv = Statevector.from_instruction(qc).data
        analytic = _analytic_dicke_statevector(n_qubits, k)
        assert np.allclose(np.abs(sv), np.abs(analytic), atol=1e-9)


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6])
def test_dicke_qg_z_profile_matches_circuit_for_every_k(n_qubits):
    for k in range(n_qubits + 1):
        qc = dicke_state_circuit(n_qubits, k)
        numeric = qg_z_profile_of_circuit(qc, n_qubits)
        assert np.allclose(numeric, dicke_qg_z_profile(n_qubits, k), atol=1e-9)


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
def test_dicke_k1_matches_w_circuit_up_to_global_phase(n_qubits):
    sv_dicke = Statevector.from_instruction(dicke_state_circuit(n_qubits, 1)).data
    sv_w = Statevector.from_instruction(w_circuit(n_qubits)).data
    phase = None
    for a, b in zip(sv_w, sv_dicke):
        if abs(a) > 1e-9:
            phase = b / a
            break
    assert phase is not None
    assert np.allclose(sv_dicke, sv_w * phase, atol=1e-9)


def test_dicke_state_circuit_k_equals_zero_and_n():
    n_qubits = 4
    sv_zero = Statevector.from_instruction(dicke_state_circuit(n_qubits, 0)).data
    expected_zero = np.zeros(2 ** n_qubits, dtype=complex)
    expected_zero[0] = 1.0
    assert np.allclose(sv_zero, expected_zero, atol=1e-9)

    sv_all = Statevector.from_instruction(dicke_state_circuit(n_qubits, n_qubits)).data
    expected_all = np.zeros(2 ** n_qubits, dtype=complex)
    expected_all[-1] = 1.0
    assert np.allclose(sv_all, expected_all, atol=1e-9)


def test_dicke_state_circuit_rejects_invalid_k():
    with pytest.raises(ValueError):
        dicke_state_circuit(4, -1)
    with pytest.raises(ValueError):
        dicke_state_circuit(4, 5)


def test_dicke_qg_z_profile_rejects_invalid_k():
    with pytest.raises(ValueError):
        dicke_qg_z_profile(4, -1)
    with pytest.raises(ValueError):
        dicke_qg_z_profile(4, 5)


if __name__ == "__main__":
    print("Run via `pytest tests/test_circuits.py -v` for full parametrized coverage.")
    qc = ghz_circuit(3)
    profile = qg_z_profile_of_circuit(qc, 3)
    assert np.allclose(profile, [0.0, 0.0, 0.0], atol=1e-9)
    print("Smoke check passed.")
