"""
Tests for qang.knitting: the closed-form cutting cost in qg units, pinned
against qiskit-addon-cutting's own QPD decompositions, plus an end-to-end
cut-and-reconstruct check on a circuit whose observables depend on theta.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from qang.knitting import (
    controlled_rotation_cut_gamma,
    cut_sampling_overhead,
    pauli_rotation_cut_gamma,
    total_sampling_overhead,
)

THETAS = [0.0, 0.1, 0.4, math.pi / 4, 1.2, math.pi / 2, 2.5, math.pi]


def test_anchor_values():
    assert pauli_rotation_cut_gamma(1.0) == pytest.approx(1.0)
    assert pauli_rotation_cut_gamma(-1.0) == pytest.approx(1.0)
    assert pauli_rotation_cut_gamma(0.0) == pytest.approx(3.0)
    assert cut_sampling_overhead(0.0) == pytest.approx(9.0)
    assert controlled_rotation_cut_gamma(1.0) == pytest.approx(1.0)
    assert controlled_rotation_cut_gamma(-1.0) == pytest.approx(3.0)


def test_rejects_out_of_range_qg():
    with pytest.raises(ValueError):
        pauli_rotation_cut_gamma(1.2)
    with pytest.raises(ValueError):
        cut_sampling_overhead(0.5, gate="toffoli")


def test_total_overhead_is_product_of_single_cuts():
    qgs = [0.9, 0.0, -0.3]
    expected = np.prod([cut_sampling_overhead(q) for q in qgs])
    assert total_sampling_overhead(qgs) == pytest.approx(expected)


@pytest.mark.parametrize("theta", THETAS)
def test_pauli_rotations_match_qiskit_addon_cutting(theta):
    pytest.importorskip("qiskit_addon_cutting")
    from qiskit.circuit.library import RXXGate, RYYGate, RZXGate, RZZGate
    from qiskit_addon_cutting.qpd import QPDBasis

    for gate in (RXXGate, RYYGate, RZZGate, RZXGate):
        lib = QPDBasis.from_instruction(gate(theta)).overhead  # gamma^2
        assert cut_sampling_overhead(math.cos(theta)) == pytest.approx(lib, abs=1e-9)


@pytest.mark.parametrize("theta", THETAS)
def test_controlled_rotations_match_qiskit_addon_cutting(theta):
    pytest.importorskip("qiskit_addon_cutting")
    from qiskit.circuit.library import CPhaseGate, CRXGate, CRYGate, CRZGate
    from qiskit_addon_cutting.qpd import QPDBasis

    for gate in (CRXGate, CRYGate, CRZGate, CPhaseGate):
        lib = QPDBasis.from_instruction(gate(theta)).overhead
        got = cut_sampling_overhead(math.cos(theta), gate="controlled_rotation")
        assert got == pytest.approx(lib, abs=1e-9)


def test_x0_on_plus_plus_reads_the_qg_of_an_rzz_coupling():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector

    for theta in np.linspace(0, math.pi, 7):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.rzz(theta, 0, 1)
        x0 = Statevector(qc).expectation_value(SparsePauliOp("IX")).real
        assert x0 == pytest.approx(math.cos(theta), abs=1e-12)


def test_end_to_end_cut_reconstructs_theta_dependent_observables():
    pytest.importorskip("qiskit_addon_cutting")
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit_addon_cutting import (
        generate_cutting_experiments,
        partition_problem,
        reconstruct_expectation_values,
    )
    from qiskit_aer.primitives import SamplerV2

    theta = math.pi / 4
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.ry(0.7, 1)
    qc.rzz(theta, 0, 1)
    observables = SparsePauliOp(["IX", "XI", "XX"])
    exact = [Statevector(qc).expectation_value(SparsePauliOp(p)).real for p in observables.paulis.to_labels()]

    part = partition_problem(circuit=qc, partition_labels="AB", observables=observables.paulis)
    subexp, coeffs = generate_cutting_experiments(
        circuits=part.subcircuits, observables=part.subobservables, num_samples=np.inf
    )
    sampler = SamplerV2(seed=42)
    results = {label: sampler.run(subexp[label], shots=20000).result() for label in subexp}
    rec = reconstruct_expectation_values(results, coeffs, part.subobservables)

    assert sum(abs(c) for c, _ in coeffs) == pytest.approx(pauli_rotation_cut_gamma(math.cos(theta)))
    for e, r in zip(exact, rec):
        assert r == pytest.approx(e, abs=0.03)
