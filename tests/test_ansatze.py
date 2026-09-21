"""
Tests for qang.ansatze: reusable variational ansatz constructors, both in
theta-space and in the new qg-native form (parameters supplied directly
as qg_Z values).

Checked:
  1. single_excitation_ansatz(2, 0, 1, theta) exactly reproduces
     examples/vqe_h2_qg_vs_theta.py's h2_ansatz(theta), statevector for
     statevector, across several theta values.
  2. hardware_efficient_ansatz raises on a mismatched parameter count,
     and produces the expected statevector for a hand-computed 1-qubit,
     0-rep case (a bare RY).
  3. qg_ry_layer applied with qg_Z = cos(theta) reproduces a plain RY(theta)
     layer exactly, for both a bare-float and an explicit Qang input.
  4. hardware_efficient_ansatz_qg with qg_params = cos(theta_i) exactly
     reproduces hardware_efficient_ansatz with those same theta_i, for
     several (n_qubits, reps) configurations -- confirming the qg-native
     ansatz is a faithful, purely-relabeled reparameterization of the
     standard one, not an approximation of it.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import math

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qang.core import Qang
from qang.ansatze import (
    hardware_efficient_ansatz,
    hardware_efficient_ansatz_qg,
    qg_ry_layer,
    single_excitation_ansatz,
)


@pytest.mark.parametrize("theta", [0.0, 0.3, 1.0, 2.0, math.pi - 0.1])
def test_single_excitation_ansatz_matches_h2_example(theta):
    from vqe_h2_qg_vs_theta import h2_ansatz

    qc_lib = single_excitation_ansatz(2, occupied_qubit=0, virtual_qubit=1, theta=theta)
    qc_example = h2_ansatz(theta)
    sv_lib = Statevector.from_instruction(qc_lib).data
    sv_example = Statevector.from_instruction(qc_example).data
    assert np.allclose(sv_lib, sv_example, atol=1e-12)


def test_single_excitation_ansatz_rejects_same_qubit():
    with pytest.raises(ValueError):
        single_excitation_ansatz(2, occupied_qubit=0, virtual_qubit=0, theta=0.5)


def test_hardware_efficient_ansatz_rejects_wrong_param_count():
    with pytest.raises(ValueError):
        hardware_efficient_ansatz(3, reps=1, params=[0.1, 0.2, 0.3])  # needs 6


def test_hardware_efficient_ansatz_single_qubit_zero_reps_is_bare_ry():
    theta = 0.77
    qc = hardware_efficient_ansatz(1, reps=0, params=[theta])
    reference = QuantumCircuit(1)
    reference.ry(theta, 0)
    assert np.allclose(
        Statevector.from_instruction(qc).data,
        Statevector.from_instruction(reference).data,
        atol=1e-12,
    )


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.5, 2.7, math.pi])
def test_qg_ry_layer_matches_plain_ry_with_float_qg(theta):
    qg_value = math.cos(theta)
    qc = QuantumCircuit(1)
    qg_ry_layer(qc, [0], [qg_value])
    reference = QuantumCircuit(1)
    reference.ry(theta, 0)
    assert np.allclose(
        Statevector.from_instruction(qc).data,
        Statevector.from_instruction(reference).data,
        atol=1e-9,
    )


def test_qg_ry_layer_accepts_explicit_qang_instances():
    theta = 1.234
    qg = Qang(math.cos(theta), mode="polar")
    qc = QuantumCircuit(2)
    qg_ry_layer(qc, [0, 1], [qg, qg])
    reference = QuantumCircuit(2)
    reference.ry(theta, 0)
    reference.ry(theta, 1)
    assert np.allclose(
        Statevector.from_instruction(qc).data,
        Statevector.from_instruction(reference).data,
        atol=1e-9,
    )


def test_qg_ry_layer_rejects_length_mismatch():
    qc = QuantumCircuit(2)
    with pytest.raises(ValueError):
        qg_ry_layer(qc, [0, 1], [0.5])


@pytest.mark.parametrize("n_qubits,reps", [(1, 0), (2, 1), (3, 2), (4, 1)])
def test_hardware_efficient_ansatz_qg_matches_theta_space_version(n_qubits, reps):
    rng = np.random.default_rng(42 + n_qubits * 10 + reps)
    n_params = n_qubits * (reps + 1)
    thetas = rng.uniform(0.0, math.pi, size=n_params)
    qg_params = np.cos(thetas)

    qc_theta = hardware_efficient_ansatz(n_qubits, reps, thetas)
    qc_qg = hardware_efficient_ansatz_qg(n_qubits, reps, qg_params)

    sv_theta = Statevector.from_instruction(qc_theta).data
    sv_qg = Statevector.from_instruction(qc_qg).data
    assert np.allclose(sv_theta, sv_qg, atol=1e-9)


def test_hardware_efficient_ansatz_qg_rejects_wrong_param_count():
    with pytest.raises(ValueError):
        hardware_efficient_ansatz_qg(2, reps=1, qg_params=[0.1, 0.2, 0.3])  # needs 4


if __name__ == "__main__":
    print("Run via `pytest tests/test_ansatze.py -v` for full parametrized coverage.")
    qc = single_excitation_ansatz(2, 0, 1, 0.5)
    assert qc.num_qubits == 2
    print("Smoke check passed.")
