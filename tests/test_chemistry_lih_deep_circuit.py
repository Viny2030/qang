"""
Tests for examples/chemistry_lih_deep_circuit.py: a 60-ECR, 6-qubit LiH
circuit where the qg witness diagnoses unital noise and the qg filter
does not help.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import pytest

pytest.importorskip("qiskit")

from chemistry_lih_deep_circuit import (  # noqa: E402
    FCI_ENERGY,
    IDEAL_MEAN_QG_Z,
    hartree_fock_energy,
    ideal_energy,
    run_noisy,
)


def test_classical_references_and_noiseless_ansatz():
    assert hartree_fock_energy() - FCI_ENERGY == pytest.approx(16.263e-3, abs=1e-5)
    assert ideal_energy() - FCI_ENERGY == pytest.approx(0.665e-3, abs=1e-5)


def test_deep_circuit_witness_shows_unital_noise_and_filter_does_not_help():
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_ibm_runtime")
    r = run_noisy(seed=11)
    assert r["two_qubit_gates"] >= 50
    assert r["mean_qg_z"] < IDEAL_MEAN_QG_Z - 0.1  # pulled towards 0, not +1
    hf_err = hartree_fock_energy() - FCI_ENERGY
    assert abs(r["readout_qg_filter"] - FCI_ENERGY) >= abs(r["readout"] - FCI_ENERGY)
    assert abs(r["readout"] - FCI_ENERGY) > 5 * hf_err
