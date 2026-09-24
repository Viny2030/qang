"""
Tests for examples/chemistry_qg_symmetry_witness.py: H2 in Jordan-Wigner,
classical references vs noisy quantum energies with and without the qg
filter.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("scipy")

from qiskit.quantum_info import Statevector  # noqa: E402

from chemistry_qg_symmetry_witness import (  # noqa: E402
    FCI_ENERGY,
    HF_ENERGY,
    IDEAL_MEAN_QG_Z,
    ansatz,
    exact_energy,
    optimal_angle,
    run_noisy,
)


def test_hamiltonian_ground_state_and_hartree_fock_point():
    assert exact_energy(0.0) == pytest.approx(HF_ENERGY, abs=1e-9)
    assert exact_energy(optimal_angle()) == pytest.approx(FCI_ENERGY, abs=1e-9)


@pytest.mark.parametrize("t", [0.0, 0.3, 1.7, -2.4])
def test_ansatz_conserves_electron_number_so_ideal_mean_qg_z_is_known(t):
    p = np.abs(Statevector(ansatz(t)).data) ** 2
    weights = np.array([bin(i).count("1") for i in range(16)])
    assert np.all(p[weights != 2] < 1e-12)
    assert np.sum(p * (1 - 2 * weights / 4)) == pytest.approx(IDEAL_MEAN_QG_Z, abs=1e-12)


@pytest.fixture(scope="module")
def noisy_rows():
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_ibm_runtime")
    return {d: run_noisy(d, seed=11) for d in (0, 50)}


def test_mean_qg_z_witnesses_electron_loss(noisy_rows):
    r0, r50 = noisy_rows[0], noisy_rows[50]
    assert 0 < r0["mean_qg_z"] < r50["mean_qg_z"]
    assert r50["mean_qg_z"] > 0.1
    assert r50["kept_fraction"] < r0["kept_fraction"]


def test_qg_filter_beats_readout_mitigation_and_hartree_fock(noisy_rows):
    hf_err = HF_ENERGY - FCI_ENERGY
    r0 = noisy_rows[0]
    err = {k: abs(r0[k] - FCI_ENERGY) for k in ("raw", "readout", "readout_qg_filter")}
    assert err["readout_qg_filter"] < err["readout"] / 2 < err["raw"]
    assert err["readout_qg_filter"] < hf_err / 2
    r50 = noisy_rows[50]
    assert abs(r50["readout_qg_filter"] - FCI_ENERGY) < abs(r50["readout"] - FCI_ENERGY) / 4
