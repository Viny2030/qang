"""
Tests for examples/spin_checks_qg.py (RESEARCH_NOTES §55).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")

from qiskit.quantum_info import Statevector  # noqa: E402

import spin_checks_qg as P  # noqa: E402


def test_fast_statevector_matches_qiskit_and_conserves_spin():
    rng = np.random.default_rng(0)
    n, ne = 8, 4
    x = rng.normal(size=P.n_params(n))
    psi = P.statevector(x, n, ne)
    assert abs(np.vdot(psi, Statevector(P.circuit(x, n, ne)).data)) == pytest.approx(1.0)
    up, dn = P._spin_counts(n)
    assert np.sum(np.abs(psi[(up == 2) & (dn == 2)]) ** 2) == pytest.approx(1.0)


def test_hamiltonian_conserves_n_up_and_n_down():
    op, mat, n, ne, _, _ = P.S.molecule("H2O")
    up, dn = P._spin_counts(n)
    M = mat.toarray()
    for c in (up, dn):
        D = np.diag(c.astype(float))
        assert np.allclose(M @ D, D @ M)


def test_spin_projection_beats_number_projection_on_h2o():
    params = 0.1 * np.random.default_rng(1).standard_normal(P.n_params(8))
    a = P.level_a("H2O", params)
    assert a["ideal_in_spin_sector"] == pytest.approx(1.0)
    assert a["spin"] < a["N"] < a["raw"]
    assert a["spin_parity"] > a["N"]
