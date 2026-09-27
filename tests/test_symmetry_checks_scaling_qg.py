"""
Tests for examples/symmetry_checks_scaling_qg.py (RESEARCH_NOTES §54).
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

import symmetry_checks_scaling_qg as S  # noqa: E402


def test_fast_statevector_matches_qiskit():
    rng = np.random.default_rng(1)
    for n, ne in ((6, 2), (8, 4)):
        x = rng.normal(size=3 * len(S._pairs(n)))
        assert abs(np.vdot(S.statevector(x, n, ne), Statevector(S.circuit(x, n, ne)).data)) == pytest.approx(1.0)


def test_h2o_active_space_hamiltonian():
    op, mat, n, ne, e_exact, e_hf = S.molecule("H2O")
    assert (n, ne) == (8, 4)
    assert e_hf > e_exact
    assert 1e3 * (e_hf - e_exact) == pytest.approx(7.4, abs=0.1)
    w = np.array([bin(i).count("1") for i in range(2**n)])
    Nop = np.diag(w.astype(float))
    M = mat.toarray()
    assert np.allclose(M @ Nop, Nop @ M)


def test_h4_hartree_fock_gap():
    _, _, n, ne, e_exact, e_hf = S.molecule("H4")
    assert (n, ne) == (8, 4)
    assert 1e3 * (e_hf - e_exact) == pytest.approx(167.0, abs=0.1)


def test_checks_ordering_on_h2o():
    params = np.zeros(3 * len(S._pairs(8)))
    a = S.level_a("H2O", params)
    assert a["raw"] > a["parity"] > a["mod4"] >= a["number"] - 1.0
    assert a["mod4"] < 0.3 * a["raw"]
    assert a["kept_mod4"] == pytest.approx(a["kept_number"], abs=0.02)
