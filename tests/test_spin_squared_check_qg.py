"""
Tests for examples/spin_squared_check_qg.py (RESEARCH_NOTES §69): the S^2
singlet projection after the Z-diagonal checks.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("pyscf")
pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")
pytest.importorskip("qiskit_aer")

import spin_squared_check_qg as Q  # noqa: E402


def test_singlet_projector_is_a_projector():
    P, S2 = Q.singlet_projector(4, 1, 1)
    assert np.allclose(P @ P, P, atol=1e-10)
    assert np.allclose(S2 @ P, 0 * P, atol=1e-10)  # S^2 = 0 on its range
    assert np.real(np.trace(P)) == pytest.approx(3.0)  # 3 singlets for 1 up + 1 down in 2 orbitals (4 states, 1 triplet component)


def test_h2o_s2_gain():
    r = Q.study("H2O", count_paulis=False)
    assert r["singlet_weight_ideal"] > 0.9999
    assert r["S2"] < 0.75 * r["spin"] < r["N"] < r["raw"]
    assert r["kept_S2"] > 0.6
