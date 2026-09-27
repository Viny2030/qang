"""
Tests for examples/lih_parity_verification_qg.py (RESEARCH_NOTES §52):
symmetry checks that reach every measurement group of LiH.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit.quantum_info import Operator, SparsePauliOp  # noqa: E402

import lih_parity_verification_qg as V  # noqa: E402


def test_parity_and_mod4_commute_with_the_hamiltonian():
    H = V.L.HAMILTONIAN.to_matrix()
    parity = SparsePauliOp("Z" * V.N).to_matrix()
    U = np.diag([1j ** bin(i).count("1") for i in range(2**V.N)])
    assert np.allclose(H @ parity, parity @ H)
    assert np.allclose(H @ U, U @ H)


def test_hadamard_test_reads_n_mod_4():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    for w, expect in ((2, 1), (4, 0), (0, 0)):
        qc = QuantumCircuit(V.N + 1)
        for q in range(w):
            qc.x(q)
        qc.h(V.N)
        for q in range(V.N):
            qc.cp(math.pi / 2, V.N, q)
        qc.h(V.N)
        p = Statevector(qc).probabilities([V.N])
        assert p[expect] == pytest.approx(1.0)


def test_ideal_projections_ordering():
    r = V.exact_levels(1, "all_to_all")
    assert r["raw"] > r["parity"] > r["mod4"] >= r["number"] - 0.2
    assert r["number"] < 0.05 * r["raw"]
    assert r["kept_mod4"] == pytest.approx(r["kept_number"], abs=0.01)


def test_ancilla_checks_with_shots(monkeypatch):
    monkeypatch.setattr(V, "SHOTS", 60000)
    p = V.ancilla_check(1, mod4=False)
    m = V.ancilla_check(1, mod4=True)
    assert p["extra_cx"] == 6 and m["extra_cx"] == 18
    assert p["parity"] < 0.75 * p["raw"]
    assert m["parity"] < 0.25 * m["raw"]
    assert m["parity"] < p["parity"]
