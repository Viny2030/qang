"""
Tests for examples/lattice_gauge_gauss_qg.py (RESEARCH_NOTES §66): Gauss-law
witnesses and filters in a Z2 lattice gauge theory.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit_aer")
pytest.importorskip("scipy")

import lattice_gauge_gauss_qg as G  # noqa: E402


def test_symmetries_exact():
    cg, cn = G.check_symmetries()
    assert cg < 1e-12 and cn < 1e-12


def test_noiseless_circuit_stays_physical():
    p = G.probabilities(2)
    est = G.estimates(p)
    assert est["kept_Gauss+N"] == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(np.abs(est["witness_G"]), 1.0, atol=1e-9)


def test_explicit_hopping_gate_is_exact():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    qc = QuantumCircuit(3)
    G.hop(qc, 0, 1, 2, 0.3)
    U = G.hopping_unitary(0.3)
    assert abs(np.vdot(U.flatten(), Operator(qc).data.flatten())) / 8 == pytest.approx(1.0, abs=1e-12)


def test_gauss_beats_number_filter_over_the_run():
    for noise in (dict(p2=0.01), dict(gamma=0.01), dict(p2=0.005, gamma=0.002, e=0.01)):
        t = G.aggregate(**noise)
        assert t["Gauss+N"] <= t["Gauss"] < t["N"] < t["raw"]
    t = G.aggregate(gamma=0.01)
    assert 1 - t["Gauss+N"] / t["raw"] < 0.4  # T1 leaks through the gates
