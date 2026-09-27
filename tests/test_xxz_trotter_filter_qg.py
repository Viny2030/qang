"""
Tests for examples/xxz_trotter_filter_qg.py (RESEARCH_NOTES §60): XXZ
Trotter dynamics with the qg number filter, per-noise reach, and the
CNOT vs number-conserving compilation.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit_aer")
import scipy.linalg as sl
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from xxz_trotter_filter_qg import (
    bond,
    bond_native,
    idle_decay_check,
    imbalance,
    noise_decomposition,
    probabilities,
    study,
)

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1])


@pytest.mark.parametrize("gate", [bond, bond_native])
@pytest.mark.parametrize("fold", [1, 3])
def test_bond_is_exact(gate, fold):
    a, b = 0.3, 0.2
    qc = QuantumCircuit(2)
    gate(qc, 0, 1, a, b, fold)
    target = sl.expm(-1j * (a * (np.kron(X, X) + np.kron(Y, Y)) + b * np.kron(Z, Z)))
    assert abs(np.vdot(target.flatten(), Operator(qc).data.flatten())) / 4 == pytest.approx(1, abs=1e-12)


def test_noiseless_conserves_n_and_starts_at_one():
    p = probabilities(6, 3, 0.25)
    _, kept = imbalance(p, 6, True)
    assert kept == pytest.approx(1, abs=1e-12)
    assert imbalance(probabilities(6, 0, 0.25), 6)[0] == pytest.approx(1.0)


def test_decay_between_steps_fully_filtered():
    er, ef = idle_decay_check()
    assert er > 0.01
    assert ef < 1e-12


def test_noise_decomposition():
    _, rows = noise_decomposition()
    d = {(name.split()[0] + name.split()[1], comp): (er, ef) for name, comp, er, ef, _ in rows}
    removed = {k: 1 - ef / er for k, (er, ef) in d.items()}
    assert removed[("2qdepolarizing", "3 CNOT")] == pytest.approx(0.411, abs=0.01)
    assert removed[("amplitudedamping", "3 CNOT")] == pytest.approx(0.485, abs=0.01)
    assert removed[("amplitudedamping", "N-conserving")] > 0.99
    assert removed[("readoute", "3 CNOT")] > 0.9
    assert removed[("Zdephasing", "N-conserving")] < 0.01


def test_compilation_changes_filter_reach():
    kw = dict(n=6, steps_list=(1, 4), p2=0.002, gamma=0.02, e=0.01)
    cnot = study(**kw)
    nat = study(native=True, **kw)
    for c, m in zip(cnot, nat):
        assert abs(m["filt"] - m["ideal"]) < 0.5 * abs(c["filt"] - c["ideal"])
        assert m["kept"] < c["kept"]
    first = study(n=6, steps_list=(1,), p2=0.01, gamma=0.005, e=0.01)[0]
    assert abs(first["fzne"] - first["ideal"]) < abs(first["zne"] - first["ideal"]) / 5
