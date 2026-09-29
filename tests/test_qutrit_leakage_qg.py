"""
Tests for examples/qutrit_leakage_qg.py (RESEARCH_NOTES §70): the qutrit
(leakage) test of the §64 decoders.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("pymatching")

import qutrit_leakage_qg as Q  # noqa: E402
import surface_code_circuit_t1_qg as SC  # noqa: E402


@pytest.mark.parametrize("logical", [0, 1])
def test_noiseless_no_leak(logical):
    c = SC.Code(3)
    meas, final, flag = Q.simulate_leak(c, 3, 300, np.random.default_rng(0), 0, 0, 0, 0, 0, logical)
    det, lg = SC.detectors(c, meas, final)
    assert not det.any() and not flag.any()
    assert set(lg.tolist()) == {bool(logical)}


def test_leaked_qubits_read_as_one():
    c = SC.Code(3)
    _, final, flag = Q.simulate_leak(c, 3, 4000, np.random.default_rng(1), 0, 0, 0, 0.05, 0.0, 1)
    assert flag.any()
    assert final[flag].all()  # |2> is read as 1 without readout noise


def test_window_edges_are_subsets():
    c = SC.Code(3)
    full = Q.qubit_edges(c, 3)
    last = Q.qubit_edges(c, 3, window=1)
    lo = 3 * c.na
    for x in range(c.nd):
        assert last[x] <= full[x] and last[x]
        assert all(max(e) >= lo or e[0] >= lo for e in last[x])


def test_flags_do_not_help_t1_reweighting_does():
    r = Q.run(shots=12000, seed=3, **dict(Q.CASES)["leakage + T1-dominated"])
    assert r["witness"] > 1
    assert r["standard"] / r["qg, no leakage flags"] > 1.3 and r["z_qgnoflag_vs_std"] > 3
    assert r["leakage-aware"] >= r["standard"]
    assert r["z_qglast_vs_qgnoflag"] <= 0
