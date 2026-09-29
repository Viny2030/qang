"""
Tests for examples/qutrit_leakage_rounds_qg.py (RESEARCH_NOTES §70b).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("pymatching")

import qutrit_leakage_rounds_qg as R  # noqa: E402
import surface_code_circuit_t1_qg as SC  # noqa: E402


@pytest.mark.parametrize("logical", [0, 1])
def test_noiseless(logical):
    c = SC.Code(3)
    meas, final, rflag, fflag = R.simulate(c, 3, 300, np.random.default_rng(0), 0, 0, 0, 0, 0, 0.8, logical)
    det, lg = SC.detectors(c, meas, final)
    assert not det.any() and not rflag.any() and not fflag.any()
    assert set(lg.tolist()) == {bool(logical)}


def test_leaks_from_both_levels_and_flags_only_leaked():
    c = SC.Code(3)
    rng = np.random.default_rng(2)
    _, _, rflag0, f0 = R.simulate(c, 3, 3000, rng, 0, 0, 0, 0.02, 0.0, 1.0, 0)
    assert f0.any()  # the all-|0> codeword still leaks (from |0>)
    # with h = 1 and no return, a qubit flagged in a round is still leaked at the end
    assert not (rflag0.any(axis=1) & ~f0).any()


def test_erased_edges_windows():
    c = SC.Code(3)
    per = R.erasure_sets(c, 3)
    none = R.erased_edges(per[0], 3, [0, 0, 0], False, True)
    fin = R.erased_edges(per[0], 3, [0, 0, 0], True, False)
    rnd = R.erased_edges(per[0], 3, [1, 0, 0], True, True)
    assert not none and fin and fin <= rnd


def test_flags_and_reweighting_combine():
    r = R.run(shots=20000, seed=70, **dict(R.CASES)["leakage + T1-dominated"])
    assert r["witness"] > 1
    assert r["qg + round flags"] < r["qg, no flags"] < r["standard"]
    assert r["ratio_A"] > 1.2 and r["z_A"] > 3
    assert r["z_B"] > 2
