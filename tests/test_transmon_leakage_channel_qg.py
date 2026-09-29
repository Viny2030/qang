"""
Tests for examples/transmon_leakage_channel_qg.py (RESEARCH_NOTES §72).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pymatching")

import surface_code_circuit_t1_qg as SC  # noqa: E402
import transmon_leakage_channel_qg as T  # noqa: E402


@pytest.fixture(scope="module")
def cz():
    return T.best_cz()


def test_unitary_and_cz(cz):
    hold, r = cz
    U = T.cz_unitary(hold)
    assert np.allclose(U.conj().T @ U, np.eye(9), atol=1e-9)
    assert r["fidelity"] > 0.999
    assert abs(r["cphase"] - math.pi) < 0.1


def test_leaks_only_from_11(cz):
    _, r = cz
    for inp in ((0, 0), (0, 1), (1, 0)):
        assert r["leak"][inp] < 1e-12
    assert r["leak"][(1, 1)] > 0
    assert r["a_eff"] == 0.0


def test_leaked_qubit_kicks_like_one(cz):
    _, r = cz
    assert 0.9 < r["kappa"] <= 1.0


@pytest.mark.parametrize("logical", [0, 1])
def test_sim_noiseless(logical):
    c = SC.Code(3)
    sim = T.make_sim(0, 0, 0, 0, 0.0, 0.97)
    meas, final, flag = sim(c, 3, 300, np.random.default_rng(0), logical)
    det, lg = SC.detectors(c, meas, final)
    assert not det.any() and not flag.any()
    assert set(lg.tolist()) == {bool(logical)}


def test_flag_useless_erasure_hurts_qg_gains(cz):
    _, r = cz
    s = T.surface(r["a_eff"], r["kappa"], "T1-dominated", 3, shots=20000, seed=72)
    assert abs(s["z_bayes_vs_std"]) < 3
    assert s["z_erasure_vs_std"] < -2
    assert s["standard"] / s["qg"] > 1.5 and s["z_qg_vs_std"] > 3
