"""
Tests for examples/qutrit_bayes_weight_qg.py (RESEARCH_NOTES §71).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("pymatching")

import qutrit_bayes_weight_qg as B  # noqa: E402
import surface_code_circuit_t1_qg as SC  # noqa: E402


def test_p_flag_limits():
    assert B.p_flag(0.0, 0.002) == 0.002
    assert B.p_flag(1.0, 0.002) == 0.5
    assert abs(B.p_flag(0.5, 0.002) - 1 / 3) < 1e-12


@pytest.mark.parametrize("model", ["M0", "M(0.5)"])
def test_noiseless(model):
    a, ret, read2 = B.MODELS[model]
    c = SC.Code(3)
    meas, final, flag = B.simulate(c, 3, 300, np.random.default_rng(0), 0, 0, 0, 0, 0, a, ret, read2, 1)
    det, lg = SC.detectors(c, meas, final)
    assert not det.any() and not flag.any() and lg.all()


def test_leak_from_zero_scales_with_a():
    c = SC.Code(3)
    rates = []
    for a in (0.0, 1.0):
        _, _, flag = B.simulate(c, 3, 4000, np.random.default_rng(1), 0, 0, 0, 0.01, 0, a, "random", "random", 0)
        rates.append(flag.mean())
    assert 0 < rates[0] < rates[1]  # leaking also from |0> adds leaks


def test_bayes_equals_erasure_at_a_one_and_never_hurts_in_M0():
    r1 = B.run(B.MODELS["M(1)"], d=3, shots=8000, seed=5, **B.REGIMES["T1-dominated"])
    assert r1["bayes"] == r1["erasure"] and r1["qg + bayes"] == r1["qg + erasure"]
    r0 = B.run(B.MODELS["M0"], d=3, shots=20000, seed=711, **B.REGIMES["T1-dominated"])
    assert r0["z_bayes_vs_std"] > -1 and r0["z_erasure_vs_std"] < -2
    assert r0["ratio_P4"] > 1.4
