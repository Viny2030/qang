"""
Tests for examples/surface_code_circuit_t1_qg.py (RESEARCH_NOTES §64):
circuit-level Z-memory with T1; the T1-aware decoder and the qg switch.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("pymatching")

import surface_code_circuit_t1_qg as S  # noqa: E402


@pytest.mark.parametrize("d", [3, 5])
def test_layout(d):
    c = S.Code(d)
    assert c.na == (d * d - 1) // 2 == len(c.xstabs)
    for zs in c.zstabs:
        for xs in c.xstabs:
            assert len(set(zs.values()) & set(xs.values())) % 2 == 0
    for layer in c.layers:  # no qubit used twice in a layer
        used = [q for pair in layer for q in pair]
        assert len(used) == len(set(used))


@pytest.mark.parametrize("logical", [0, 1])
def test_noiseless_circuit(logical):
    c = S.Code(3)
    meas, final = S.simulate(c, 3, 500, np.random.default_rng(0), noise=False, logical=logical)
    det, lg = S.detectors(c, meas, final)
    assert not det.any()
    assert set(lg.tolist()) == {bool(logical)}


def test_t1_aware_gain_and_switch():
    r = S.run(3, 3, shots=30000, p2=0.0, gamma=0.003, q=0.001)
    assert r["standard"] / r["t1_aware"] > 1.7
    assert r["mcnemar_z"] > 8
    assert S.witness_ratio(3, 3, 0.0, 0.003, 0.001) > 1
    assert S.witness_ratio(3, 3, 0.004, 0.0005, 0.002) < 1


def test_decoder_scales_quadratically():
    a = S.run(3, 3, shots=30000, p2=0.0005, gamma=0.003, q=0.002)["standard"]
    b = S.run(3, 3, shots=30000, p2=0.00025, gamma=0.0015, q=0.001)["standard"]
    assert 2.5 < a / b < 6
