"""
Tests for examples/erasure_qubits_qg.py (RESEARCH_NOTES §73).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("pymatching")

import erasure_qubits_qg as X  # noqa: E402
import surface_code_circuit_t1_qg as SC  # noqa: E402


@pytest.mark.parametrize("logical", [0, 1])
def test_noiseless(logical):
    c = SC.Code(3)
    meas, final, her = X.simulate(c, 3, 300, np.random.default_rng(0), 0, 0, 0, 1.0, logical)
    det, lg = SC.detectors(c, meas, final)
    assert not det.any() and not any(her)
    assert set(lg.tolist()) == {bool(logical)}


def test_heralds_are_located_decays():
    c = SC.Code(3)
    tab = X.edge_table(c, 3)
    _, _, her = X.simulate(c, 3, 2000, np.random.default_rng(1), 0, 0.01, 0, 1.0, 1)
    flat = [kx for h in her for kx in h]
    assert flat and all(x < c.nd for _, x in flat)
    assert sum(kx in tab for kx in flat) / len(flat) > 0.95


def test_no_heralds_erasure_is_standard():
    r = X.run(0.0, d=3, shots=8000, seed=5, **X.REGIME)
    assert r["erasure"] == r["standard"] and r["heralds_per_shot"] == 0
    assert r["gain"] > 1.3


def test_naive_combination_hurts_calibrated_does_not():
    c = X.calibrated(0.99, d=3, shots=20000, seed=74)
    assert c["qg + erasure"] > 1.5 * c["erasure"]
    assert c["z_cal"] > -2
    assert c["erasure, calibrated"] <= c["erasure"]
