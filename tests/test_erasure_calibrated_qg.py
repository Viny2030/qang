"""
Tests for examples/erasure_calibrated_qg.py (RESEARCH_NOTES §74).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import pytest

pytest.importorskip("pymatching")

import erasure_calibrated_qg as C  # noqa: E402


def test_h_zero_calibration_is_identity():
    r = C.run(0.0, d=3, shots=6000, seed=3, **C.REGIMES["T1-dominated"])
    assert r["erasure"] == r["erasure, calibrated"]
    assert r["gain"] > 1.3


def test_rule_gains_at_half_heralding():
    r = C.run(0.5, d=3, shots=20000, seed=740, **C.REGIMES["T1-dominated"])
    assert r["gain"] > 1.25 and r["z_gain"] > 3


def test_rule_does_not_hurt_at_high_heralding():
    r = C.run(0.99, d=3, shots=20000, seed=740, **C.REGIMES["T1-dominated"])
    assert r["z_gain"] > -2.5
    assert r["erasure, calibrated"] < r["erasure"]


def test_misestimated_h():
    r = C.run(0.5, d=3, h_dec=0.6, shots=20000, seed=741, **C.REGIMES["T1-dominated"])
    assert r["gain"] > 1.2
