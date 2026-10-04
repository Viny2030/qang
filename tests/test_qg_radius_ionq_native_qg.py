"""Tests for examples/qg_radius_ionq_native_qg.py (§104): verdict only, no submission."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_verdict():
    pytest.importorskip("sklearn")
    import qg_radius_ionq_native_qg as Q

    r = {"trained: deficit error with qang": 0.08, "trained: deficit error without qang": 0.25,
         "echo: false deficit of the excited qubit, with qang": [0.5] * 5,
         "echo: false deficit of the excited qubit, without qang": [0.75] * 5}
    assert Q.verdict({"aria-1": r, "forte-1": r}) == {"Q1": True, "Q2": True, "Q3": True, "Q4": True}
