"""Tests for examples/qnn_multiclass_echo_qg.py (§113): verdict only."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_verdict():
    pytest.importorskip("sklearn")
    import qnn_multiclass_echo_qg as E

    q = {"without qang": {"accuracy": 0.7, "differ": 20}, "qang": {"accuracy": 0.8, "differ": 10},
         "qang + echo": {"accuracy": 0.82, "differ": 6}}
    h = {"without qang": {"accuracy": 0.7, "differ": 20}, "qang": {"accuracy": 0.85, "differ": 8},
         "qang + echo": {"accuracy": 0.86, "differ": 6}}
    assert E.verdict([{"qubit": q, "head": h}]) == {"E1": True, "E2": True, "E3": True, "E4": True}
