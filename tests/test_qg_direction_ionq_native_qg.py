"""Tests for examples/qg_direction_ionq_native_qg.py (§103): circuits and verdict, no submission."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_block_switch_and_verdict():
    pytest.importorskip("qiskit")
    pytest.importorskip("sklearn")
    import qg_direction_ionq_native_qg as N

    none = N.circuits("qis-none", 0.7)[0]
    block = N.circuits("qis-block", 0.7)[0]
    assert sum(1 for d in none.data if d.operation.name == "cx") == 0
    assert sum(1 for d in block.data if d.operation.name == "cx") == 16
    r = {"error with qang": 0.03, "error without qang": 0.2}
    res = {"m": {"qis-block": {"error without qang": 0.012}, "qis-none": {"error without qang": 0.010}, "native-block": r}}
    assert N.verdict(res) == {"N1": True, "N2": True, "N3": True}
