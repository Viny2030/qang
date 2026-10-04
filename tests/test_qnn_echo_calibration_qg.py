"""Tests for examples/qnn_echo_calibration_qg.py (§106): verdict only."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_verdict():
    pytest.importorskip("sklearn")
    import qnn_echo_calibration_qg as C

    p = {"without qang": {"accuracy": 0.95, "agree": 180, "decision error": 0.3},
         "qang": {"accuracy": 0.96, "agree": 184, "decision error": 0.1},
         "qang + echo": {"accuracy": 0.96, "agree": 185, "decision error": 0.06}}
    assert C.verdict([{"pooled": p}]) == {"L1": True, "L2": True, "L3": True, "L4": True}
