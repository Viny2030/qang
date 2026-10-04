"""Tests for examples/qnn_distorting_encoding_qg.py (§101): quantities and verdict."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import qnn_distorting_encoding_qg as D  # noqa: E402


def test_quantities_and_verdict():
    row = {"P-zz filter": 0.90, "P-zz raw": 0.92, "P-zz both": 0.92, "P-z filter": 0.90, "P-z raw": 0.90,
           "D-zz filter": 0.95, "D-zz raw": 0.95}
    q = D.quantities([row])
    assert abs(q["interaction"] - 0.02) < 1e-12 and abs(q["P-zz both-raw"]) < 1e-12
    s = {k: (v, v - 0.001, v + 0.001) for k, v in q.items()}
    assert D.verdict(s) == {"D1": True, "D2": True, "D3": True, "D4": True, "D5": True}
