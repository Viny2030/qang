"""Tests for examples/qnn_two_channel_qg.py (§95): quantities and verdict."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import qnn_two_channel_qg as B  # noqa: E402


def test_quantities_and_verdict():
    row = {f"{m} {r}": v for m in B.MODELS for r, v in (("filter", 0.95), ("raw", 0.94), ("both", 0.96))}
    q = B.quantities([row, row])
    assert abs(q["W both-raw"] - 0.02) < 1e-12 and abs(q["E both-best"] - 0.01) < 1e-12
    s = {k: (v, v - 0.001, v + 0.001) for k, v in q.items()}
    assert B.verdict(s) == {"B1": True, "B2": True, "B3": False, "B4": True}
