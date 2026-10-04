"""Tests for examples/qg_direction_radians_qg.py (§102)."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import qg_direction_radians_qg as A  # noqa: E402


def test_channels_and_exact_angles():
    q = A.bloch(1.0)
    a, b = A.angles_exact(A.channel(q, "depolarizing", 0.3))
    assert abs(a - 1.0) < 1e-12 and abs(b - 1.0) > 0.01
    a, b = A.angles_exact(A.channel(q, "dephasing", 0.7))
    assert abs(b - 1.0) < 1e-12 and abs(a - 1.0) > 0.01


def test_small_sim_run_has_all_fields():
    rows = A.part_sim(reps=2, grid=np.array([0.5, 2.5]))
    assert len(rows) == len(A.CHANNELS) + 1
    assert set(A.verdict_sim(rows)) == {"A1", "A2", "A3", "A4", "A5"}
    assert all(r["exact error with qang"] < math.pi for r in rows)
