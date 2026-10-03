"""Tests for examples/qg_filter_scaling_lowk_qg.py (§96)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np

import qg_filter_scaling_lowk_qg as H  # noqa: E402


def test_noise_kinds_and_small_run():
    H.CIRCUITS, H.REPS = 1, 2
    rng = np.random.default_rng(2)
    for kind in H.NOISES:
        rows = H.run_config(4, 2, 4, 0.05, kind, rng)
        assert len(rows) == len(H.SHOTS)
        assert all(0 < r["K"] <= 1 for r in rows)
    g, ph = H.noise_args("unequal", 0.05, 4, rng)
    assert np.isclose(np.mean(g), 0.05) and ph == 0.0
