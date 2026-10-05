"""Tests for examples/qnn_trainability_qg.py (§112)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import qnn_trainability_qg as T  # noqa: E402


def test_small_cell_filter_is_exact_and_weight1_ratio_is_K2():
    r = T.cell(4, 1, 2, np.random.default_rng(0), draws=8)
    assert r["max |qang - noiseless|"] < 1e-8
    assert abs(r["ratio raw"] / r["K"] ** 2 - 1) < 1e-5
    assert r["median shots qang"] > 0 and r["median shots raw"] > 0
