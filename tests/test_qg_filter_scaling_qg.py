"""Tests for examples/qg_filter_scaling_qg.py (§94)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np

import qg_filter_scaling_qg as G  # noqa: E402


def test_small_configuration_keeps_known_fraction_and_rule_fields():
    G.CIRCUITS, G.REPS = 2, 3
    rows, kept_err = G.run_config(6, 3, 2, np.random.default_rng(1))
    assert kept_err < 1e-12
    assert [r["shots"] for r in rows] == list(G.SHOTS)
    for r in rows:
        assert r["K"] == (1 - G.GAMMA) ** (3 * r["depth"])
        assert r["pred qang"] > 0 and r["pred raw"] > 0
