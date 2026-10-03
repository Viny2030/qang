"""Tests for examples/qg_radius_witness_qg.py (§98): the filtered radius
deficit is the noiseless tangle under equal T1, and product states show no
false entanglement with qang."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np

import qg_radius_witness_qg as E  # noqa: E402


def test_filtered_deficit_exact_under_equal_t1():
    rows = E.run_config(4, 2, 4, "equal", np.random.default_rng(1), circuits=2, reps=3)
    assert [r["shots"] for r in rows] == list(E.SHOTS)
    for r in rows:
        assert r["exact max err qang"] < 1e-10
        assert r["exact err raw"] > 0.05


def test_product_family_no_false_entanglement():
    rng = np.random.default_rng(2)
    for kind in E.NOISES:
        p = E.product_family(4, 1, 8, kind, rng)
        assert p["max deficit qang"] == 0.0
    assert E.product_family(4, 1, 8, "equal", rng)["min raw deficit excited"] > 0.5


def test_unbiased_deficit_from_counts():
    zs = np.array([[1.0], [-1.0]])
    assert E.deficit_from_counts(np.array([5, 0]), zs)[0] == 0.0
    assert np.isnan(E.deficit_from_counts(np.array([1, 0]), zs)[0])
