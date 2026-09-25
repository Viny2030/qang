"""
Tests for examples/few_shot_tomography_qg.py: few-shot single-qubit
tomography, per-axis qg-Haar vs Jeffreys, LI, MLE and full Bayes.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("scipy")

import few_shot_tomography_qg as T  # noqa: E402


def test_ensembles_are_valid_bloch_vectors():
    rng = np.random.default_rng(0)
    for ens in T.ENSEMBLES:
        r = T.sample_states(ens, 500, rng)
        n = np.linalg.norm(r, axis=1)
        assert np.all(n <= 1 + 1e-12)
        if "pure" in ens:
            assert np.allclose(n, 1)
    assert np.all(np.linalg.norm(T.sample_states("strongly mixed", 500, rng), axis=1) <= 0.3 + 1e-12)


def test_closed_forms_per_axis():
    k = np.array([[10.0, 5.0, 0.0]])
    e = T.estimate_all(k, 10, T.sample_states("uniform mixed", 500, np.random.default_rng(1)))
    assert e["LI"][0] == pytest.approx([1.0, 0.0, -1.0])
    assert np.linalg.norm(e["qg-Haar"][0]) <= 1 + 1e-12
    assert np.linalg.norm(e["MLE"][0]) == pytest.approx(1.0, abs=1e-6)  # LI outside the ball -> MLE on the sphere


def test_qg_haar_wins_among_closed_forms_on_mixed_states():
    for ens in ("uniform mixed", "strongly mixed"):
        t = T.benchmark(ens, 10, reps=800, seed=2)
        assert t["qg-Haar"]["mse"] < t["Jeffreys"]["mse"] < t["LI+proj"]["mse"]
        assert t["qg-Haar"]["mse"] < t["MLE"]["mse"]
        assert t["Bayes"]["mse"] <= t["qg-Haar"]["mse"]


def test_qg_haar_loses_on_pure_states_and_li_is_unphysical():
    for ens in ("Haar pure", "pure near pole"):
        t = T.benchmark(ens, 10, reps=800, seed=3)
        assert t["MLE"]["mse"] < t["qg-Haar"]["mse"]
        assert t["LI"]["unphysical"] > 0.5
        assert t["Bayes"]["mse"] < t["MLE"]["mse"]


def test_mismatched_prior_is_expensive():
    m = T.mismatched_prior(reps=600)
    assert m["strongly mixed"] > 0.4 and m["pure near pole"] > 0.1
