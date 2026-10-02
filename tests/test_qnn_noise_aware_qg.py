"""
Tests for examples/qnn_noise_aware_qg.py (RESEARCH_NOTES §77): the exact
block simulator for T1 and F3 (training under T1 with the qg filter gives
exactly the parameters of noiseless training).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("sklearn")

import qnn_noise_aware_qg as N  # noqa: E402
import qnn_unary_norm_qg as E  # noqa: E402


@pytest.mark.parametrize("gamma", [0.0, 0.08, 0.4])
def test_block_simulator_equals_density_matrix(gamma):
    rng = np.random.default_rng(5)
    th = rng.uniform(-3, 3, E.NP)
    psi = E.encode(rng.uniform(-1, 1, (6, 4)))
    full = E.probs_noisy(th, psi, "T1", gamma)
    assert np.allclose(N.probs_t1(th, psi, gamma), full, atol=1e-12)


def test_block_simulator_keeps_trace():
    rng = np.random.default_rng(6)
    psi = E.encode(rng.uniform(-1, 1, (3, 4)))
    pr = N.probs_t1(rng.uniform(-3, 3, E.NP), psi, 0.3)
    assert np.allclose(pr.sum(axis=1), 1)


def test_F3_training_under_t1_with_filter_equals_noiseless_training():
    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, (12, 4))
    y = (X[:, 0] > 0).astype(int)
    psi = E.encode(X)
    clean = N.adam_train(E.NP, E.N, N.readout_E(psi), y, np.random.default_rng(1), epochs=8)
    noisy = N.adam_train(E.NP, E.N, N.readout_E(psi, ("T1", 0.2), filt=True), y, np.random.default_rng(1), epochs=8)
    assert np.max(np.abs(clean - noisy)) < 1e-8
    # without the filter, training under T1 moves the parameters
    raw = N.adam_train(E.NP, E.N, N.readout_E(psi, ("T1", 0.2)), y, np.random.default_rng(1), epochs=8)
    assert np.max(np.abs(clean - raw)) > 1e-4


def test_small_run_and_verdict_keys():
    out, rows = N.run_dataset("iris", seed=1, splits=1, epochs=2)
    assert out["max dparam F3"] < 1e-8
    assert out["E filter"] == out["E noisy+f"]
    assert 0 < out["kept"] < 1
    v = N.verdict({"iris": out})
    assert set(v) == {"R1", "R2", "R3", "R4"} and v["R1"]


@pytest.mark.parametrize("gamma", [0.08, 0.3])
def test_kept_fraction_is_one_minus_gamma_to_the_depth(gamma):
    rng = np.random.default_rng(8)
    psi = E.encode(rng.uniform(-1, 1, (5, 4)))
    pr = N.probs_t1(rng.uniform(-3, 3, E.NP), psi, gamma)
    depth = len(E.sublayers(np.zeros(E.NP)))
    assert np.allclose(pr[:, E.WEIGHT == 1].sum(axis=1), (1 - gamma) ** depth)
