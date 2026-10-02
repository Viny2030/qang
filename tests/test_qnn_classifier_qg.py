"""
Tests for examples/qnn_classifier_qg.py (RESEARCH_NOTES §75).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

import qnn_classifier_qg as Q  # noqa: E402


def test_encodings():
    rng = np.random.default_rng(1)
    Xs = rng.uniform(-1, 1, (6, 4))
    pB = np.abs(Q.encode("B", Xs)) ** 2
    assert np.allclose(pB @ Q.ZSIGN, Xs)  # arccos encoding: qg_Z of qubit i = x_i
    pD = np.abs(Q.encode("D", Xs)) ** 2
    assert np.allclose(pD.sum(axis=1), 1) and np.allclose(pD[:, Q.WEIGHT != 1].sum(axis=1), 0)


@pytest.mark.parametrize("model", ["A", "D"])
def test_unitary_and_zero_noise(model):
    rng = np.random.default_rng(2)
    th = rng.uniform(-3, 3, Q.n_params(model))
    for L in Q.sublayers(model, th):
        assert np.allclose(L.conj().T @ L, np.eye(Q.DIM))
    psi0 = Q.encode(model, rng.uniform(-1, 1, (3, 4)))
    U = np.eye(Q.DIM)
    for L in Q.sublayers(model, th):
        U = L @ U
    assert np.allclose(Q.probs_noisy(model, th, psi0, "T1", 0.0), np.abs(psi0 @ U.T) ** 2)
    assert np.allclose(Q.probs_noisy(model, th, psi0, "depol", 0.1).sum(axis=1), 1)


def test_filter_undoes_t1_leak_for_weight_conserving_model():
    rng = np.random.default_rng(3)
    th = rng.uniform(-3, 3, Q.n_params("D"))
    psi0 = Q.encode("D", rng.uniform(-1, 1, (4, 4)))
    U = np.eye(Q.DIM)
    for L in Q.sublayers("D", th):
        U = L @ U
    exact = Q.readout_from_probs("D", np.abs(psi0 @ U.T) ** 2)
    noisy = Q.probs_noisy("D", th, psi0, "T1", 0.05)
    assert noisy[:, Q.WEIGHT > 1].sum() < 1e-12  # T1 never raises the weight
    r_f = Q.readout_from_probs("D", noisy, filt=True)
    r_n = Q.readout_from_probs("D", noisy)
    assert np.mean(np.abs(r_f - exact)) < np.mean(np.abs(r_n - exact))


def test_small_training_run():
    pytest.importorskip("sklearn")
    res, _ = Q.run_dataset("iris", seed=75, splits=1, epochs=25)
    assert res["C"] > 0.8 and res["logistic"] > 0.8
    assert 0 <= res["D T1 filter"] <= 1
