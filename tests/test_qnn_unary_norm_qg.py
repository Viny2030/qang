"""
Tests for examples/qnn_unary_norm_qg.py (RESEARCH_NOTES §76): the T1
exactness of the qg filter (F1) and the classical simulability of the
weight-conserving QNN (F2).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

import qnn_unary_norm_qg as E  # noqa: E402


def test_encoding_keeps_norm_in_weight_one():
    rng = np.random.default_rng(1)
    Xs = rng.uniform(-1, 1, (5, 4))
    psi = E.encode(Xs)
    assert np.allclose(np.linalg.norm(psi, axis=1), 1)
    assert np.allclose((psi**2)[:, E.WEIGHT != 1].sum(axis=1), 0)
    # the norm of x is recoverable from the constant component
    v = E.augmented(Xs)
    assert np.allclose(np.linalg.norm(v[:, :4] / v[:, 4:5], axis=1), np.linalg.norm(Xs, axis=1))


@pytest.mark.parametrize("gamma", [0.03, 0.2, 0.5])
def test_F1_filter_makes_t1_exact(gamma):
    rng = np.random.default_rng(2)
    th = rng.uniform(-3, 3, E.NP)
    psi = E.encode(rng.uniform(-1, 1, (4, 4)))
    exact = E.local_z(E.probs_exact(th, psi))
    noisy = E.probs_noisy(th, psi, "T1", gamma)
    assert np.allclose(E.local_z(noisy, filt=True), exact, atol=1e-10)
    assert not np.allclose(E.local_z(noisy), exact)


def test_F2_qnn_is_a_classical_quadratic_form():
    rng = np.random.default_rng(3)
    params = np.concatenate([rng.uniform(-3, 3, E.NP), rng.normal(0, 1, E.N), [0.3]])
    Xs = rng.uniform(-1, 1, (6, 4))
    z_q = E.local_z(E.probs_exact(params[:E.NP], E.encode(Xs))) @ params[E.NP:E.NP + E.N] + params[-1]
    U = np.eye(E.DIM)
    for L in E.sublayers(params[:E.NP]):
        U = L @ U
    idx = [1 << (E.N - 1 - i) for i in range(E.N)]  # weight-1 basis states
    O = U[np.ix_(idx, idx)]
    assert np.allclose(O.T @ O, np.eye(E.N))  # the network is an orthogonal matrix on that sector
    c = params[E.NP:E.NP + E.N]
    v = E.augmented(Xs)
    z_c = c.sum() - 2 * np.einsum("si,ij,sj->s", v, O.T @ np.diag(c) @ O, v) + params[-1]
    assert np.allclose(z_q, z_c)


def test_small_training_run():
    pytest.importorskip("sklearn")
    res, _ = E.run_dataset("iris", seed=76, splits=1, epochs=25)
    assert res["E"] > 0.8
    assert res["E T1 filter"] == res["E"]
