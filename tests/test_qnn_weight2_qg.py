"""
Tests for examples/qnn_weight2_qg.py (RESEARCH_NOTES §79): the weight-2
encoding, the exact block simulator (any weight, per-qubit T1, dephasing),
F4 at weight 2 with qang and its absence without qang.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("sklearn")

import qnn_realistic_noise_qg as R  # noqa: E402
import qnn_unary_norm_qg as E  # noqa: E402
import qnn_weight2_qg as W  # noqa: E402


def test_weight2_encoding_is_normalized_and_in_sector():
    psi = W.encode_w2(np.random.default_rng(1).uniform(-1, 1, (5, 4)))
    assert np.allclose(np.linalg.norm(psi, axis=1), 1)
    assert np.allclose((psi**2)[:, W.WEIGHT != 2].sum(axis=1), 0)


@pytest.mark.parametrize("model", ["E", "W"])
@pytest.mark.parametrize("cond", ["T1", "H", "D"])
def test_block_simulator_equals_full_density_matrix(model, cond):
    rng = np.random.default_rng(2)
    th = rng.uniform(-3, 3, E.NP)
    psi = W.encode(model, rng.uniform(-1, 1, (4, 4)))
    got = W.probs_block(th, psi, W.MODELS[model], *W.CONDITIONS[cond])
    assert np.allclose(got, R.probs_E_full(th, psi, *W.CONDITIONS[cond]), atol=1e-12)


def test_noiseless_block_equals_exact():
    rng = np.random.default_rng(3)
    th = rng.uniform(-3, 3, E.NP)
    psi = W.encode_w2(rng.uniform(-1, 1, (4, 4)))
    assert np.allclose(W.probs_block(th, psi, 2), E.probs_exact(th, psi), atol=1e-12)


def test_F4_weight2_with_and_without_qang():
    rng = np.random.default_rng(4)
    th = rng.uniform(-3, 3, E.NP)
    psi = W.encode_w2(rng.uniform(-1, 1, (4, 4)))
    exact = W.local_z(E.probs_exact(th, psi), 2)
    noisy = W.probs_block(th, psi, 2, *W.CONDITIONS["T1"])
    assert np.allclose(W.local_z(noisy, 2, qang=True), exact, atol=1e-12)  # with qang: exact
    assert np.abs(W.local_z(noisy, 2) - exact).max() > 0.05  # without qang: biased
    assert np.allclose(noisy[:, W.IDX[2]].sum(axis=1), (1 - W.GAMMA) ** 18)


def test_small_run_keys():
    out, _ = W.run_dataset("iris", seed=1, splits=1, epochs=2)
    v = W.verdict({"iris": out})
    assert set(v) == {"P1", "P2", "P3", "P4", "P5"} and v["P1"]
    assert "diff" in W.table({"iris": out})
