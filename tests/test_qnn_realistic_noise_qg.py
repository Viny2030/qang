"""
Tests for examples/qnn_realistic_noise_qg.py (RESEARCH_NOTES §78): the exact
simulators with per-qubit T1 and dephasing, F4 (the filter is exact in every
fixed-weight sector under equal T1) and its failure under unequal T1.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("sklearn")

import qnn_classifier_qg as Q  # noqa: E402
import qnn_realistic_noise_qg as R  # noqa: E402
import qnn_unary_norm_qg as E  # noqa: E402


@pytest.mark.parametrize("cond", ["H", "D"])
def test_block_simulator_equals_full_density_matrix(cond):
    rng = np.random.default_rng(1)
    th = rng.uniform(-3, 3, E.NP)
    psi = E.encode(rng.uniform(-1, 1, (5, 4)))
    gam, phi = R.CONDITIONS[cond]["E"]
    assert np.allclose(R.probs_E_block(th, psi, gam, phi), R.probs_E_full(th, psi, gam, phi), atol=1e-12)


def test_per_qubit_A_matches_uniform_reference():
    rng = np.random.default_rng(2)
    th = rng.uniform(-3, 3, Q.n_params("A"))
    psi = Q.encode("A", rng.uniform(-1, 1, (4, 4)))
    ref = Q.probs_noisy("A", th, psi, "T1", 0.08)
    assert np.allclose(R.probs_A(th, psi, np.full(Q.N, 0.08), 0.0), ref, atol=1e-12)


def test_F4_filter_exact_in_every_weight_sector():
    for err, kerr in R.check_F4(gamma=0.15, seed=3).values():
        assert err < 1e-12 and kerr < 1e-12


def test_unequal_t1_breaks_exactness():
    rng = np.random.default_rng(4)
    th = rng.uniform(-3, 3, E.NP)
    psi = E.encode(rng.uniform(-1, 1, (4, 4)))
    exact = E.local_z(E.probs_exact(th, psi))
    noisy = R.probs_E_block(th, psi, R.gammas(E.N), 0.0)
    assert np.abs(E.local_z(noisy, filt=True) - exact).max() > 1e-3


def test_dephasing_keeps_all_weight_one_population():
    rng = np.random.default_rng(5)
    th = rng.uniform(-3, 3, E.NP)
    psi = E.encode(rng.uniform(-1, 1, (3, 4)))
    pr = R.probs_E_block(th, psi, np.zeros(E.N), 0.2)
    assert np.allclose(pr[:, E.WEIGHT == 1].sum(axis=1), 1)


def test_small_run_keys():
    out, _ = R.run_dataset("iris", seed=1, splits=1, epochs=2)
    v = R.verdict({"iris": out})
    assert set(v) == {"S1", "S2", "S3", "S4", "S5"} and v["S1"]
    assert 0 < out["H kept"] < 1
