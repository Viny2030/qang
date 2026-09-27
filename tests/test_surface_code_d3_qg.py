"""
Tests for examples/surface_code_d3_qg.py (RESEARCH_NOTES §59): the d = 3
rotated surface code read in qg -- code capacity, two-weight estimate of p
and readout gain, T1 witness and T1-aware decoding with its limit.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

from surface_code_d3_qg import (
    N,
    POP,
    X_STABS,
    Z_STABS,
    ZL,
    XL,
    codewords,
    estimate_p_b,
    logical_error_bitflip,
    pseudo_threshold,
    readout_dist,
    sample_estimate,
    syndrome,
    syndrome_qg,
    t1_density_matrix_check,
    t1_study,
)


def test_code_structure():
    # X and Z stabilizers commute, logicals anticommute with each other only
    for zs in Z_STABS:
        for xs in X_STABS:
            assert len(set(zs) & set(xs)) % 2 == 0
    assert POP[ZL & XL] % 2 == 1
    assert len(codewords(0)) == 16
    assert all(syndrome(c) == (0, 0, 0, 0) for c in codewords(0) + codewords(1))


def test_code_capacity():
    assert logical_error_bitflip(0.01) == pytest.approx(1.731e-3, rel=1e-3)
    assert logical_error_bitflip(1e-4) / 1e-8 == pytest.approx(18, rel=0.01)
    assert pseudo_threshold() == pytest.approx(0.0753, abs=5e-4)


@pytest.mark.parametrize("p,q", [(0.01, 0.02), (0.03, 0.05)])
def test_two_weight_estimate(p, q):
    sq = syndrome_qg(p, q)
    ph, bh = estimate_p_b(sq[4], sq[2])
    assert ph == pytest.approx(p, abs=1e-12)
    assert bh == pytest.approx(1 - 2 * q, abs=1e-12)
    naive = (1 - sq[4] ** 0.25) / 2
    assert naive > 1.3 * p  # one weight alone absorbs the readout error


def test_two_weight_finite_shots():
    ps, bs = sample_estimate(0.03, 0.02, 2000, 60, np.random.default_rng(1))
    assert abs(ps.mean() - 0.03) < 3 * ps.std() / np.sqrt(len(ps)) + 1e-3
    assert abs(bs.mean() - 0.96) < 0.01


def test_t1_density_matrix_matches_classical_model():
    zs, pops = t1_density_matrix_check(0.2)
    assert np.allclose(zs, 0.2, atol=1e-12)
    dist = readout_dist(0.2)
    assert max(abs(pops[i] - dist.get(i, 0.0)) for i in range(1 << N)) < 1e-12


@pytest.mark.parametrize("g", [0.01, 0.1])
def test_t1_witness_and_decoders(g):
    r = t1_study(g)
    assert r["mean_qg_z"] == pytest.approx(g, abs=1e-12)
    # syndromes: (1-g)^w + g^w, the twirl gives (1-g)^w -- a g^w difference
    assert r["stab_qg"][Z_STABS[0]] == pytest.approx((1 - g) ** 4 + g**4, abs=1e-12)
    assert r["stab_qg"][Z_STABS[2]] == pytest.approx((1 - g) ** 2 + g**2, abs=1e-12)
    assert r["pL_mw"] > r["pL_twirl"]
    assert r["pL_mw"] / r["pL_t1"] > 2.1
    assert r["pL_ml"] == pytest.approx(r["pL_t1"], rel=1e-3)


def test_hard_t1_rule_breaks_soft_ml_does_not():
    r = t1_study(0.03, 0.003)
    assert r["gamma_hat"] == pytest.approx(0.03, abs=1e-4)
    assert r["p_hat"] == pytest.approx(0.003, abs=1e-4)
    assert r["pL_t1"] > 1.5 * r["pL_mw"]
    assert r["pL_ml"] <= r["pL_mw"] * (1 + 1e-9)
    r = t1_study(0.1, 0.01)
    assert r["pL_mw"] / r["pL_ml"] == pytest.approx(1.17, abs=0.02)
