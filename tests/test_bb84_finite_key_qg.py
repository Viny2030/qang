"""
Tests for examples/bb84_finite_key_qg.py: the finite-key length of
Tomamichel et al. (2012), its approach to the asymptotic rate, and the qg
diagnosis (drift vs attack, including an attack hidden under drift).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("scipy")

import bb84_finite_key_qg as F  # noqa: E402
import bb84_qg_eve_vs_noise as B  # noqa: E402


def test_mu_and_key_length_formula():
    n, k = 1e5, 1e4
    m = math.sqrt((n + k) / (n * k) * (k + 1) / k * math.log(2 / F.EPS_SEC))
    assert F.mu(n, k) == pytest.approx(m)
    ell = n * (1 - F.h(0.03 + m)) - 1.1 * n * F.h(0.02) - math.log2(2 / (F.EPS_SEC**2 * F.EPS_COR))
    assert F.key_length(n, k, 0.03, 0.02) == pytest.approx(ell)


def test_asymptotic_rate_matches_shor_preskill_with_ec_inefficiency():
    _, qz, qx = F.channel()
    assert F.asymptotic_rate(qz, qx) == pytest.approx(1 - F.h(qx) - 1.1 * F.h(qz))
    assert F.asymptotic_rate(qz, qx) == pytest.approx(0.707, abs=1e-3)


def test_finite_rate_grows_towards_the_asymptote():
    _, qz, qx = F.channel()
    rs = [F.optimize(n, qz, qx)[0] for n in (1e4, 1e6, 1e9)]
    assert rs[0] == pytest.approx(0.120, abs=0.005)
    assert rs[0] < rs[1] < rs[2] < F.asymptotic_rate(qz, qx)
    assert rs[2] > 0.9 * F.asymptotic_rate(qz, qx)


def test_no_key_for_tiny_blocks_and_drift_costs_key():
    _, qz, qx = F.channel()
    assert F.optimize(500, qz, qx)[0] == 0.0
    base = F.optimize(1e5, qz, qx)[0]
    _, qz4, qx4 = F.channel(gamma=0.04)
    _, qz6, qx6 = F.channel(gamma=0.06)
    assert base > F.optimize(1e5, qz4, qx4)[0] > F.optimize(1e5, qz6, qx6)[0] > 0


def test_statistics_are_nested_and_nonnegative():
    rng = np.random.default_rng(0)
    for kw in ({}, {"gamma": 0.05}, {"f": 0.05}):
        sa, sd = F.statistics(*F.block(B.error_probabilities(**kw), 20000, 4000, rng))
        assert sa >= -1e-9 and sd >= -1e-9


def test_diagnosis_attributes_drift_attack_and_mixture():
    d = F.diagnose(10000, reps=120)
    assert d["baseline"]["abort"] < 0.05
    assert d["T1 drift 0.06"]["abort"] > 0.6
    for name in ("T1 drift 0.04", "T1 drift 0.06", "T1-mimicking attack"):
        assert d[name]["drift"] > 0.95 and d[name]["attack"] < 0.06
    for name in ("intercept f = 0.02", "intercept f = 0.05"):
        assert d[name]["attack"] > 0.9 and d[name]["drift"] < 0.06
    mix = d["drift 0.04 + intercept 0.02"]
    assert mix["attack"] > 0.85 and mix["drift"] > 0.95


def test_one_parameter_glrt_misses_attack_hidden_under_drift():
    rng = np.random.default_rng(5)
    n, k = 100000, 7609
    base = [B.glrt_statistic(*F.block(B._BASE_PROBS, n, k, rng)) for _ in range(100)]
    th = float(np.quantile(base, 0.99))
    mix = B.error_probabilities(gamma=0.04, f=0.02)
    hits = [B.glrt_statistic(*F.block(mix, n, k, rng)) > th for _ in range(60)]
    assert np.mean(hits) < 0.1
