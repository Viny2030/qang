"""
Tests for examples/battery_ergotropy_qg.py (RESEARCH_NOTES §56).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("scipy")

import battery_ergotropy_qg as B  # noqa: E402


def test_closed_forms_match_general_formula():
    rng = np.random.default_rng(0)
    for _ in range(100):
        v = rng.normal(size=3)
        v *= rng.uniform() ** (1 / 3) / np.linalg.norm(v)
        w = B.ergotropy(*v)
        assert w == pytest.approx(B.ergotropy_general(B.rho_from_bloch(*v), [0.0, 1.0]), abs=1e-12)
        assert w == pytest.approx(B.ergotropy_incoherent(v[2]) + B.ergotropy_coherent(*v), abs=1e-12)


def test_storage_formula_matches_lindblad():
    for th, t, ratio in ((2.2, 0.8, 1.0), (math.pi / 2, 1.5, 2.0), (2.8, 0.3, 0.5)):
        assert B.stored_ergotropy(th, t, ratio) == pytest.approx(B.lindblad_check(th, t, ratio, steps=2000), abs=1e-6)


def test_inverted_battery_dies_at_t1_ln2():
    assert B.stored_ergotropy(math.pi, math.log(2) - 1e-3, 1.0) > 0
    assert B.stored_ergotropy(math.pi, math.log(2) + 1e-3, 1.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("ratio,expect", ((2.0, math.log(4 / 3)), (1.0, math.log(1.5))))
def test_crossover_closed_forms(ratio, expect):
    assert B.crossover_time_analytic(ratio) == pytest.approx(expect, abs=1e-6)
    assert B.crossover_time(ratio) == pytest.approx(expect, abs=2e-3)


def test_tilted_battery_keeps_charge_after_inversion_death():
    th, w = B.best_angle(1.0, 2.0)
    assert 0.5 * math.pi < th < 0.65 * math.pi and w == pytest.approx(0.129, abs=2e-3)
    assert B.best_angle(0.25, 1.0)[0] == pytest.approx(math.pi)


def test_certification_plug_in_overestimates_lcb_is_safe():
    rp, qz = B.aged(math.pi, 0.8, 1.0)
    c = B.certify((rp, 0.0, qz), 100, trials=1000)
    assert c["truth"] == 0.0 and c["plug_over"] > 0.9
    rp2, qz2 = B.aged(math.pi, 0.5, 1.0)
    c2 = B.certify((rp2, 0.0, qz2), 1000, trials=1000)
    assert c2["lcb_unsafe"] == 0.0 and 0 < c2["lcb_mean"] < c2["truth"]


def test_dicke_charge_is_locked_and_fragile():
    g, l, m = B.register_ergotropy(B.dicke(4, 2), 4)
    assert g == pytest.approx(2.0) and l == pytest.approx(0.0, abs=1e-12) and m == pytest.approx(0.0, abs=1e-12)
    gp, lp, _ = B.register_ergotropy(B.product_equal_energy(4, 2), 4)
    assert gp == pytest.approx(2.0) and lp == pytest.approx(2.0)
    gd, _, md = B.register_ergotropy(B._local_channel(B.dicke(4, 2), 4, 0.3, 1.0), 4)
    gp3, _, mp = B.register_ergotropy(B._local_channel(B.product_equal_energy(4, 2), 4, 0.3, 1.0), 4)
    assert gd < gp3 and md == pytest.approx(mp, abs=1e-9)
