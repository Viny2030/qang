"""Tests for qang.clock_comparison (§126) and qang.complementarity (§127)."""

import math
import os
import sys

import numpy as np
import pytest

from qang import clock_comparison as CC
from qang import complementarity as CM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))


def test_noiseless_pairs_lie_on_the_ellipse_and_fit_exactly():
    delta = 0.4
    phi = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    x, y = np.cos(phi + delta), np.cos(phi - delta)
    c = math.cos(2 * delta)
    assert np.allclose(x ** 2 + y ** 2 - 2 * x * y * c, 1 - c ** 2, atol=1e-12)
    est = CC.ellipse_estimate(x, y)
    assert est["delta"] == pytest.approx(delta, abs=1e-9)
    assert CC.moment_estimate(x, y)["delta"] == pytest.approx(delta, abs=1e-9)


def test_ellipse_needs_only_part_of_the_circle():
    # the moment estimator assumes a uniform common phase; the ellipse fit does not
    delta = 0.3
    phi = np.linspace(-0.6, 0.9, 40)
    x, y = np.cos(phi + delta), np.cos(phi - delta)
    assert CC.ellipse_estimate(x, y)["delta"] == pytest.approx(delta, abs=1e-9)
    assert abs(CC.moment_estimate(x, y)["delta"] - delta) > 0.01


def test_redshift_phase_strontium_one_metre():
    d2 = CC.redshift_phase(429.228004229873e12, 1.0, 1.0)
    assert d2 == pytest.approx(2 * math.pi * 429.228e12 * 9.80665 / 299792458.0 ** 2, rel=1e-6)
    assert 0.29 < d2 < 0.30


def test_simulate_pairs_shapes_and_range():
    x, y = CC.simulate_pairs(0.2, 30, 100, np.random.default_rng(0))
    assert x.shape == y.shape == (30,) and np.all(np.abs(x) <= 1) and np.all(np.abs(y) <= 1)


def test_duality_is_the_bloch_radius():
    for x, y, z in [(0, 0, 1), (1, 0, 0), (0.3, -0.4, 0.5), (0, 0, 0)]:
        assert CM.duality_sum(x, y, z) == pytest.approx(x * x + y * y + z * z)
    assert CM.predictability(-0.7) == 0.7 and CM.visibility(0.6, 0.8) == pytest.approx(1.0)


def test_entropic_uncertainty_bound_and_equality():
    assert CM.entropic_sum(1.0, 0.0) == pytest.approx(1.0)           # |0>: H(Z) = 0, H(X) = 1
    assert CM.entropic_bound(0.0, 0.0, 1.0) == pytest.approx(1.0)
    assert CM.entropic_sum(0.0, 0.0) == pytest.approx(2.0)           # maximally mixed
    assert CM.entropic_bound(0.0, 0.0, 0.0) == pytest.approx(2.0)
    rng = np.random.default_rng(3)
    for _ in range(200):
        v = rng.normal(size=3)
        v *= rng.uniform() ** (1 / 3) / np.linalg.norm(v)
        x, y, z = v
        assert CM.entropic_sum(z, x) >= CM.entropic_bound(x, y, z) - 1e-12


def test_counts_interfaces():
    r2, se = CM.duality_from_counts((500, 1000), (500, 1000), (1000, 1000))
    assert r2 == pytest.approx(1.0, abs=0.01) and se >= 0
    assert CM.entropic_sum_from_counts((100, 100), (50, 100), "none") == pytest.approx(1.0)
    assert CM.entropic_sum_from_counts((100, 100), (50, 100)) > 1.0
    with pytest.raises(ValueError):
        CM.entropic_sum_from_counts((1, 2), (1, 2), "bogus")


def test_example_verdicts():
    import clock_redshift_qg as R
    import complementarity_qg as Q

    good_r = {"G1": {0.15: 0.01}, "G2": (0.01, 0.05), "G3": (0.2, 0.01), "G4": (0.147, 0.147, 0.02)}
    assert all(R.verdict(good_r).values())
    good_q = {"U1": (0.0, 1.0), "U2": (0.0, 0.0), "U3": (1e-4, 0.0025, 0.002, 0.003), "U4": (0.9, 0.5)}
    assert all(Q.verdict(good_q).values())
