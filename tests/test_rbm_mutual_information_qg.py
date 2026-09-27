"""
Tests for examples/rbm_mutual_information_qg.py (RESEARCH_NOTES §47): the
I-eta bounds of the RBM review written in qg marginals, and the trained
RBMs on and off the lower bound.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("scipy")

import rbm_mutual_information_qg as R  # noqa: E402


@pytest.mark.parametrize("eta", (0.05, 0.3, 0.5, 0.8, 0.95))
def test_bounds_are_the_marginal_extremes(eta):
    assert R.mutual_information(0.0, 0.0, eta) == pytest.approx(R.lower_bound(eta), abs=1e-12)
    r = math.sqrt(1 - eta)
    assert R.mutual_information(r, r, eta) == pytest.approx(R.upper_bound(eta), abs=1e-12)
    grid = np.linspace(-0.99, 0.99, 67)
    vals = []
    for a in grid:
        for b in grid:
            if R.pair_distribution(a, b, eta).min() >= 0:
                vals.append(R.mutual_information(a, b, eta))
    assert min(vals) >= R.lower_bound(eta) - 1e-12
    assert max(vals) <= R.upper_bound(eta) + 1e-12


def test_small_bias_expansion():
    eta, q = 0.1, 0.05  # leading order in both eta and the marginals
    gap = R.mutual_information(q, q, eta) - R.lower_bound(eta)
    assert gap == pytest.approx(R.small_bias_gap(q, q, eta), rel=0.1)


def test_marginal_entropy_is_qg_s():
    for q in (-0.7, 0.0, 0.4):
        p = (1 + q) / 2
        assert R.qg_s(q) == pytest.approx(-p * math.log2(p) - (1 - p) * math.log2(1 - p))


def test_driver_ground_state_is_z2_symmetric_without_field():
    H = R.driver_matrix(1.0, 0.0)
    w, v = np.linalg.eigh(H)
    p = v[:, 0] ** 2
    assert np.allclose(p @ R._V, 0.0, atol=1e-10)
    Hf = R.driver_matrix(1.0, 0.3)
    pf = np.linalg.eigh(Hf)[1][:, 0] ** 2
    assert np.all(pf @ R._V > 0.1)


def _gap_and_marginals(x):
    st = R.pair_statistics(x)
    gap = np.mean([p[3] - R.lower_bound(p[2]) for p in st])
    return gap, np.mean([abs(p[0]) for p in st]), np.mean([abs(p[1]) for p in st])


def test_symmetric_driver_puts_the_learner_on_the_lower_bound():
    x, fit = R.train(1.0, 0.0, 0)
    gap, qv, qh = _gap_and_marginals(x)
    assert fit["fidelity"] > 0.999
    assert gap < 1e-5 and qv < 0.01 and qh < 0.01


def test_longitudinal_field_moves_it_off():
    x, fit = R.train(1.0, 0.3, 0)
    gap, qv, _ = _gap_and_marginals(x)
    assert fit["fidelity"] > 0.999
    assert qv > 0.3 and gap > 5e-3


def test_ordered_phase_two_learners_two_positions():
    xs, fs = R.train(0.5, 0.0, 0, symmetric=True)
    gap_s, qv_s, _ = _gap_and_marginals(xs)
    assert fs["fidelity"] > 0.999 and gap_s < 1e-6 and qv_s < 0.01
    xb, fb = R.train(0.5, 0.0, 0)
    gap_b, qv_b, _ = _gap_and_marginals(xb)
    assert fb["fidelity"] < 0.6 and qv_b > 0.5 and gap_b > 5e-3
