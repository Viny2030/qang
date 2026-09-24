"""
Tests for examples/control_quantization_qg_vs_theta.py: b-bit angle grids
uniform in theta vs uniform in qg.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

from control_quantization_qg_vs_theta import (  # noqa: E402
    benchmark_distributions,
    conditional_probabilities,
    loaded_distribution,
    loading_errors,
    qg_grid,
    single_qubit_errors,
    theta_grid,
)

HAAR = np.arccos(np.random.default_rng(0).uniform(-1, 1, 50_000))


def test_grids_have_2_to_the_b_points_and_cover_the_sphere():
    for b in (3, 6):
        for g in (theta_grid(b), qg_grid(b)):
            assert len(g) == 2**b
            assert g.min() == pytest.approx(0.0, abs=1e-12)
            assert g.max() == pytest.approx(math.pi, abs=1e-12)


@pytest.mark.parametrize("bits", [4, 6, 8, 10])
def test_finding_a_worst_case_probability_error_ratio_is_pi_over_2(bits):
    th = single_qubit_errors(HAAR, bits, "theta")[0]
    qg = single_qubit_errors(HAAR, bits, "qg")[0]
    assert qg == pytest.approx(1.0 / (2 * (2**bits - 1)), rel=0.02)  # grid half-step
    assert th / qg == pytest.approx(math.pi / 2, rel=0.03)


@pytest.mark.parametrize("bits", [4, 6, 8, 10])
def test_finding_a_worst_case_infidelity_favors_theta_and_the_gap_grows(bits):
    th = single_qubit_errors(HAAR, bits, "theta")[2]
    qg = single_qubit_errors(HAAR, bits, "qg")[2]
    assert qg > 5 * th
    if bits >= 8:
        assert qg > 100 * th


def test_exact_loading_reproduces_the_distribution():
    p = np.random.default_rng(3).dirichlet(np.ones(32))
    assert np.allclose(loaded_distribution(conditional_probabilities(p)), p, atol=1e-12)


def _cases():
    for n in (4, 6, 8):
        for name, d in benchmark_distributions(n).items():
            for b in (4, 6, 8):
                yield n, name, b, loading_errors(d, b, "theta"), loading_errors(d, b, "qg")


def test_finding_b_qg_grid_lowers_total_variation_distance_almost_always():
    ratios = [th[0] / qg[0] for *_, th, qg in _cases()]
    assert sum(r > 1 for r in ratios) == 43
    assert np.median(ratios) > 1.5


def test_finding_b_infidelity_depends_on_smooth_vs_sparse():
    smooth, sparse = [], []
    for _, name, _, th, qg in _cases():
        (smooth if name in ("normal", "lognormal", "Dirichlet(1)") else sparse).append(th[1] / qg[1])
    assert sum(r > 1 for r in smooth) == 25 and np.median(smooth) > 1.9
    assert sum(r < 1 for r in sparse) == 15 and np.median(sparse) < 0.35
