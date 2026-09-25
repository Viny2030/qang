"""
Tests for examples/ising_coherence_witness_qg.py: the §25 entropy gap is
the relative entropy of coherence, and single-qubit qg gives a tight
superadditive lower bound on thermal transverse-field Ising states.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import ising_coherence_witness_qg as I  # noqa: E402
from qang.core import qg_s_from_qg_z  # noqa: E402
from qang.multiqubit import per_qubit_qg_z, qg_correlation  # noqa: E402


def test_gap_equals_relative_entropy_of_coherence_and_section_25_formula():
    rho = I.thermal_state(1.5, 0.5)
    n = I.N_SITES
    s_qg = n * qg_s_from_qg_z(float(np.mean(per_qubit_qg_z(rho, n)))) - qg_correlation(rho, n)
    assert s_qg - I.von_neumann_bits(rho) == pytest.approx(I.exact_coherence(rho), abs=1e-9)


def test_no_coherence_without_transverse_field():
    for beta in (0.5, 2.0):
        r = I.exact_row(beta, 0.0)
        assert r["C"] == pytest.approx(0, abs=1e-9) and r["sum"] == pytest.approx(0, abs=1e-12)
        assert r["qg_x"] == pytest.approx(0, abs=1e-12)


def test_single_qubit_bound_matches_direct_computation():
    # |r| = 0.6 at qg_Z = 0: C = 1 - H((1 + 0.6)/2)
    assert I.qubit_bound_from_qg(0.6, 0.0) == pytest.approx(1 - I.binary_entropy(0.8))
    assert I.qubit_bound_from_qg(0.0, 0.3) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("beta,g", [(0.25, 1.0), (0.5, 0.5), (1.5, 1.5), (3.0, 2.0)])
def test_superadditive_sum_bound_is_valid_and_tight(beta, g):
    r = I.exact_row(beta, g)
    assert r["qubit"] <= r["sum"] <= r["C"] + 1e-12
    assert r["sum"] / r["C"] > 0.9


def test_basis_entropy_bound_is_mostly_useless_here():
    assert I.exact_row(1.5, 1.0)["basis"] < -4
    assert I.exact_row(0.25, 2.0)["basis"] > 0  # the only positive corner of the grid


def test_graph_state_defeats_the_qg_bound():
    c, b = I.graph_state_counterexample()
    assert c == pytest.approx(I.N_SITES, abs=1e-9) and b == pytest.approx(0.0, abs=1e-12)


def test_finite_shots_and_false_positives():
    t = I.shot_table(shots_list=(1000,), cases=((1.5, 1.5),), reps=60, seed=2)
    r = t[(1.5, 1.5, 1000)]
    assert abs(r["sum"][0] - r["exact"]["sum"]) < 0.1
    fp_qx, fp_basis = I.false_positive_rate(1000, reps=200)
    assert fp_qx < 0.03 and fp_basis == 0.0
