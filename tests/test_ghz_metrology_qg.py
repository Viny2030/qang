"""
Tests for examples/ghz_metrology_qg.py (RESEARCH_NOTES §58): GHZ vs N
independent qubits under Markovian / Gaussian dephasing, readout error and
T1, with the qg witness.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import pytest

from ghz_metrology_qg import gain, ghz_parity_exact, n_star


@pytest.mark.parametrize("p,g,e", [(0.05, 0.0, 0.0), (0.0, 0.1, 0.0), (0.05, 0.1, 0.01)])
@pytest.mark.parametrize("n", [3, 4])
def test_exact_parity_formula(n, p, g, e):
    phi = 0.3
    par, mq = ghz_parity_exact(n, phi, p, g, e)
    pred = math.cos(n * phi) * (1 - 2 * p) ** n * (1 - g) ** (n / 2) * (1 - 2 * e) ** n
    assert par == pytest.approx(pred, abs=1e-10)
    assert mq == pytest.approx(g, abs=1e-10)  # T1 witness; 0 under dephasing alone


@pytest.mark.parametrize("n", [2, 10, 100])
def test_markov_no_gain(n):
    assert gain(n, 0.0, "markov") == pytest.approx(1.0, rel=1e-3)
    assert gain(n, 0.002, "markov") < 1.0


@pytest.mark.parametrize("n", [2, 10, 100])
def test_gauss_sqrt_n(n):
    assert gain(n, 0.0, "gauss") == pytest.approx(math.sqrt(n), rel=2e-3)


@pytest.mark.parametrize("e", [0.005, 0.01, 0.02])
def test_readout_cap(e):
    ns = n_star(e)
    assert abs(ns - 3 / (4 * abs(math.log(1 - 2 * e)))) <= 1
    assert gain(ns, e, "gauss") < 2.1


def test_short_window_heisenberg_then_flat():
    assert gain(10, 0.0, "markov", t_max=0.01) > 8.0
    flat = 1 / (2 * math.e * 0.01 * math.exp(-0.02))
    assert gain(300, 0.0, "markov", t_max=0.01) == pytest.approx(flat, rel=5e-3)
    assert gain(100, 0.01, "markov", t_max=0.01) < 1.0
