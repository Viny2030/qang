"""
Tests for examples/amplitude_estimation_chebyshev_qg.py: Grover amplitude
estimation as Chebyshev polynomials in qg, the m^2/(1 - qg^2) Fisher
factor, and MLAE vs Monte Carlo with and without noise.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

import amplitude_estimation_chebyshev_qg as G  # noqa: E402


def test_grover_iterations_are_chebyshev_polynomials_in_qg():
    t = 0.37
    q0 = math.cos(2 * t)
    for k in range(6):
        assert G.flag_qg(q0, k) == pytest.approx(math.cos(2 * (2 * k + 1) * t), abs=1e-12)


def test_fisher_factor_is_m_squared_over_one_minus_qg_squared():
    q = np.linspace(-0.97, 0.97, 25)
    for m in (1, 3, 5, 9, 17):
        assert np.allclose(G.fisher_depth(m, q) * (1 - q**2) / m**2, 1.0, atol=1e-6)


def test_qiskit_grover_circuit_matches():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    t = 0.41  # A = Ry(2t): cos t |0> + sin t |1>, "good" = |1>
    for k in range(4):
        qc = QuantumCircuit(1)
        qc.ry(2 * t, 0)
        for _ in range(k):
            qc.z(0)                     # S_good: flip the sign of |1>
            qc.ry(-2 * t, 0)            # A^dagger
            qc.x(0)
            qc.z(0)
            qc.x(0)                     # S_0 (up to a global phase)
            qc.ry(2 * t, 0)             # A
        p1 = abs(Statevector(qc).data[1]) ** 2
        assert 1 - 2 * p1 == pytest.approx(G.flag_qg(math.cos(2 * t), k), abs=1e-12)


def test_mlae_beats_monte_carlo_noiselessly():
    n_q, e_ae, e_mc = G.run(0.3, 5, 100, reps=60, seed=1)
    assert n_q == 13300 and e_mc / e_ae > 3


def test_noise_must_be_in_the_likelihood():
    _, naive, mc = G.run(0.3, 6, 100, reps=40, p_true=0.01, p_model=0.0, seed=2)
    _, aware, _ = G.run(0.3, 6, 100, reps=40, p_true=0.01, p_model=0.01, seed=2)
    assert naive > mc and aware < mc / 2
