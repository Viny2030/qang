"""
Tests for examples/ode_qg_vs_angle.py: a differentiable-quantum-circuit
ODE solver with qg vs angle encoding, and a classical spectral control.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("scipy")

from ode_qg_vs_angle import (  # noqa: E402
    ENCODINGS,
    model_and_theta_derivative,
    solve_classical_chebyshev,
    solve_quantum,
)


def test_parameter_shift_derivative_matches_finite_differences():
    rng = np.random.default_rng(0)
    p = rng.uniform(-np.pi, np.pi, 3 * 4)
    th = np.linspace(0.2, 2.9, 17)
    _, df = model_and_theta_derivative(p, th, 3)
    h = 1e-6
    fp, _ = model_and_theta_derivative(p, th + h, 3)
    fm, _ = model_and_theta_derivative(p, th - h, 3)
    assert np.allclose(df, (fp - fm) / (2 * h), atol=1e-7)


def test_qg_encoding_wins_on_exponential_decay():
    qg = solve_quantum("qg", "decay", 3, restarts=4)
    angle = solve_quantum("angle_pi/2", "decay", 3, restarts=4)
    assert qg < 2e-4
    assert angle > 5 * qg


def test_angle_encoding_wins_on_the_sine():
    assert solve_quantum("angle_pi/2", "sine", 2) * 10 < solve_quantum("qg", "sine", 2)


def test_classical_spectral_method_with_same_parameter_count_is_near_exact():
    for problem in ("decay", "sine"):
        assert solve_classical_chebyshev(problem, 13) < 1e-10
