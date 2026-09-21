"""
Tests for examples/zne_qg_vs_theta_space.py: which space to extrapolate
in for zero-noise extrapolation (ZNE) depends on the noise mechanism.

Checked:
  1. For Bloch-vector-shrinkage noise (linear in qg_Z by construction),
     ZNE performed in qg_Z-space recovers the exact noiseless value to
     machine precision, while ZNE performed in theta-space has a small
     but clearly nonzero systematic bias.
  2. For coherent rotation-drift noise (linear in theta by construction),
     the roles are exactly reversed.
  3. The bias in the "wrong" space grows with the noise strength
     parameter, confirming it is a real systematic effect and not
     numerical noise.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

from zne_qg_vs_theta_space import (
    bloch_vector_shrinkage_noise,
    coherent_rotation_drift_noise,
    zne_in_qg_space,
    zne_in_theta_space,
)

THETA0 = np.pi / 3.0
QG0 = np.cos(THETA0)
LAMBDAS = np.array([1.0, 2.0, 3.0])


def test_bloch_vector_shrinkage_exact_in_qg_space():
    qg_noisy = bloch_vector_shrinkage_noise(QG0, gamma=0.08, lambdas=LAMBDAS)
    recovered = zne_in_qg_space(qg_noisy, LAMBDAS)
    assert recovered == pytest.approx(QG0, abs=1e-9)


def test_bloch_vector_shrinkage_biased_in_theta_space():
    qg_noisy = bloch_vector_shrinkage_noise(QG0, gamma=0.08, lambdas=LAMBDAS)
    recovered = zne_in_theta_space(qg_noisy, LAMBDAS)
    error = abs(recovered - QG0)
    assert error > 1e-4  # a real, non-negligible systematic bias
    assert error < 0.05  # but still a small correction, not a gross failure


def test_coherent_rotation_drift_exact_in_theta_space():
    qg_noisy = coherent_rotation_drift_noise(THETA0, delta=0.05, lambdas=LAMBDAS)
    recovered = zne_in_theta_space(qg_noisy, LAMBDAS)
    assert recovered == pytest.approx(QG0, abs=1e-9)


def test_coherent_rotation_drift_biased_in_qg_space():
    qg_noisy = coherent_rotation_drift_noise(THETA0, delta=0.05, lambdas=LAMBDAS)
    recovered = zne_in_qg_space(qg_noisy, LAMBDAS)
    error = abs(recovered - QG0)
    assert error > 1e-4
    assert error < 0.05


@pytest.mark.parametrize("gamma", [0.02, 0.05, 0.1, 0.15])
def test_wrong_space_bias_grows_with_noise_strength(gamma):
    """The theta-space bias for Model A should grow (roughly monotonically)
    as the noise strength gamma increases -- confirming this is a real
    systematic effect tied to the noise magnitude, not an artifact."""
    qg_noisy_weak = bloch_vector_shrinkage_noise(QG0, gamma=gamma, lambdas=LAMBDAS)
    error = abs(zne_in_theta_space(qg_noisy_weak, LAMBDAS) - QG0)
    qg_noisy_weaker = bloch_vector_shrinkage_noise(QG0, gamma=gamma / 2.0, lambdas=LAMBDAS)
    error_weaker = abs(zne_in_theta_space(qg_noisy_weaker, LAMBDAS) - QG0)
    assert error > error_weaker


def test_correct_space_always_beats_wrong_space_for_both_models():
    qg_noisy_A = bloch_vector_shrinkage_noise(QG0, gamma=0.08, lambdas=LAMBDAS)
    err_correct_A = abs(zne_in_qg_space(qg_noisy_A, LAMBDAS) - QG0)
    err_wrong_A = abs(zne_in_theta_space(qg_noisy_A, LAMBDAS) - QG0)
    assert err_correct_A < err_wrong_A

    qg_noisy_B = coherent_rotation_drift_noise(THETA0, delta=0.05, lambdas=LAMBDAS)
    err_correct_B = abs(zne_in_theta_space(qg_noisy_B, LAMBDAS) - QG0)
    err_wrong_B = abs(zne_in_qg_space(qg_noisy_B, LAMBDAS) - QG0)
    assert err_correct_B < err_wrong_B


if __name__ == "__main__":
    print("Run via `pytest tests/test_zne_qg_vs_theta_space.py -v` for full coverage.")
    qg_noisy = bloch_vector_shrinkage_noise(QG0, gamma=0.08, lambdas=LAMBDAS)
    assert zne_in_qg_space(qg_noisy, LAMBDAS) == pytest.approx(QG0, abs=1e-9)
    print("Smoke check passed.")
