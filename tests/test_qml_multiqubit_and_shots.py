"""
Tests for examples/qml_multiqubit_and_shots.py: the qg-encoding result on
two qubits, with two input features, and under finite-shot training.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("scipy")

from qml_encoding_qg_vs_angle import reupload_output  # noqa: E402
from qml_multiqubit_and_shots import (  # noqa: E402
    ENCODINGS,
    TARGETS_2D,
    X2_TEST,
    X2_TRAIN,
    classical_chebyshev_fit,
    cubic,
    fit,
    model_output,
    train_with_shots,
)


def test_one_qubit_model_matches_the_single_qubit_example():
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 37)
    for L in (1, 3):
        p = rng.uniform(-np.pi, np.pi, 3 * (L + 1))
        a = ENCODINGS["qg"](x)
        assert np.allclose(model_output(p, a[:, None], 1, L), reupload_output(p, a, L), atol=1e-12)


def test_finding_b_product_of_features_is_exact_with_qg_only():
    t = TARGETS_2D["x1 * x2"]
    assert fit(ENCODINGS["qg"], t, X2_TRAIN, X2_TEST, 2, 2, restarts=3) < 1e-12
    assert fit(ENCODINGS["angle_pi/2"], t, X2_TRAIN, X2_TEST, 2, 2, restarts=3) > 1e-5


def test_finding_b_periodic_product_is_exact_with_angle_pi_only():
    t = TARGETS_2D["sin(pi x1) cos(pi x2)"]
    assert fit(ENCODINGS["angle_pi"], t, X2_TRAIN, X2_TEST, 2, 2, restarts=3) < 1e-12
    assert fit(ENCODINGS["qg"], t, X2_TRAIN, X2_TEST, 2, 2, restarts=3) > 0.05


def test_finding_c_qg_advantage_survives_1000_shot_training():
    qg = train_with_shots(ENCODINGS["qg"], cubic, 3, 1000, seed=0)
    angle = train_with_shots(ENCODINGS["angle_pi/2"], cubic, 3, 1000, seed=0)
    assert qg < 1e-4
    assert angle > 10 * qg


def test_classical_chebyshev_fit_is_exact():
    assert classical_chebyshev_fit(cubic, 3) < 1e-25
