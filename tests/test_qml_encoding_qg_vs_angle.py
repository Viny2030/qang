"""
Tests for examples/qml_encoding_qg_vs_angle.py: qg encoding theta =
arccos(x) vs angle encoding in a single-qubit data re-uploading regressor.

Checked:
  1. With identity processing, L qg-encoded layers give T_L(x) exactly.
  2. Finding A: L qg layers fit a degree-L polynomial exactly and L - 1
     do not; on x^3 - 0.5x at L = 3 qg beats both fixed-scale and
     trainable-scale angle encoding.
  3. Finding B: angle encoding wins on non-polynomial targets, and a wrong
     scale makes it fail.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("scipy")

from numpy.polynomial import chebyshev as C  # noqa: E402

from qml_encoding_qg_vs_angle import (  # noqa: E402
    ENCODINGS,
    TARGETS,
    fit,
    fit_trainable_scale,
    random_bounded_chebyshev,
    reupload_output,
)


@pytest.mark.parametrize("n_layers", [1, 2, 3, 5])
def test_identity_processing_gives_chebyshev_polynomial(n_layers):
    x = np.linspace(-1, 1, 101)
    out = reupload_output(np.zeros(3 * (n_layers + 1)), ENCODINGS["qg"](x), n_layers)
    t_l = C.chebval(x, [0] * n_layers + [1])
    assert np.max(np.abs(out - t_l)) < 1e-12


@pytest.mark.parametrize("degree", [2, 3])
def test_finding_a_degree_L_polynomial_needs_exactly_L_layers(degree):
    target = random_bounded_chebyshev(degree, np.random.default_rng(10 + degree))
    assert fit(ENCODINGS["qg"], target, degree)[1] < 1e-10
    assert fit(ENCODINGS["qg"], target, degree - 1)[1] > 1e-6


def test_finding_a_qg_beats_angle_encodings_on_a_cubic():
    target = TARGETS["x^3 - 0.5x"]
    qg = fit(ENCODINGS["qg"], target, 3)[1]
    best_fixed = min(fit(ENCODINGS[e], target, 3)[1] for e in ("angle_1", "angle_pi/2", "angle_pi"))
    trainable = fit_trainable_scale(target, 3)[1]
    assert qg < 1e-12
    assert best_fixed > 1e-4
    assert trainable > 1e-5


def test_finding_b_angle_pi_is_exact_on_sin_pi_x_where_qg_is_not():
    target = TARGETS["sin(pi x)"]
    assert fit(ENCODINGS["angle_pi"], target, 1)[1] < 1e-12
    assert fit(ENCODINGS["qg"], target, 1)[1] > 0.1


def test_finding_b_trainable_angle_beats_qg_on_runge():
    target = TARGETS["Runge"]
    assert fit_trainable_scale(target, 4)[1] * 1000 < fit(ENCODINGS["qg"], target, 4)[1]


def test_finding_b_wrong_angle_scale_fails():
    assert fit(ENCODINGS["angle_1"], TARGETS["cos(2 pi x)"], 4)[1] > 0.3
