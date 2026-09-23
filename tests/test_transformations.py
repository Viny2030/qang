"""
Tests for qang.transformations.transition_probability_qang.

Includes the exact cross-check referenced by Section 13 of the full
reference notebook: 200 random pairs checked against an explicit
statevector overlap computation, independent of the closed form.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from qang.transformations import transition_probability_qang


def _transition_probability_via_statevectors(q1: float, q2: float) -> float:
    """Reference implementation via explicit real-amplitude statevectors,
    used only to independently verify the closed form above."""
    theta1, theta2 = math.acos(q1), math.acos(q2)
    psi1 = np.array([math.cos(theta1 / 2), math.sin(theta1 / 2)])
    psi2 = np.array([math.cos(theta2 / 2), math.sin(theta2 / 2)])
    return float(np.dot(psi1, psi2)) ** 2


def test_anchor_north_pole_to_t_gate():
    # matches the paper's own anchor values: |0> (qg_Z=1.0) -> T-gate state (qg_Z=1/sqrt(2))
    p = transition_probability_qang(1.0, 1.0 / math.sqrt(2.0))
    expected = (1.0 + 1.0 / math.sqrt(2.0)) / 2.0
    assert abs(p - expected) < 1e-9


def test_identity_transition_is_certain():
    for q in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        assert abs(transition_probability_qang(q, q) - 1.0) < 1e-9


def test_orthogonal_poles_never_transition():
    assert abs(transition_probability_qang(1.0, -1.0)) < 1e-9


def test_symmetry():
    rng = np.random.default_rng(42)
    for _ in range(50):
        a, b = rng.uniform(-1.0, 1.0, size=2)
        assert abs(
            transition_probability_qang(a, b) - transition_probability_qang(b, a)
        ) < 1e-12


def test_rejects_out_of_range():
    with pytest.raises(ValueError):
        transition_probability_qang(1.5, 0.0)
    with pytest.raises(ValueError):
        transition_probability_qang(0.0, -1.5)


def test_cross_check_against_statevector_overlap_200_random_pairs():
    """The exact cross-check Section 13 of the notebook refers to:
    200 random (q1, q2) pairs, closed form vs. explicit statevector overlap."""
    rng = np.random.default_rng(0)
    pairs = rng.uniform(-1.0, 1.0, size=(200, 2))
    max_err = max(
        abs(transition_probability_qang(a, b) - _transition_probability_via_statevectors(a, b))
        for a, b in pairs
    )
    assert max_err < 1e-9


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)
