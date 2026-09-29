"""
Tests for the signed qg updates in qang.gradients and
examples/signed_qg_range_qg.py (RESEARCH_NOTES §65).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import pytest

from qang.gradients import signed_natural_qg_step, signed_qg_step, signed_theta


def test_signed_theta_covers_circle():
    for t in (-3.0, -1.0, -0.2, 0.0, 0.5, 2.9):
        q, s = math.cos(t), (1 if math.sin(t) >= 0 else -1)
        assert signed_theta(q, s) == pytest.approx(t, abs=1e-12)


def test_reflection_through_pole():
    q, s = signed_qg_step(1.0, 1, grad_theta=0.2, lr=0.3)
    assert s == -1 and -1.0 <= q <= 1.0
    q, s = signed_natural_qg_step(1.0, 1, grad_theta=0.2, lr=0.3)
    assert signed_theta(q, s) == pytest.approx(-0.06, abs=1e-12)


def test_natural_step_is_theta_step_to_first_order():
    t, g, lr = 0.7, 0.3, 1e-4
    q, s = signed_natural_qg_step(math.cos(t), 1, g, lr)
    assert signed_theta(q, s) == pytest.approx(t - lr * g, abs=1e-8)


pytest.importorskip("qiskit")
import signed_qg_range_qg as S  # noqa: E402


def test_h2_trap_removed():
    h = S.h2_study(lrs=(0.1, 0.3))
    assert h[(0.1, "qg_clipped")][0] > 1e-2  # unsigned: trapped at Hartree-Fock
    assert h[(0.1, "signed_qg")][0] < 1e-6
    assert h[(0.3, "signed_natural_qg")][0] < 1e-6
    assert abs(h[(0.3, "signed_natural_qg")][1] - h[(0.3, "theta")][1]) <= 2
    assert h[(0.3, "signed_qg")][0] > 1e-2  # Euclidean qg step unstable


def test_landscapes():
    b = S.landscape_study(lrs=(0.5, 1.0), n=150)
    assert b[(0.5, "signed_natural_qg")] == pytest.approx(b[(0.5, "theta")], abs=0.03)
    assert b[(0.5, "qg_clipped")] < 0.1
    assert b[(1.0, "theta_pole_damped")] > b[(1.0, "signed_natural_qg")]
