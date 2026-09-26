"""
Tests for examples/bb84_qg_eve_vs_noise.py: BB84 error structure in qg
units, and the qg monitor vs the QBER monitor.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

import bb84_qg_eve_vs_noise as B  # noqa: E402


def test_ideal_channel_has_no_errors_and_qg_is_plus_minus_one():
    e = B.error_probabilities(gamma=0, p=0, e01=0, e10=0)
    assert np.allclose(e, 0)
    assert np.allclose(B.qg_per_state(e), [1, -1, 1, -1])


def test_t1_gives_asymmetric_errors_intercept_resend_symmetric():
    t1 = B.error_probabilities(gamma=0.1, p=0, e01=0, e10=0)
    assert t1[0] == pytest.approx(0) and t1[1] == pytest.approx(0.1)
    assert B.z_asymmetry(t1) == pytest.approx(0.2)
    ir = B.error_probabilities(gamma=0, p=0, e01=0, e10=0, f=0.2)
    assert np.allclose(ir, 0.05)  # f/4 in every category
    assert B.z_asymmetry(ir) == pytest.approx(0)
    # intercept-resend does not move the asymmetry of the baseline channel
    assert B.z_asymmetry(B.error_probabilities(f=0.1)) == pytest.approx(B.z_asymmetry(B.error_probabilities()))


def test_secret_fraction():
    assert B.shor_preskill(0, 0) == pytest.approx(1)
    assert B.shor_preskill(0.11, 0.11) == pytest.approx(0, abs=0.01)  # the BB84 threshold


def test_qg_monitor_ignores_t1_drift_but_catches_intercept_resend():
    th = B.calibrate_thresholds(2000, reps=600, seed=3)
    q_drift, g_drift = B.alarm_rates(B.error_probabilities(gamma=0.06), 2000, th, reps=200, seed=4)
    assert q_drift > 0.8 and g_drift < 0.05
    q_ir, g_ir = B.alarm_rates(B.error_probabilities(f=0.10), 2000, th, reps=200, seed=5)
    assert q_ir > 0.9 and g_ir > 0.9
    q0, g0 = B.alarm_rates(B.error_probabilities(), 2000, th, reps=300, seed=6)
    assert q0 < 0.04 and g0 < 0.04
