"""
Tests for examples/bb84_attacks_beyond_ir_qg.py (RESEARCH_NOTES §49):
Pauli identities between attacks and noise, and the extended flags.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

import bb84_attacks_beyond_ir_qg as A  # noqa: E402
import bb84_qg_eve_vs_noise as B  # noqa: E402


def test_pauli_form_matches_the_section_40_channel():
    rng = np.random.default_rng(0)
    for _ in range(10):
        g, p, f = rng.uniform(0, 0.1), rng.uniform(0, 0.05), rng.uniform(0, 0.3)
        e01, e10 = rng.uniform(0, 0.02), rng.uniform(0, 0.03)
        assert np.allclose(A.error_probabilities(g, p, e01, e10, px=f / 4, pz=f / 4),
                           B.error_probabilities(g, p, e01, e10, f), atol=1e-14)


@pytest.mark.parametrize("d", (0.005, 0.0125, 0.03))
def test_cloner_is_intercept_resend_for_bob(d):
    assert np.allclose(A.error_probabilities(py=d), A.error_probabilities(px=d, pz=d), atol=1e-15)


@pytest.mark.parametrize("f", (0.02, 0.04, 0.1))
def test_z_only_intercept_is_t2_drift_for_bob(f):
    p0 = A.BASE["p"]
    p_eq = (1 - (1 - 2 * p0) * (1 - f)) / 2
    assert np.allclose(A.error_probabilities(pz=f / 2), A.error_probabilities(p=p_eq), atol=1e-15)


@pytest.fixture(scope="module")
def flags():
    rng = np.random.default_rng(3)
    th3 = A.calibrate3(reps=150)
    th2 = A.F.calibrate(A.N_KEY, A.K_TEST, 150)

    def run(kw, reps=60):
        probs = A.error_probabilities(**kw)
        h = np.zeros(5)
        for _ in range(reps):
            kc, nc = A.F.block(probs, A.N_KEY, A.K_TEST, rng)
            a, t1, t2 = A.statistics3(kc, nc)
            a2, d2 = A.F.statistics(kc, nc)
            h += [a > th3[0], t1 > th3[1], t2 > th3[2], a2 > th2[0], d2 > th2[1]]
        return h / reps

    return run


def test_cloner_flagged_as_attack(flags):
    r = flags({"py": 0.0125})
    assert r[0] > 0.95 and r[2] < 0.15


def test_t2_drift_false_alarm_in_two_family_model_fixed_in_three(flags):
    r = flags({"p": 0.03})
    assert r[3] > 0.1  # §43 model: false attack alarms
    assert r[0] < 0.1 and r[2] > 0.95


def test_z_only_intercept_hides_as_t2(flags):
    r = flags({"pz": 0.02})
    assert r[0] < 0.1 and r[2] > 0.95


def test_x_only_intercept_and_mixture_are_caught(flags):
    r = flags({"px": 0.02})
    assert r[0] > 0.95 and r[2] < 0.1
    m = flags({"p": 0.02, "py": 0.0125})
    assert m[0] > 0.95 and m[2] > 0.9
