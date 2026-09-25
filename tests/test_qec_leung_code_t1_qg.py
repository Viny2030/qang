"""
Tests for examples/qec_leung_code_t1_qg.py: the 4-qubit Leung code for
amplitude damping, its dephasing boundary, the T2/T1 rule and the qg
witness policy.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

import qec_leung_code_t1_qg as L  # noqa: E402


def test_codewords_are_orthonormal():
    v = L.leung_encoder()
    assert np.allclose(v.T @ v, np.eye(2))


def test_recoveries_are_trace_preserving():
    for rec in (L.polar_recovery(0.02), L.petz_recovery(0.02, 0.005)):
        s = sum(k.conj().T @ k for k in rec)
        assert np.allclose(s, np.eye(16), atol=1e-9)


def test_noiseless_channel_is_perfect():
    assert L.leung_infidelity(0.0, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_pure_t1_is_corrected_to_second_order():
    i1, i2 = L.leung_infidelity(0.01, 0), L.leung_infidelity(0.001, 0)
    assert i1 / i2 == pytest.approx(100, rel=0.05)  # O(gamma^2)
    assert i1 == pytest.approx(9.15e-5, rel=0.02)
    assert L.infidelity("none", 0.01, 0) / i1 > 30
    assert L.leung_infidelity(0.01, 0, "petz") > i1  # Petz is worse, still O(gamma^2)
    assert 0.40 < L.crossover_gamma() < 0.48


def test_repetition_codes_do_not_correct_t1():
    for kind in ("bit", "phase"):
        assert L.repetition_infidelity(kind, 0.01, 0) > L.infidelity("none", 0.01, 0)


def test_dephasing_boundary_is_p_equal_gamma_over_four():
    for g in (0.002, 0.01):
        assert L.leung_boundary_ratio(g) == pytest.approx(0.25, abs=0.01)


def test_t2_over_t1_rule():
    for t in (0.002, 0.01, 0.03):
        best = dict(L.best_by_t2_ratio(t))
        assert best[0.3] == "phase" and best[0.8] == "none" and best[1.2] == "leung" and best[1.9] == "leung"
    g, p = L.rates_from_times(0.01, 1.0)  # T2 = T1  <=>  p = gamma/4 to first order
    assert p / g == pytest.approx(0.25, rel=0.02)


def test_policy_with_leung_option_beats_section_27_policy():
    t = L.policy_table(n_instances=120, shots=1000, seed=5)
    qg = t["qg witness (none/phase/leung)"]
    assert qg["regret"] < t["§27 policy (none/phase)"]["regret"]
    assert qg["regret"] < 0.15 and t["always leung"]["regret"] > 1.0
    assert t["oracle_share"]["leung"] > 0.1


def test_qiskit_cross_check():
    pytest.importorskip("qiskit_aer")
    for g, p in ((0.02, 0.0), (0.05, 0.01)):
        assert L.qiskit_leung_infidelity(g, p) == pytest.approx(L.leung_infidelity(g, p), abs=1e-10)
