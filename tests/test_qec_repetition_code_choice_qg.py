"""
Tests for examples/qec_repetition_code_choice_qg.py: choosing a
repetition code (or none) under T1 + dephasing with the qg witness.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

from qec_repetition_code_choice_qg import (  # noqa: E402
    OPTIONS,
    boundary_ratio,
    estimate_noise,
    kraus_t1_tphi,
    logical_infidelity,
    policy_table,
    witness_qg,
)


def test_noise_channel_is_trace_preserving():
    k = kraus_t1_tphi(0.03, 0.02)
    assert np.allclose(sum(m.T @ m for m in k), np.eye(2))


def test_noiseless_is_perfect_and_codes_correct_their_error():
    for kind in OPTIONS:
        assert logical_infidelity(kind, 0.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    # phase code: dephasing only enters at second order
    assert logical_infidelity("phase", 0.0, 0.01) < 0.1 * logical_infidelity("none", 0.0, 0.01)


def test_repetition_codes_do_not_help_against_t1():
    for g in (0.005, 0.02, 0.05):
        bare = logical_infidelity("none", g, 0.0)
        assert logical_infidelity("bit", g, 0.0) > bare
        assert logical_infidelity("phase", g, 0.0) > bare


def test_boundary_is_p_equal_gamma():
    for g in (0.002, 0.01):
        assert boundary_ratio(g) == pytest.approx(1.0, abs=0.05)


def test_witness_inverts_exactly():
    for g, p in ((0.01, 0.0), (0.02, 0.03), (0.001, 0.04)):
        gh, ph = estimate_noise(*witness_qg(g, p))
        assert gh == pytest.approx(g, abs=1e-12) and ph == pytest.approx(p, abs=1e-12)


def test_qiskit_circuit_matches_exact_model():
    pytest.importorskip("qiskit")
    from qec_repetition_code_choice_qg import qiskit_logical_infidelity

    for kind in OPTIONS:
        assert qiskit_logical_infidelity(kind, 0.02, 0.01) == pytest.approx(
            logical_infidelity(kind, 0.02, 0.01), abs=1e-6)


def test_qg_witness_policy_beats_fixed_choices():
    t = policy_table(n_instances=60, shots=1000, seed=5)
    assert t["qg witness"]["regret"] < 0.1
    for k in ("always phase", "always none", "always bit", "qg_Z only"):
        assert t["qg witness"]["mean_infidelity"] < t[k]["mean_infidelity"]
    assert math.isclose(t["qg_Z only"]["mean_infidelity"], t["always none"]["mean_infidelity"])
