"""Tests for examples/qec_syndrome_destructive_qg.py (§93) and the §92 joint fit."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("scipy")

import qec_syndrome_destructive_qg as D  # noqa: E402
import qec_syndrome_jointfit_qg as J  # noqa: E402


def test_destructive_code_states_have_no_flips():
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    for logical in (0, 1):
        cz = sim.run(transpile(D.build(0, logical, "Z"), sim), shots=200, seed_simulator=1).result().get_counts()
        cx = sim.run(transpile(D.build(0, logical, "X"), sim), shots=200, seed_simulator=1).result().get_counts()
        assert D.flips(cz, cx) == (0, 0, 200, 200)


def test_joint_fit_recovers_synthetic_parameters():
    per = {}
    for t in J.DELAYS_US:
        pz, px = J.model_rates(t, 0.05, 0.12, 250.0, 120.0)
        n = 10**7
        per[t] = [pz * 2 * n, px * n, n]
    fit = J.joint_fit(per)
    assert fit["T1"] == pytest.approx(250.0, rel=1e-4)
    assert fit["T_phi"] == pytest.approx(120.0, rel=1e-4)
    assert fit["r0_ZZ"] == pytest.approx(0.05, rel=1e-4)
