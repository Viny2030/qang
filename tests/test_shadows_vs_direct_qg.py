"""
Tests for examples/shadows_vs_direct_qg.py (RESEARCH_NOTES §62): classical
shadows vs direct measurement for the qg quantities.
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit_aer")

import shadows_vs_direct_qg as S  # noqa: E402


def _random_rho(seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(64, 4)) + 1j * rng.normal(size=(64, 4))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def test_shadow_variance_formula_is_exact():
    rho = _random_rho()
    settings, P = S.all_settings(rho)
    for label in ({0: "Z"}, {1: "X", 3: "Y"}, {0: "Z", 2: "Z", 5: "X"}):
        e = S.pauli_expect(rho, label)
        est = np.zeros_like(P)
        for si, st in enumerate(settings):
            if all(st[q] == b for q, b in label.items()):
                est[si] = 3 ** len(label) * np.prod(S._SIGN[:, list(label)], axis=1)
        w = P / len(settings)
        assert (w * est).sum() == pytest.approx(e, abs=1e-10)
        assert (w * est**2).sum() - e**2 == pytest.approx(S.shadow_var_pauli(e, len(label)), abs=1e-9)


def test_l18_is_balanced():
    ca = S.orthogonal_array(6)
    assert len(ca) == 18
    for i, j in itertools.combinations(range(6), 2):
        for a in "XYZ":
            for b in "XYZ":
                assert sum(1 for s in ca if s[i] == a and s[j] == b) == 2


def test_xxz_ratios():
    rho = S.xxz_state()
    r = S.tasks_single(rho)
    assert r["T1 all qg_Z"] == pytest.approx(3.0, abs=0.05)
    assert r["T2 register-mean qg_Z"] > 5
    assert r["T3 all ZZ"] == pytest.approx(9.0, abs=0.2)
    assert r["T4 all 3n single-qubit qg"] == pytest.approx(1.0, abs=0.02)
    assert r["T5 all 2-qubit correlators"] == pytest.approx(1.0, abs=0.02)
    import xxz_trotter_filter_qg as X

    imb, Nn = X._tables(6)
    t6 = S.filtered_ratio(rho, imb, Nn == 3)
    assert t6["shadow_mean"] == pytest.approx(t6["filtered"], abs=1e-10)
    assert t6["ratio"] > 20


def test_lih_energy_grouping_beats_shadows():
    rho, ham = S.lih_state()
    t7 = S.energy_ratio(rho, ham)
    assert t7["ratio"] > 2
