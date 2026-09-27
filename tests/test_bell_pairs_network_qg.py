"""
Tests for examples/bell_pairs_network_qg.py (RESEARCH_NOTES §57).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

import bell_pairs_network_qg as B  # noqa: E402


@pytest.mark.parametrize("kind", ("dephasing", "bit flip", "Y flip", "depolarizing", "amplitude damping"))
def test_fidelity_from_three_correlations_is_exact(kind):
    rho = B.noisy_pair(kind, 0.13)
    c = B.correlations(rho)
    assert B.fidelity_from_qg(*c[:3]) == pytest.approx(float(np.real(B.PHI_PLUS @ rho @ B.PHI_PLUS)))
    assert sum(B.bell_weights(*c[:3]).values()) == pytest.approx(1.0)


def test_signatures():
    assert B.correlations(B.noisy_pair("dephasing", 0.1))[2] == pytest.approx(1.0)
    assert B.correlations(B.noisy_pair("bit flip", 0.1))[0] == pytest.approx(1.0)
    assert B.correlations(B.noisy_pair("Y flip", 0.1))[1] == pytest.approx(-1.0)
    for kind in ("dephasing", "bit flip", "depolarizing"):
        assert B.correlations(B.noisy_pair(kind, 0.1))[3] == pytest.approx(0.0, abs=1e-12)
    assert B.correlations(B.noisy_pair("amplitude damping", 0.2))[3] == pytest.approx(0.2)


def test_textbook_slot_fails_for_y_flip_and_rule_fixes_it():
    w = B.bell_weights(*B.correlations(B.noisy_pair("Y flip", 0.1))[:3])
    assert B.dejmps(w, "Psi-")[0] < w["Phi+"]
    assert B.dejmps(w, B.best_slot(w))[0] == pytest.approx(0.954, abs=1e-3)
    d = B.distillation_table({"a": ("amplitude damping", 0.2), "y": ("Y flip", 0.1)}, trials=300)
    assert d["y"]["qg_right"] == 1.0 and d["a"]["qg_right"] > 0.85


def test_aged_closed_forms():
    for t, r in ((0.3, 1.0), (0.7, 2.0), (1.2, 0.5)):
        assert np.allclose(B.correlations(B.aged_pair(t, r))[:4], B.aged_correlations_closed(t, r), atol=1e-12)


def test_distillation_extends_the_cutoff():
    assert B.cutoff(2.0) == pytest.approx(0.184, abs=2e-3)
    assert B.cutoff(2.0, True) == pytest.approx(0.395, abs=2e-3)
    assert B.cutoff(0.2, True) > B.cutoff(0.2)
