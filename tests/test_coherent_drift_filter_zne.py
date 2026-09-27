"""
Tests for examples/coherent_drift_filter_zne.py (RESEARCH_NOTES §48):
native circuits are exact, folding is blind to coherent MS over-rotation,
ZNE passes the coherent error through and the qg filter removes most of it.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ionq")

from qiskit.quantum_info import Statevector  # noqa: E402

import chemistry_qg_symmetry_witness as CH  # noqa: E402
import coherent_drift_filter_zne as C  # noqa: E402


def test_native_circuits_reproduce_the_ideal_distribution():
    t = CH.optimal_angle()
    for label, ops in C.h2_native():
        qc = CH._measure_circuit(t, label, 0)
        qc.remove_final_measurements()
        assert np.allclose(C.run(ops, 4, 0.0, 0.0, 0.0), Statevector(qc).probabilities(), atol=1e-12)


@pytest.mark.parametrize("eps", (0.0, 0.03, -0.05))
def test_folding_is_blind_to_coherent_over_rotation(eps):
    _, ops = C.h2_native()[0]
    p1 = C.run(ops, 4, eps, 0.0, 0.0)
    for s in (3, 5):
        assert np.allclose(C.run(C.fold(ops, s), 4, eps, 0.0, 0.0), p1, atol=1e-12)


def _methods(eps):
    per = [C.h2_energies([eps] * len(C.h2_native()), s) for s in C.SCALES]
    raw = [r for r, _ in per]
    flt = [f for _, f in per]
    return raw, flt, float(C.RICHARDSON @ raw), float(C.RICHARDSON @ flt)


def test_zne_passes_the_coherent_error_through():
    raw0, _, zne0, _ = _methods(0.0)
    raw5, _, zne5, _ = _methods(0.05)
    coh_raw, _ = C.coherent_only(0.05)
    assert coh_raw == pytest.approx(2.97, abs=0.05)
    assert zne5 - zne0 == pytest.approx(coh_raw, abs=0.1)
    assert raw0[0] < raw0[1] < raw0[2]


def test_filter_removes_most_of_the_coherent_error():
    coh_raw, coh_filt = C.coherent_only(0.05)
    assert coh_filt < 0.1 * coh_raw
    _, flt0, _, zf0 = _methods(0.0)
    _, flt5, _, zf5 = _methods(0.05)
    assert flt5[0] - flt0[0] < 0.3
    assert zf5 < 2.0 and zf0 < 2.0


def test_drift_widens_zne_more_than_the_filter():
    d = C.h2_drift(0.02, 0.02, reps=30, seed=1)
    assert d["zne"]["std"] > 3 * d["filter"]["std"]
    assert d["filter"]["mean"] == pytest.approx(5.67, abs=0.1)
