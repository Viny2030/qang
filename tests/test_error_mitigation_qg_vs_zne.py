"""
Tests for examples/error_mitigation_qg_vs_zne.py: which noise the qg
symmetry filter fixes, which ZNE fixes, and the qg witness that tells
them apart.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("scipy")

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402

from error_mitigation_qg_vs_zne import RICHARDSON, SCALES, fold_cx, mitigated_energies  # noqa: E402
from chemistry_qg_symmetry_witness import FCI_ENERGY, ansatz  # noqa: E402

SEEDS = (11, 12, 13)


def _mean_errors(noise, delay_us=0.0):
    runs = [mitigated_energies(noise, delay_us, 20000, s) for s in SEEDS]
    out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    for k in ("raw", "qg_filter", "zne", "zne_qg"):
        out[k] = 1e3 * (out[k] - FCI_ENERGY)
    return out


def test_folding_keeps_the_unitary_and_multiplies_cx():
    qc = ansatz(0.3)
    for s in SCALES:
        f = fold_cx(qc, s)
        assert Operator(f).equiv(Operator(qc))
        assert f.count_ops().get("cx", 0) == s * qc.count_ops()["cx"]
    with pytest.raises(ValueError):
        fold_cx(QuantumCircuit(1), 2)


def test_richardson_weights_are_exact_for_quadratics():
    assert RICHARDSON.sum() == pytest.approx(1.0)
    lam = np.array(SCALES, dtype=float)
    for poly in ([1.0, 0.0, 0.0], [0.3, -2.0, 0.7]):
        vals = poly[0] + poly[1] * lam + poly[2] * lam**2
        assert RICHARDSON @ vals == pytest.approx(poly[0])


def test_dephasing_is_invisible_to_the_witness_and_to_the_filter_but_not_to_zne():
    r = _mean_errors(("dephasing", 0.10))
    assert r["kept_fraction"] == pytest.approx(1.0)
    assert abs(r["mean_qg_z"]) < 0.01
    assert r["qg_filter"] == pytest.approx(r["raw"])
    assert r["raw"] > 6 and abs(r["zne"]) < 6


def test_t1_is_flagged_by_the_witness_and_removed_by_the_filter():
    r = _mean_errors(("T1", 0.10))
    assert r["mean_qg_z"] > 0.08 and r["kept_fraction"] < 0.85
    assert r["raw"] > 100
    assert abs(r["qg_filter"]) < 15  # >= 88% of the error removed, no extra circuits


def test_strong_depolarizing_needs_both_methods():
    r = _mean_errors(("depolarizing", 0.10))
    assert r["raw"] > 150 and r["qg_filter"] > 50
    assert abs(r["zne_qg"]) < abs(r["zne"])


def test_idle_t1_on_fake_brisbane_defeats_zne_but_not_the_filter():
    pytest.importorskip("qiskit_ibm_runtime")
    r = _mean_errors("brisbane", 50.0)
    assert r["mean_qg_z"] > 0.1
    assert r["zne"] > 150
    assert r["qg_filter"] < 45 and r["zne_qg"] < 45
