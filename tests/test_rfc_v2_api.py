"""
Tests for the three RFC v2 additions: the signed Cirq gate, the unified
qg_Z estimator and the sector-exposure analysis pass.
"""

import math

import numpy as np
import pytest

from qang.statistics import bayes_qg_estimate, delta_qg_estimate, qg_estimate, wilson_qg_estimate


def test_qg_estimate_dispatch_matches_individual_estimators():
    assert qg_estimate(30, 100) == bayes_qg_estimate(30, 100, 0.95)
    w = qg_estimate(30, 100, method="wilson")
    assert abs(w.low - wilson_qg_estimate(30, 100, 1.959964).low) < 1e-6
    d = qg_estimate(30, 100, method="delta", confidence=0.99)
    assert abs(d.high - delta_qg_estimate(30, 100, 2.575829).high) < 1e-6


def test_qg_estimate_intervals_at_a_pole():
    for m in ("bayes", "wilson"):
        e = qg_estimate(50, 50, method=m)
        assert e.high <= 1.0 + 1e-12 and e.low < 1.0  # informative at the pole
    d = qg_estimate(50, 50, method="delta")
    assert d.low == d.high == 1.0  # the delta method collapses


def test_qg_estimate_rejects_bad_input():
    with pytest.raises(ValueError):
        qg_estimate(3, 10, method="nope")
    with pytest.raises(ValueError):
        qg_estimate(3, 10, confidence=1.5)


def test_signed_cirq_gate_matches_qiskit():
    cirq = pytest.importorskip("cirq")
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    from qang.cirq_gate import signed_rqang_gate
    from qang.qiskit_gate import SignedRQangGate

    for q in (-0.8, 0.0, 0.3, 0.95):
        for s in (1, -1):
            uc = cirq.unitary(signed_rqang_gate(q, s))
            uq = Operator(SignedRQangGate(q, s).definition).data
            assert np.allclose(uc, uq, atol=1e-12)
            psi = uc[:, 0]
            z = abs(psi[0]) ** 2 - abs(psi[1]) ** 2
            x = 2 * np.real(np.conj(psi[0]) * psi[1])
            assert abs(z - q) < 1e-12 and abs(x - s * math.sqrt(1 - q * q)) < 1e-12
    with pytest.raises(ValueError):
        signed_rqang_gate(0.2, 0)


def test_sector_exposure_pass():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.transpiler import PassManager

    from qang.sectors import SectorExposurePass, sector_exposure

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)  # Bell state: all population outside the weight-1 sector
    pm = PassManager([SectorExposurePass(weight=1)])
    out = pm.run(qc)
    assert out == qc  # analysis only
    res = pm.property_set["sector_exposure"]
    assert res == sector_exposure(qc, 1)
    assert abs(res["mean"] - 1.0) < 1e-12
