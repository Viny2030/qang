"""Tests for qang.postselection (§164)."""

import math

import pytest

from qang import postselection as PS

qiskit = pytest.importorskip("qiskit")


def _counts(qc):
    from qiskit.quantum_info import Statevector

    return {k: int(round(v * 1e9)) for k, v in Statevector(qc).probabilities_dict().items()}


@pytest.mark.parametrize("phi", [0.0, math.pi / 3, math.pi / 2, 2 * math.pi / 3, math.pi])
def test_pctc_formulas(phi):
    assert PS.pctc_from_counts(_counts(PS.pctc_circuit(phi, False, measure=False)))["success"] == pytest.approx(
        PS.pctc_success(phi), abs=1e-6)
    r = PS.pctc_from_counts(_counts(PS.pctc_circuit(phi, True, measure=False)))
    assert r["qg_z_S"] == pytest.approx(PS.pctc_controlled_qg_z(phi), abs=1e-6)
    assert math.acos(2 * PS.pctc_success(phi) - 1) == pytest.approx(phi, abs=1e-6)


def test_yoshida_kitaev_decoder():
    from qiskit import QuantumCircuit

    scr = PS.find_scrambler()
    r = PS.yk_from_counts(_counts(PS.yk_decoder_circuit(scr, measure=False)))
    assert r["success"] == pytest.approx(0.25, abs=1e-6) and r["fidelity"] == pytest.approx(1.0, abs=1e-6)
    r = PS.yk_from_counts(_counts(PS.yk_decoder_circuit(QuantumCircuit(3), measure=False)))
    assert r["success"] == pytest.approx(1.0, abs=1e-6) and r["fidelity"] == pytest.approx(0.25, abs=1e-6)
