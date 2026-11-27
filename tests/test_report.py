import math

import pytest

from qang import probability as pb
from qang.core import qg_s_from_qg_z
from qang.qga import qga_from_p0, qga_from_radians
from qang.report import REPORT_FIELDS, qang_report


def test_report_canonical_fields_and_exact_readings():
    r = qang_report({"0": 750, "1": 250}, 250.0, 1000)

    assert tuple(r) == REPORT_FIELDS
    assert r["p0"] == pytest.approx(0.75)
    assert r["amplitude_0"] == pytest.approx(math.sqrt(0.75))
    assert r["amplitude_1"] == pytest.approx(0.5)
    assert r["qg_Z"] == pytest.approx(0.5)
    assert r["m_qg"] == pytest.approx(500.0)
    assert r["qga"] == pytest.approx(qga_from_p0(0.75))
    assert r["qga"] == pytest.approx(1000.0 / 6.0)
    assert r["theta_rad"] == pytest.approx(math.pi / 3.0)
    assert r["phi_qga"] is None
    assert r["qg_S"] == pytest.approx(qg_s_from_qg_z(0.5))
    assert r["N"] == 1000
    assert r["sigma_m"] == pytest.approx(1000.0 / math.sqrt(1000.0))

    lo, hi = r["ci_95"]
    assert lo < r["qga"] < hi
    assert lo == pytest.approx(qga_from_radians(math.pi / 3.0 - 1.96 / math.sqrt(1000.0)))
    assert hi == pytest.approx(qga_from_radians(math.pi / 3.0 + 1.96 / math.sqrt(1000.0)))


@pytest.mark.parametrize("phi", [0.0, 125.0, 999.0])
def test_report_echoes_declared_azimuth(phi):
    r = qang_report({"0": 750, "1": 250}, 250.0, 1000, phi_qga=phi)
    assert r["phi_qga"] == pytest.approx(phi)


@pytest.mark.parametrize("phi", [-1.0, 1000.0, math.nan, math.inf, True, "125"])
def test_report_rejects_invalid_azimuth(phi):
    with pytest.raises(ValueError, match="phi_qga"):
        qang_report({"0": 750, "1": 250}, 250.0, 1000, phi_qga=phi)


def test_report_honours_sdk_qubit_order():
    counts = {"10": 600, "01": 400}
    r_qiskit = qang_report(counts, 500.0, 1000, sdk="qiskit")
    r_cirq = qang_report(counts, 500.0, 1000, sdk="cirq")

    assert r_qiskit["qg_Z"] == pytest.approx(0.2)
    assert r_cirq["qg_Z"] == pytest.approx(-0.2)


def test_report_poles_and_weight_filter():
    north = qang_report({(0,): 1000}, 0.0, 1000)
    south = qang_report({(1,): 1000}, 500.0, 1000)

    assert north["amplitude_0"] == pytest.approx(1.0)
    assert north["amplitude_1"] == pytest.approx(0.0)
    assert north["theta_rad"] == pytest.approx(0.0)
    assert north["qg_Z"] == pytest.approx(1.0)
    assert north["m_qg"] == pytest.approx(1000.0)
    assert north["sigma_m"] == pytest.approx(0.0)
    assert north["ci_95"][0] == pytest.approx(0.0)
    assert south["amplitude_0"] == pytest.approx(0.0)
    assert south["amplitude_1"] == pytest.approx(1.0)
    assert south["theta_rad"] == pytest.approx(math.pi)
    assert south["qg_Z"] == pytest.approx(-1.0)
    assert south["m_qg"] == pytest.approx(-1000.0)
    assert south["ci_95"][1] == pytest.approx(500.0)

    filtered = qang_report(
        {"00": 500, "01": 300, "10": 200},
        250.0,
        500,
        sdk="cirq",
        weight=1,
    )
    assert filtered["N"] == 500
    assert filtered["p0"] == pytest.approx(0.6)


def test_report_rejects_inconsistent_or_ambiguous_inputs():
    with pytest.raises(ValueError, match="n_shots"):
        qang_report({"0": 750, "1": 250}, 250.0, 999)
    with pytest.raises(ValueError, match="sdk"):
        qang_report({"01": 600, "00": 400}, 250.0, 1000)
    with pytest.raises(ValueError, match="no shot has weight"):
        qang_report({(0, 0): 1000}, 250.0, 1000, weight=1)


def test_probability_aliases():
    assert pb.m_qg_from_p0 is pb.milliqang_from_p0
    assert pb.sigma_m is pb.shot_noise_milliqang
    assert pb.qg_tau is pb.dissipative_deficit


def test_gate_aliases():
    pytest.importorskip("qiskit")
    from qang import qiskit_gate

    assert qiskit_gate.R_qg is qiskit_gate.RQangGate

    pytest.importorskip("cirq")
    from qang import cirq_gate

    assert cirq_gate.R_qg is cirq_gate.rqang_gate

    pytest.importorskip("pennylane")
    from qang import pennylane_gate

    assert pennylane_gate.R_qg is pennylane_gate.rqang


@pytest.mark.parametrize(
    "module",
    [
        "qang.qscramble",
        "qang.postselection",
        "qang.qctc",
        "qang.qstring",
    ],
)
def test_analogy_only_demarcation(module):
    import importlib

    pytest.importorskip("jax")
    assert "ANALOGY ONLY" in importlib.import_module(module).__doc__


def test_pqc_is_explicitly_against():
    pytest.importorskip("jax")
    from qang import qdemarc

    assert "AGAINST (direct)" in qdemarc.__doc__
    assert "Do not apply ``qang_report``" in qdemarc.__doc__
