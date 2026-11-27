"""tools/verify_cap47.py — Cap. 47: la cadena de medición auditable.

Comprueba, con primitivas ya existentes y sin nueva API, que el orden de bits,
la lectura, el intervalo, la covarianza y el prerregistro declarado forman una
cadena reproducible. No mide hardware ni afirma ganancia alguna.

    PYTHONPATH=. python tools/verify_cap47.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from qang.interop import canonical_counts
from qang.probability import analyze, readings_from_p0
from qang.statistics import (
    direction_estimate,
    qg_covariance,
    qg_estimate,
    qg_s_estimate,
    radius2_estimate,
    register_witness_variance,
)

ROOT = Path(__file__).resolve().parent.parent


def check_prereg() -> None:
    """Los cinco campos del prerregistro están declarados en las dos ediciones."""
    fields = {
        "es": ("orden de bits", "estimador", "nivel de confianza", "bases", "fases"),
        "en": ("bit order", "estimator", "confidence level", "bases", "phases"),
    }
    for name in ("es", "en"):
        path = ROOT / "book" / name / "qjax47.qmd"
        text = path.read_text(encoding="utf-8")
        for field in fields[name]:
            assert field in text, f"{name}: falta {field!r} en el prerregistro"


def check_counts() -> None:
    """El mismo diccionario crudo respeta el orden de cada SDK."""
    raw = {"01": 300, "10": 700}
    qiskit = canonical_counts(raw, sdk="qiskit", n_qubits=2)
    cirq = canonical_counts(raw, sdk="cirq", n_qubits=2)
    assert qiskit == {(1, 0): 300, (0, 1): 700}, qiskit
    assert cirq == {(0, 1): 300, (1, 0): 700}, cirq
    assert sum(qiskit.values()) == sum(cirq.values()) == 1000


def check_reading() -> None:
    """Una lectura y su intervalo salen de la misma probabilidad."""
    reading = analyze(750, 1000, target_qg_z=0.4)
    direct = readings_from_p0(0.75)
    assert abs(reading.p0[0] - 0.75) < 1e-12
    assert abs(reading.qg_z[0] - 0.5) < 1e-12
    assert abs(reading.milliqang[0] - 500.0) < 1e-12
    assert abs(reading.theta[0] - math.acos(0.5)) < 1e-12
    assert reading.p0[1] < 0.75 < reading.p0[2]
    assert abs(reading.deviation_milliqang - 100.0) < 1e-12
    assert abs(direct["qg_z"] - reading.qg_z[0]) < 1e-12
    assert abs(direct["theta"] - reading.theta[0]) < 1e-12


def check_intervals() -> None:
    """Cada estimador declara su método; ninguno inventa una confianza universal."""
    methods = {}
    for method in ("bayes", "wilson", "delta"):
        estimate = qg_estimate(750, 1000, method=method)
        methods[method] = estimate
        assert estimate.low <= estimate.qg_z <= estimate.high
        assert abs(estimate.qg_z - 0.5) < 0.05
    assert methods["wilson"].method == "wilson"
    assert methods["delta"].method == "delta"
    pole_wilson = qg_estimate(1000, 1000, method="wilson")
    pole_delta = qg_estimate(1000, 1000, method="delta")
    assert pole_wilson.low < pole_delta.low == pole_delta.high == 1.0
    s_est, s_low, s_high = qg_s_estimate(750, 1000)
    assert s_low <= s_est <= s_high
    assert abs(s_est - (-(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25)))) < 0.02


def check_covariance() -> None:
    """La covarianza usa los mismos disparos y conserva las correlaciones."""
    probs = np.array([0.5, 0.0, 0.0, 0.5])
    q, cov = qg_covariance(probs, 2, 1000)
    assert q.shape == (2,) and cov.shape == (2, 2)
    assert np.allclose(q, 0.0)
    assert np.allclose(cov, np.full((2, 2), 0.001))
    full = register_witness_variance(probs, 2, 1000)
    naive = register_witness_variance(probs, 2, 1000, independent=True)
    assert abs(full - 0.001) < 1e-12
    assert abs(naive - 0.0005) < 1e-12
    assert full > naive


def check_multi_basis() -> None:
    """Radio y dirección declaran las tres bases que las producen."""
    counts = [(750, 1000), (500, 1000), (500, 1000)]
    r2, error = radius2_estimate(counts)
    theta, r2_again = direction_estimate(*counts)
    assert 0.0 < r2 <= 1.0 and error > 0.0
    assert abs(r2 - r2_again) < 1e-12
    assert 0.0 <= theta <= math.pi


def main() -> None:
    """Ejecuta los seis bloques y reporta la cadena completa."""
    check_prereg()
    check_counts()
    check_reading()
    check_intervals()
    check_covariance()
    check_multi_basis()
    print("[1] prerregistro ES/EN: 5 campos declarados antes de la medición")
    print("[2] canonical_counts: qiskit (1,0)/(0,1), cirq (0,1)/(1,0), 1000 disparos")
    print("[3] analyze/readings_from_p0: p0=0.75, qg_Z=0.5, m-qg=500, theta=acos(0.5)")
    print("[4] qg_estimate: bayes/wilson/delta; en el polo Wilson no colapsa, delta sí")
    print("[5] covarianza: Bell, sigma^2=0.001; testigo 0.001 frente a 0.0005 ingenua")
    print("[6] radio/direccion: tres bases declaradas, r2 en (0,1], theta en [0,pi]")
    print("verify_cap47: OK")


if __name__ == "__main__":
    main()
