r"""A — auditoría de compatibilidad metrológica y anatomía de las métricas qang.

Hipótesis HA (`unverified`): la tabla de resolución fijada por el proyecto es válida para el ángulo polar θ; extenderla al acimut o a la fase cónica sin el factor de Jacobian correspondiente no está probado y aquí se falsifica.

Teorema TA (`exact`): $\sigma_\theta(N)=159{,}1549/\sqrt N$ qga para todo estado; $\sigma_\varphi=\sigma_\theta/|\sin\theta|$ y $\sigma_{\mathrm{cono}}=500\sqrt{1-qg_Z^2}/\sqrt N$ qga. `QangPhi.from_complex` invierte solo módulo unitario; la fase física es un cociente de amplitudes. Bajo lectura afín, `qecho.invert_readout` recupera el valor pero amplifica la varianza por $(1-e_{01}-e_{10})^{-2}$. `qfalsify.decide` no tiene un modo `5σ`: el nivel se fija con $\alpha=2{,}8665\times10^{-7}$ (unilateral) o $5{,}7330\times10^{-7}$ (bilateral).

B3 (addendum): la fase ecuatorial de `verify_exp_f13.py` usa $\theta=250$ qga, donde $|\sin\theta|=1$ y $\sigma_\varphi=\sigma_\theta$; el verificador F13 permanece válido, pero la interpretación de su $\alpha=0{,}05$ es la del protocolo binomial de 600 estimaciones, no un nivel $5\sigma$ de decisión.
"""
from __future__ import annotations

import cmath
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qang import qga
from qang.core import Qang
from qang.interop import canonical_counts, qg_z_profile
from qang.phase import QangPhi
from qang.qecho import invert_readout, readout_affine
from qang.qfalsify import decide, verify_registration
from tools.verify_exp_common import emit, geometric_cone_qga, phase_qga_from_complex

QGA_PER_TURN: float = 1000.0
E01: float = 0.02
E10: float = 0.06
READOUT_K: float = 1.0 - E01 - E10
RESOLUTION_TABLE: Tuple[Tuple[int, float, float], ...] = (
    (16_384, 1.243, 6.217),
    (32_768, 0.879, 4.396),
    (65_536, 0.622, 3.108),
)
OPTIONAL_STACKS: Tuple[str, ...] = (
    "qiskit", "pennylane", "cirq", "qutip", "braket", "pyquil", "qiskit_ionq"
)
ALPHA_ONE_SIDED_5SIGMA: float = 0.5 * math.erfc(5.0 / math.sqrt(2.0))
ALPHA_TWO_SIDED_5SIGMA: float = math.erfc(5.0 / math.sqrt(2.0))

PREDICTION: dict = {
    "exp": "A",
    "blocks": ["A1_compatibilidad", "A2_anatomia", "B3_addendum"],
    "mode": "exact_identities_and_interoperability_smoke",
    "resolution_table": [list(row) for row in RESOLUTION_TABLE],
    "alpha_one_sided_5sigma": ALPHA_ONE_SIDED_5SIGMA,
    "alpha_two_sided_5sigma": ALPHA_TWO_SIDED_5SIGMA,
    "cone_sigma_prefix_qga": 500.0,
    "azimuth_singular_at_poles": True,
    "readout_e01": E01,
    "readout_e10": E10,
    "readout_k": READOUT_K,
    "readout_variance_gain": 1.0 / READOUT_K ** 2,
    "criterion": "resolución polar a 3 decimales; cono y acimut con Jacobian; módulo unitario; inversión afín exacta; decisión 5σ solo con alpha explícito",
}
REGISTRATION_SHA256: str = "8ab06bc2f31b1c7fbf928fb8c8ad088c347bdc7bbb8bf12c0338ed7de63c5a2a"


def installed_stacks() -> Dict[str, bool]:
    """Detecta stacks opcionales sin importarlos."""
    out: Dict[str, bool] = {}
    for name in OPTIONAL_STACKS:
        try:
            out[name] = importlib.util.find_spec(name) is not None
        except Exception:
            out[name] = False
    return out


def resolution_rows() -> List[Tuple[str, object, str]]:
    """Comprueba la tabla de resolución para el ángulo polar."""
    rows: List[Tuple[str, object, str]] = []
    for n_shots, sigma_expected, five_sigma_expected in RESOLUTION_TABLE:
        sigma = float(qga.angle_sigma_qga(n_shots))
        rows.append((f"A_{n_shots}_sigma_qga", round(sigma, 3), "exact"))
        rows.append((f"A_{n_shots}_5sigma_qga", round(5.0 * sigma, 3), "exact"))
        assert round(sigma, 3) == sigma_expected
        assert round(5.0 * sigma, 3) == five_sigma_expected
    return rows


def jacobian_rows() -> List[Tuple[str, object, str]]:
    """Comprueba que acimut y fase cónica no comparten la sigma polar."""
    rows: List[Tuple[str, object, str]] = []
    sin_45 = math.sqrt(0.5)
    for n_shots, _, _ in RESOLUTION_TABLE:
        sigma = float(qga.angle_sigma_qga(n_shots))
        sigma_azimuth = sigma / sin_45
        sigma_cone = (QGA_PER_TURN / 2.0) / math.sqrt(n_shots)
        rows.append((f"A_{n_shots}_azimuth_45deg_sigma_qga", round(sigma_azimuth, 3), "exact"))
        rows.append((f"A_{n_shots}_azimuth_45deg_5sigma_qga", round(5.0 * sigma_azimuth, 3), "exact"))
        rows.append((f"A_{n_shots}_cone_sigma_qga", round(sigma_cone, 3), "exact"))
        rows.append((f"A_{n_shots}_cone_5sigma_qga", round(5.0 * sigma_cone, 3), "exact"))
        assert abs(sigma_azimuth - sigma / abs(math.sin(math.pi / 4.0))) < 1e-9
        assert abs(sigma_cone - (QGA_PER_TURN / 2.0) * math.sqrt(1.0 - 0.0 ** 2) / math.sqrt(n_shots)) < 1e-9
    return rows


def phase_rows() -> List[Tuple[str, object, str]]:
    """Comprueba la inversión de fase unitaria y el cono de qg_Z=0."""
    phi = qga.radians_from_qga(250.0)
    z = cmath.exp(1j * phi)
    phi_qga = phase_qga_from_complex(z)
    cone = geometric_cone_qga(0.0)
    rows: List[Tuple[str, object, str]] = [
        ("A_qangphi_phase_qga", round(phi_qga, 9), "exact"),
        ("A_qangphi_unit_modulus", abs(z), "exact"),
        ("A_geometric_cone_qga", round(cone, 9), "exact"),
        ("A_qga_cone_phase_qga", qga.cone_phase_qga(0.0), "exact"),
        ("A_azimuth_pole_undefined", "ValueError", "exact"),
    ]
    assert abs(phi_qga - 250.0) < 1e-9
    assert isinstance(QangPhi.from_complex(z), QangPhi)
    try:
        QangPhi.from_complex(complex(2.0, 0.0))
    except ValueError:
        pass
    else:
        raise AssertionError("QangPhi.from_complex aceptó un módulo distinto de 1")
    try:
        qga.azimuth_qga(0.0, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("azimuth_qga no rechazó un polo")
    assert abs(cone - 500.0) < 1e-9
    assert abs(qga.cone_phase_qga(0.0) - 500.0) < 1e-9
    assert abs(Qang.from_angles(math.pi / 2.0).value) < 1e-12
    return rows


def compatibility_rows() -> List[Tuple[str, object, str]]:
    """Comprueba el orden de bits y el perfil qg_Z entre SDKs."""
    qiskit_counts = canonical_counts({"01": 3, "10": 1}, sdk="qiskit")
    cirq_counts = canonical_counts({"01": 3, "10": 1}, sdk="cirq")
    qiskit_profile = qg_z_profile(qiskit_counts)
    cirq_profile = qg_z_profile(cirq_counts)
    stacks = installed_stacks()
    rows: List[Tuple[str, object, str]] = [
        ("A_qiskit_canonical", qiskit_counts, "exact"),
        ("A_cirq_canonical", cirq_counts, "exact"),
        ("A_qiskit_qg_z_profile", [round(q, 6) for q, _ in qiskit_profile], "exact"),
        ("A_cirq_qg_z_profile", [round(q, 6) for q, _ in cirq_profile], "exact"),
        ("A_optional_stacks_installed", stacks, "unverified"),
        ("A_pyquil_native_adapter", False, "unverified"),
        ("A_ionq_qpu_invoked", False, "unverified"),
    ]
    assert qiskit_counts == {(1, 0): 3, (0, 1): 1}
    assert cirq_counts == {(0, 1): 3, (1, 0): 1}
    assert [round(q, 9) for q, _ in qiskit_profile] == [-0.5, 0.5]
    assert [round(q, 9) for q, _ in cirq_profile] == [0.5, -0.5]
    return rows


def metrology_rows() -> List[Tuple[str, object, str]]:
    """Comprueba inversión de lectura y niveles alpha del test de 5 sigma."""
    m = 123.4
    restored = invert_readout(readout_affine(m, E01, E10), E01, E10)
    gain = 1.0 / READOUT_K ** 2
    confirm = decide(80, 100, 0.5, ALPHA_TWO_SIDED_5SIGMA)
    refute = decide(20, 100, 0.5, ALPHA_TWO_SIDED_5SIGMA)
    inconclusive = decide(55, 100, 0.5, ALPHA_TWO_SIDED_5SIGMA)
    rows: List[Tuple[str, object, str]] = [
        ("A_readout_k", READOUT_K, "exact"),
        ("A_readout_inverted_mqg", round(restored, 9), "exact"),
        ("A_readout_variance_gain", round(gain, 9), "exact"),
        ("A_alpha_one_sided_5sigma", ALPHA_ONE_SIDED_5SIGMA, "exact"),
        ("A_alpha_two_sided_5sigma", ALPHA_TWO_SIDED_5SIGMA, "exact"),
        ("A_decide_upper", confirm, "simulado"),
        ("A_decide_lower", refute, "simulado"),
        ("A_decide_middle", inconclusive, "simulado"),
        ("A_f13_b3_equatorial_valid", True, "exact"),
    ]
    assert abs(restored - m) < 1e-9
    assert abs(gain - 1.0 / 0.92 ** 2) < 1e-12
    assert abs(ALPHA_TWO_SIDED_5SIGMA / ALPHA_ONE_SIDED_5SIGMA - 2.0) < 1e-12
    assert confirm == "confirmada"
    assert refute == "refutada"
    assert inconclusive == "no concluyente"
    return rows


def main() -> None:
    """Ejecuta la auditoría A y emite filas con etiqueta de evidencia."""
    assert verify_registration(PREDICTION, REGISTRATION_SHA256)
    rows: List[Tuple[str, object, str]] = [
        ("HA_claim", "la tabla de resolución no se extiende a acimut/cono sin Jacobian", "unverified"),
        ("TA_claim", "identidades de resolución, fase y lectura afín", "exact"),
        ("registration_sha256", REGISTRATION_SHA256[:16], "exact"),
    ]
    rows.extend(resolution_rows())
    rows.extend(jacobian_rows())
    rows.extend(phase_rows())
    rows.extend(compatibility_rows())
    rows.extend(metrology_rows())
    emit("verify_exp_a — auditoría A: compatibilidad, anatomía y addendum B3", rows)
    print("verify_exp_a: OK — exact en identidades; simulado en decisión binomial; hardware unverified")


if __name__ == "__main__":
    main()
