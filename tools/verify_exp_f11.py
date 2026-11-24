"""F11 — materia y antimateria: conjugación de carga como mapa antípoda en el qubit de Dirac.

Hipótesis H11 (`unverified`): la asimetría bariónica es una populación y no una fase; la firma angular de la conjugación CP en un qubit efectivo es un desplazamiento de 500 qga en el azimut y el complemento

    p0 -> 1 - p0,   theta_qga -> 500 - theta_qga,   phi_qga -> phi_qga + 500  (mod 1000).

Teorema T11 (`exact`): ``qang.qanti.conjugate(psi) = sigma_y psi*`` lleva el vector de Bloch a ``-r``, con ``self_conjugate_overlap = 0``, y la holonomía de un lazo conjugado es el complejo conjugado de la original.

Circuito testigo: 1 qubit, theta_qga = 125 qga, phi_qga = 125 qga, 65_536 shots, 600 estimaciones, umbral 5 sigma = 3.108 qga y control +50 qga; decisiones con ``qang.qfalsify``.
"""
from __future__ import annotations

import cmath
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qang.geometric import geometric_phase_qg, pancharatnam_phase, state_from_bloch
from qang.qanti import bloch_mqg, conjugate, self_conjugate_overlap
from qang.qecho import invert_readout, readout_affine
from qang.qfalsify import verify_registration
from qang.qga import (
    QGA_PER_TURN,
    azimuth_qga,
    p0_from_qga,
    qga_from_p0,
    radians_from_qga,
)
from tools.verify_exp_common import (
    DELTA_5SIGMA_QGA,
    E01,
    E10,
    N_ESTIMATES,
    SEED,
    SHOTS,
    SIGMA_QGA,
    circular_distance_qga,
    emit,
    falsify_angular,
    noisy_phase_samples,
    phase_qga_from_complex,
)

THETA_QGA: float = 125.0
PHI_QGA: float = 125.0
THETA_CONJUGATE_QGA: float = 500.0 - THETA_QGA
PHI_CONJUGATE_QGA: float = (PHI_QGA + QGA_PER_TURN / 2.0) % QGA_PER_TURN
CONTROL_OFFSET_QGA: float = 50.0

PREDICTION: dict = {
    "exp": "F11",
    "domain": "materia_antimateria",
    "map": "qanti.conjugate: psi -> sigma_y psi*; Bloch r -> -r",
    "theta_qga": THETA_QGA,
    "phi_qga": PHI_QGA,
    "theta_conjugate_qga": THETA_CONJUGATE_QGA,
    "phi_conjugate_qga": PHI_CONJUGATE_QGA,
    "shots": SHOTS,
    "n_estimates": N_ESTIMATES,
    "delta_5sigma_qga": DELTA_5SIGMA_QGA,
    "control_offset_qga": CONTROL_OFFSET_QGA,
    "seed": SEED,
    "criterion": "600/600 hits and control refutada for polar and azimuth, matter and conjugate",
}
REGISTRATION_SHA256: str = "6ef2f20b7f9f2bf1441f0be65dbd65d2cdee4bc72f1ffb3f478efffc74185735"


def state_from_angles(theta_qga: float, phi_qga: float) -> np.ndarray:
    """Estado puro de un qubit con angulos polares y azimutales en qga."""
    theta = radians_from_qga(theta_qga)
    phi = radians_from_qga(phi_qga)
    return np.array([math.cos(theta / 2.0), cmath.exp(1j * phi) * math.sin(theta / 2.0)], dtype=complex)


def cone_holonomy(theta: float) -> tuple[complex, np.ndarray]:
    """Holonomia de un cono cerrado y la lista de estados que la define."""
    qg_z = math.cos(theta)
    path = [
        (math.sin(theta) * math.cos(a), math.sin(theta) * math.sin(a), qg_z)
        for a in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
    ]
    states = [state_from_bloch(r) for r in path]
    return geometric_phase_qg(path).value, states


def main() -> None:
    """Ejecuta F11 y emite filas con etiqueta de evidencia."""
    assert verify_registration(PREDICTION, REGISTRATION_SHA256)

    rng = np.random.default_rng(SEED)
    psi = state_from_angles(THETA_QGA, PHI_QGA)
    psi_conjugate = conjugate(psi)
    bloch = bloch_mqg(psi) / QGA_PER_TURN
    bloch_conjugate = bloch_mqg(psi_conjugate) / QGA_PER_TURN

    p0 = p0_from_qga(THETA_QGA)
    p0_conjugate = p0_from_qga(THETA_CONJUGATE_QGA)
    azimuth_matter = azimuth_qga(float(bloch[0]), float(bloch[1]))
    azimuth_conjugate = azimuth_qga(float(bloch_conjugate[0]), float(bloch_conjugate[1]))

    theta = radians_from_qga(THETA_QGA)
    holonomy_matter, states_matter = cone_holonomy(theta)
    holonomy_conjugate, states_conjugate = cone_holonomy(math.pi - theta)

    geometric_matter = phase_qga_from_complex(holonomy_matter)
    geometric_conjugate = phase_qga_from_complex(holonomy_conjugate)
    pancharatnam_matter = float((pancharatnam_phase(states_matter) / (2.0 * math.pi) * QGA_PER_TURN) % QGA_PER_TURN)
    pancharatnam_conjugate = float((pancharatnam_phase(states_conjugate) / (2.0 * math.pi) * QGA_PER_TURN) % QGA_PER_TURN)

    m = QGA_PER_TURN * float(bloch[2])
    m_recovered = invert_readout(readout_affine(m, E01, E10), E01, E10)

    rows: list[tuple[str, object, str]] = [
        ("H11_claim", "asimetria barionica es poblacion, no fase angular", "unverified"),
        ("T11_map", "conjugate = sigma_y psi*; Bloch r -> -r", "exact"),
        ("T11_self_conjugate_overlap", round(self_conjugate_overlap(psi), 12), "exact"),
        ("T11_bloch_conjugate", np.round(bloch_conjugate, 9).tolist(), "exact"),
        ("T11_p0_matter", round(p0, 9), "exact"),
        ("T11_p0_conjugate", round(p0_conjugate, 9), "exact"),
        ("T11_theta_conjugate_qga", THETA_CONJUGATE_QGA, "exact"),
        ("T11_azimuth_conjugate_qga", PHI_CONJUGATE_QGA, "exact"),
        ("T11_geometric_matter_qga", round(geometric_matter, 6), "exact"),
        ("T11_geometric_conjugate_qga", round(geometric_conjugate, 6), "exact"),
        ("T11_pancharatnam_matter_qga", round(pancharatnam_matter, 6), "exact"),
        ("T11_pancharatnam_conjugate_qga", round(pancharatnam_conjugate, 6), "exact"),
        ("T11_readout_inversion_m_qg", round(m_recovered, 9), "exact"),
        ("registration_sha256", REGISTRATION_SHA256[:16], "exact"),
        ("shots", SHOTS, "exact"),
        ("sigma_qga", round(SIGMA_QGA, 6), "exact"),
        ("5sigma_qga", DELTA_5SIGMA_QGA, "exact"),
    ]

    cases = (
        ("polar_matter", THETA_QGA, np.asarray([qga_from_p0(float(k) / SHOTS) for k in rng.binomial(SHOTS, p0, size=N_ESTIMATES)])),
        ("polar_conjugate", THETA_CONJUGATE_QGA, np.asarray([qga_from_p0(float(k) / SHOTS) for k in rng.binomial(SHOTS, p0_conjugate, size=N_ESTIMATES)])),
        ("azimuth_matter", PHI_QGA, noisy_phase_samples(PHI_QGA / QGA_PER_TURN, rng)),
        ("azimuth_conjugate", PHI_CONJUGATE_QGA, noisy_phase_samples(PHI_CONJUGATE_QGA / QGA_PER_TURN, rng)),
    )
    for name, prediction, samples in cases:
        verdict, hits, p_value = falsify_angular(prediction, samples)
        control_verdict, control_hits, control_p = falsify_angular(
            (prediction + CONTROL_OFFSET_QGA) % QGA_PER_TURN, samples
        )
        rows.extend(
            [
                (f"mean_{name}", round(float(np.mean(samples)), 6), "simulado"),
                (f"hits_{name}", f"{hits}/{N_ESTIMATES}", "simulado"),
                (f"p_upper_{name}", p_value, "simulado"),
                (f"decision_{name}", verdict, "simulado"),
                (f"control_{name}", f"{control_hits}/{N_ESTIMATES} {control_verdict}", "simulado"),
                (f"control_p_{name}", control_p, "simulado"),
            ]
        )
        assert hits == N_ESTIMATES
        assert control_hits == 0
        assert verdict == "confirmada" and control_verdict == "refutada"

    emit("verify_exp_f11 — materia/antimateria / conjugacion como mapa antipoda", rows)

    assert abs(float(np.max(np.abs(bloch_conjugate + bloch)))) < 1e-12
    assert self_conjugate_overlap(psi) < 1e-12
    assert abs(p0_conjugate - (1.0 - p0)) < 1e-12
    assert abs(qga_from_p0(p0) - THETA_QGA) < 1e-9
    assert abs(qga_from_p0(p0_conjugate) - THETA_CONJUGATE_QGA) < 1e-9
    assert circular_distance_qga(azimuth_matter, PHI_QGA) < 1e-9
    assert circular_distance_qga(azimuth_conjugate, PHI_CONJUGATE_QGA) < 1e-9
    assert circular_distance_qga(azimuth_conjugate, (PHI_QGA + 500.0) % QGA_PER_TURN) < 1e-9
    assert abs(phase_qga_from_complex(cmath.exp(2j * math.pi * PHI_QGA / QGA_PER_TURN)) - PHI_QGA) < 1e-9
    assert abs(geometric_matter - pancharatnam_matter) < 1e-9
    assert abs(geometric_conjugate - pancharatnam_conjugate) < 1e-9
    assert circular_distance_qga(geometric_conjugate, (QGA_PER_TURN - geometric_matter) % QGA_PER_TURN) < 1e-9
    assert abs(m_recovered - m) < 1e-9
    print("verify_exp_f11: OK")


if __name__ == "__main__":
    main()
