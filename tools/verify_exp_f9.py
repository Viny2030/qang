"""F9 — regla de Born: verificacion de p0 = cos^2(theta/2) = (1+qg_Z)/2,
de la lectura en m-qg, de la escala qga y de la resolucion angular fijada a
65536 shots en los polos, el ecuador y theta = 125 qga.

Hipotesis H9 (propuesta, `unverified`): la lectura de un qubit descrito por un
solo angulo polar depende solo de la probabilidad de Born p0; los cuatro
registros (theta, qg_Z, qg_S, m-qg) coinciden con las identidades del Cap. 1
y los estimadores por shots caen dentro de 5 sigma para todo estado puro.
Teorema T9 (`exact`): p0(theta) = cos^2(theta/2), qg_Z = 2p0 - 1 = cos(theta),
theta_qga = (1000/pi) arccos(sqrt(p0)) y la informacion de Fisher de un shot Z
es 1 por radian^2, o (2*pi/1000)^2 por qga^2, independiente del estado.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qang.probability import readings_from_p0
from qang.qga import (
    FISHER_PER_QGA2,
    QGA_PER_TURN,
    angle_sigma_qga,
    cone_phase_qga,
    p0_from_qga,
    qga_from_p0,
    qga_from_radians,
    radians_from_qga,
)
from tools.verify_exp_common import (
    DELTA_5SIGMA_QGA,
    N_ESTIMATES,
    SEED,
    SHOTS,
    SIGMA_QGA,
    emit,
    falsify_angular,
)

ANGLES_QGA: tuple[float, ...] = (0.0, 125.0, 250.0, 500.0)


def main() -> None:
    """Ejecuta la verificacion F9 y emite filas con etiqueta de evidencia."""
    rng = np.random.default_rng(SEED)
    sigma_qga = float(angle_sigma_qga(SHOTS))
    fisher_per_rad2 = FISHER_PER_QGA2 * (QGA_PER_TURN / (2.0 * math.pi)) ** 2
    theta_eq = math.pi / 2.0
    p_eq = math.cos(theta_eq / 2.0) ** 2
    dp_eq = -math.sin(theta_eq) / 2.0
    fisher_equator = dp_eq * dp_eq / (p_eq * (1.0 - p_eq))

    rows: list[tuple[str, object, str]] = [
        ("shots", SHOTS, "exact"),
        ("sigma_qga", round(sigma_qga, 6), "exact"),
        ("5sigma_qga", round(DELTA_5SIGMA_QGA, 6), "exact"),
        ("FISHER_per_qga2", FISHER_PER_QGA2, "exact"),
        ("FISHER_per_rad2", round(fisher_per_rad2, 9), "exact"),
        ("FISHER_equator_rad", round(fisher_equator, 9), "exact"),
    ]

    for theta_qga in ANGLES_QGA:
        p0 = p0_from_qga(theta_qga)
        readings = readings_from_p0(p0)
        counts = rng.binomial(SHOTS, p0, size=N_ESTIMATES)
        estimates = np.asarray([qga_from_p0(float(k) / SHOTS) for k in counts])
        verdict, hits, p_value = falsify_angular(theta_qga, estimates)
        control_verdict, control_hits, control_p = falsify_angular(
            (theta_qga + 50.0) % QGA_PER_TURN, estimates
        )
        tag = f"{int(theta_qga):03d}"
        rows.extend(
            [
                (f"p0_at_{tag}", round(p0, 9), "exact"),
                (f"theta_rad_at_{tag}", round(readings["theta"], 9), "exact"),
                (f"qg_Z_at_{tag}", round(readings["qg_z"], 9), "exact"),
                (f"m_qg_at_{tag}", round(readings["milliqang"], 6), "exact"),
                (f"phase_mean_at_{tag}", round(float(np.mean(estimates)), 6), "simulado"),
                (f"hits_at_{tag}", f"{hits}/{N_ESTIMATES}", "simulado"),
                (f"p_upper_at_{tag}", p_value, "simulado"),
                (f"decision_at_{tag}", verdict, "simulado"),
                (f"control_at_{tag}", f"{control_hits}/{N_ESTIMATES} {control_verdict}", "simulado"),
            ]
        )
        assert abs(readings["theta"] - radians_from_qga(theta_qga)) < 1e-12
        assert abs(readings["qg_z"] - math.cos(radians_from_qga(theta_qga))) < 1e-12
        assert abs(readings["milliqang"] - 1000.0 * math.cos(radians_from_qga(theta_qga))) < 1e-9
        assert abs(cone_phase_qga(readings["qg_z"]) - (1000.0 * p0) % QGA_PER_TURN) < 1e-9
        assert abs(float(np.mean(estimates)) - theta_qga) < DELTA_5SIGMA_QGA
        assert verdict == "confirmada" and control_verdict == "refutada"

    emit("verify_exp_f9 — regla de Born / p0 = cos^2(theta/2)", rows)
    assert abs(sigma_qga - SIGMA_QGA) < 1e-12
    assert abs(fisher_per_rad2 - 1.0) < 1e-12
    assert abs(fisher_equator - 1.0) < 1e-12
    assert abs(float(qga_from_radians(math.pi)) - 500.0) < 1e-9
    print("verify_exp_f9: OK")


if __name__ == "__main__":
    main()
