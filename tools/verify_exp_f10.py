"""F10 — fluctuacion retrocausal: verificacion del desplazamiento amplificado
por 1/a en el mapa P-CTC y del balance de Fisher por copia, con resolucion
angular fijada a 65536 shots.

Hipotesis H10 (propuesta, `unverified`): en un bucle postseleccionado, una
pequena fluctuacion del angulo de entrada se lee amplificada en el angulo de
salida; contando la tasa de exito la informacion por copia sigue siendo 1,
de modo que no hay senal retrocausal utilizable ni evidencia de CTC fisicas.
Teorema T10 (`exact` en el mapa): para C = 2 diag(1, a),
tan(theta'/2) = a tan(theta/2), p_post = cos^2(theta/2) + a^2 sin^2(theta/2),
dtheta'/dtheta = a/p_post (que vale 1/a en el polo sur) y
p_post F_acc + F_fracaso = 1.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qang.qga import QGA_PER_TURN, qga_from_radians
from qang.qtimeloop import (
    pctc_amplification,
    pctc_fisher,
    pctc_p_post,
    pctc_theta_out,
    shots_for_successes,
)
from tools.verify_exp_common import (
    DELTA_5SIGMA_QGA,
    N_ESTIMATES,
    SEED,
    SHOTS,
    emit,
    falsify_angular,
    noisy_phase_samples,
    signal_phase_qga,
)

A: float = 0.5
THETA_BASE: float = math.pi - 0.02
DELTA_THETA: float = 0.01
N_PAIRS: int = 2000


def main() -> None:
    """Ejecuta la verificacion F10 y emite filas con etiqueta de evidencia."""
    rng = np.random.default_rng(SEED + 2)
    theta_pert = THETA_BASE + DELTA_THETA
    theta_out_base = pctc_theta_out(A, THETA_BASE)
    theta_out_pert = pctc_theta_out(A, theta_pert)
    shift_rad = abs(theta_out_pert - theta_out_base)
    amplification = pctc_amplification(A, THETA_BASE)
    linear_prediction = amplification * DELTA_THETA
    p_post = pctc_p_post(A, THETA_BASE)
    f_acc, p_check, f_failure, f_total = pctc_fisher(A, THETA_BASE)
    a_grid = rng.uniform(0.05, 1.0, size=N_PAIRS)
    theta_grid = rng.uniform(0.01, math.pi - 0.01, size=N_PAIRS)
    max_fisher_deviation = max(
        abs(pctc_fisher(float(a), float(t))[3] - 1.0) for a, t in zip(a_grid, theta_grid)
    )
    shots_1000 = shots_for_successes(1000.0, p_post)

    signal = shift_rad / math.pi
    prediction = signal_phase_qga(signal)
    samples = noisy_phase_samples(signal, rng)
    verdict, hits, p_value = falsify_angular(prediction, samples)
    control_verdict, control_hits, control_p = falsify_angular(
        (prediction + 50.0) % QGA_PER_TURN, samples
    )

    emit(
        "verify_exp_f10 — fluctuacion retrocausal / P-CTC 1/a",
        [
            ("shots", SHOTS, "exact"),
            ("5sigma_qga", round(DELTA_5SIGMA_QGA, 6), "exact"),
            ("a", A, "exact"),
            ("theta_base_rad", round(THETA_BASE, 8), "exact"),
            ("delta_theta_rad", DELTA_THETA, "exact"),
            ("theta_out_base_rad", round(theta_out_base, 8), "exact"),
            ("theta_out_pert_rad", round(theta_out_pert, 8), "exact"),
            ("retro_shift_rad", round(shift_rad, 8), "exact"),
            ("retro_shift_qga", round(float(qga_from_radians(shift_rad)), 6), "exact"),
            ("amplification", round(amplification, 8), "exact"),
            ("linear_prediction_rad", round(linear_prediction, 8), "exact"),
            ("p_post", round(p_post, 9), "exact"),
            ("F_accepted", round(f_acc, 9), "exact"),
            ("F_success_rate", round(f_failure, 9), "exact"),
            ("F_total_per_copy", round(f_total, 12), "exact"),
            ("max_F_deviation_2000_pairs", max_fisher_deviation, "exact"),
            ("shots_for_1000_successes", round(shots_1000, 3), "exact"),
            ("phase_pred_qga", round(prediction, 6), "exact"),
            ("phase_mean_qga", round(float(np.mean(samples)), 6), "simulado"),
            ("hits", f"{hits}/{N_ESTIMATES}", "simulado"),
            ("p_upper", p_value, "simulado"),
            ("decision", verdict, "simulado"),
            ("control_hits", f"{control_hits}/{N_ESTIMATES}", "simulado"),
            ("control_decision", control_verdict, "simulado"),
            (
                "physical_retrocausality",
                "no afirmada; solo mapa P-CTC de juguete",
                "unverified",
            ),
        ],
    )
    assert abs(shift_rad - linear_prediction) < 1e-4
    assert abs(p_check - p_post) < 1e-15
    assert abs(f_acc - amplification**2) < 1e-12
    assert abs(f_total - 1.0) < 1e-12
    assert max_fisher_deviation < 1e-9
    assert abs(shots_1000 * p_post - 1000.0) < 1e-9
    assert abs(float(np.mean(samples)) - prediction) < DELTA_5SIGMA_QGA
    assert verdict == "confirmada" and control_verdict == "refutada"
    print("verify_exp_f10: OK")


if __name__ == "__main__":
    main()
