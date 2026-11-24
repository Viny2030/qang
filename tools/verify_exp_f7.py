"""F7 — superconductividad de hidruros / Josephson: verificacion de la razon
de brecha 2Delta/(kB Tc) de H3S frente a BCS debil y del control de 50 qga,
con resolucion angular 65536 shots.

Hipotesis H7 (propuesta, `unverified`): la razon 3.43 de H3S es distinguishable
del valor BCS 3.52 a la resolucion fija de 5 sigma; la correspondencia entre
datos de brecha de hidruros y un E_J de Josephson no esta verificada.
Teorema T7 (`exact`-derived): 2Delta/(kB Tc) = 60 meV/(kB*203 K) = 3.43.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qang.geometric import state_from_bloch
from tools.verify_exp_common import (
    DELTA_5SIGMA_QGA,
    N_ESTIMATES,
    SEED,
    SHOTS,
    circular_distance_qga,
    emit,
    falsify_angular,
    geometric_cone_qga,
    noisy_phase_samples,
    pancharatnam_qga,
    signal_phase_qga,
)

KB_EV_PER_K = 8.617333262e-5
TC_H3S_K = 203.0
P_H3S_GPA = 155.0
TWO_DELTA_H3S_MEV = 60.0
TC_LAH10_K = 250.0
P_LAH10_GPA = 170.0
DELTA_LAH10_MEV = 53.0
EJ_OVER_H_GHZ = 30.0
BCS_WEAK_RATIO = 3.52


def pancharatnam_loop_qga(qg_z: float) -> float:
    """Lazo conico evaluado con pancharatnam_phase sobre estados de Bloch."""
    theta = math.acos(max(-1.0, min(1.0, qg_z)))
    states = [
        state_from_bloch(
            (math.sin(theta) * math.cos(a), math.sin(theta) * math.sin(a), qg_z)
        )
        for a in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
    ]
    return pancharatnam_qga(states)


def main() -> None:
    """Ejecuta la verificacion F7 y emite filas con etiqueta de evidencia."""
    rng = np.random.default_rng(SEED)
    ratio_measured = (TWO_DELTA_H3S_MEV * 1e-3) / (KB_EV_PER_K * TC_H3S_K)
    ratio_stated = round(ratio_measured, 2)
    ratio_lah10 = (2.0 * DELTA_LAH10_MEV * 1e-3) / (KB_EV_PER_K * TC_LAH10_K)
    signal = ratio_stated / 10.0
    bcs_signal = BCS_WEAK_RATIO / 10.0
    prediction = signal_phase_qga(signal)
    control_signal = (signal + 0.05) % 1.0
    control_prediction = signal_phase_qga(control_signal)
    control_separation = circular_distance_qga(prediction, control_prediction)
    bcs_prediction = signal_phase_qga(bcs_signal)
    bcs_separation = circular_distance_qga(prediction, bcs_prediction)
    samples = noisy_phase_samples(signal, rng)
    verdict, hits, p_value = falsify_angular(prediction, samples)
    control_verdict, control_hits, control_p = falsify_angular(control_prediction, samples)
    bcs_verdict, bcs_hits, bcs_p = falsify_angular(bcs_prediction, samples)

    cone = geometric_cone_qga(0.25)
    panch = pancharatnam_loop_qga(0.25)

    emit(
        "verify_exp_f7 — H3S / hydride Josephson",
        [
            ("shots", SHOTS, "exact"),
            ("5sigma_qga", round(DELTA_5SIGMA_QGA, 6), "exact"),
            ("k_B_eV_per_K", KB_EV_PER_K, "exact"),
            ("Tc_H3S_K", TC_H3S_K, "exact"),
            ("P_H3S_GPa", P_H3S_GPA, "exact"),
            ("2Delta_H3S_meV", TWO_DELTA_H3S_MEV, "exact"),
            ("Tc_LaH10_K", TC_LAH10_K, "exact"),
            ("P_LaH10_GPa", P_LAH10_GPA, "exact"),
            ("Delta_LaH10_meV", DELTA_LAH10_MEV, "exact"),
            ("EJ_over_h_GHz_toy", EJ_OVER_H_GHZ, "exact"),
            ("ratio_measured", round(ratio_measured, 9), "exact"),
            ("ratio_stated", ratio_stated, "exact"),
            ("ratio_BCS_weak", BCS_WEAK_RATIO, "exact"),
            ("ratio_LaH10", round(ratio_lah10, 9), "exact"),
            ("signal_mod1", signal, "exact"),
            ("phase_pred_qga", round(prediction, 6), "exact"),
            ("control_pred_qga", round(control_prediction, 6), "exact"),
            ("control_separation_qga", round(control_separation, 6), "exact"),
            ("bcs_pred_qga", round(bcs_prediction, 6), "exact"),
            ("bcs_separation_qga", round(bcs_separation, 6), "exact"),
            ("phase_mean_qga", round(float(np.mean(samples)), 6), "simulado"),
            ("hits", f"{hits}/{N_ESTIMATES}", "simulado"),
            ("p_upper", p_value, "simulado"),
            ("decision", verdict, "simulado"),
            ("control_hits", f"{control_hits}/{N_ESTIMATES}", "simulado"),
            ("control_p_upper", control_p, "simulado"),
            ("control_decision", control_verdict, "simulado"),
            ("bcs_hits", f"{bcs_hits}/{N_ESTIMATES}", "simulado"),
            ("bcs_p_upper", bcs_p, "simulado"),
            ("bcs_decision", bcs_verdict, "simulado"),
            ("cone_phase_qga_qgz_0p25", round(cone, 6), "exact"),
            ("pancharatnam_qga_qgz_0p25", round(panch, 6), "exact"),
            (
                "hydride_gap_to_josephson_EJ",
                "mapeo de datos de brecha a E_J no verificado",
                "unverified",
            ),
        ],
    )
    assert abs(ratio_measured - 3.429906834) < 1e-8
    assert ratio_stated == 3.43
    assert abs(signal - 0.343) < 1e-12
    assert abs(prediction - 343.0) < 1e-6
    assert abs(control_separation - 50.0) < 1e-6
    assert abs(bcs_separation - 9.0) < 1e-6
    assert bcs_separation > DELTA_5SIGMA_QGA
    assert verdict == "confirmada"
    assert control_verdict == "refutada" and bcs_verdict == "refutada"
    assert abs(cone - panch) < 1e-6
    print("verify_exp_f7: OK")


if __name__ == "__main__":
    main()
