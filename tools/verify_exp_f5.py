"""F5 — pares radicales: verificacion de la fase de Larmor a 50 microtesla y
del singlete de peso 1 en 2 qubits, con resolucion angular 65536 shots.

Hipotesis H5 (propuesta, `unverified`): la fase de Larmor a 50 microtesla es
1.40 MHz y el rendimiento de singlete depende de la orientacion del campo;
la magnetorrecepcion in vivo permanece sin verificar.
Teorema T5 (`exact`): omega_L/2pi = g*mu_B*B/h ~ 28.0 GHz/T * B; Ritz et al.
2000 dan Phi_T 0.56 a 0 G y 0.74 a 0.5 G.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qang.sectors import filter_distribution
from tools.verify_exp_common import (
    DELTA_5SIGMA_QGA,
    N_ESTIMATES,
    SEED,
    SHOTS,
    circular_distance_qga,
    emit,
    falsify_angular,
    noisy_phase_samples,
    signal_phase_qga,
)

MU_B_J_PER_T = 9.2740100783e-24
H_J_S = 6.62607015e-34
G_FACTOR = 2.0
B_TESLA = 50e-6
A1_DIAG_MT = (1.0, 1.0, 0.0)
A2_ISO_MT = 0.5
K_INV_US = 1.0
TAU_US = 1.0
THETA_DEG = 68.0
LARMOR_MHZ_STATED = 1.40
PHI_T_0G = 0.56
PHI_T_0P5G = 0.74


def main() -> None:
    """Ejecuta la verificacion F5 y emite filas con etiqueta de evidencia."""
    rng = np.random.default_rng(SEED)
    gamma_ghz_per_t = G_FACTOR * MU_B_J_PER_T / H_J_S / 1e9
    larmor_mhz_derived = gamma_ghz_per_t * 1e9 * B_TESLA / 1e6
    larmor_cycles = LARMOR_MHZ_STATED * TAU_US
    signal = round(larmor_cycles % 1.0, 12)
    prediction = signal_phase_qga(signal)
    control_signal = (signal + 0.05) % 1.0
    control_prediction = signal_phase_qga(control_signal)
    control_separation = circular_distance_qga(prediction, control_prediction)
    samples = noisy_phase_samples(signal, rng)
    verdict, hits, p_value = falsify_angular(prediction, samples)
    control_verdict, control_hits, control_p = falsify_angular(control_prediction, samples)

    singlet = np.array([0.0, 0.5, 0.5, 0.0])
    filtered, kept = filter_distribution(singlet, 2, 1)

    emit(
        "verify_exp_f5 — pares radicales / Larmor",
        [
            ("shots", SHOTS, "exact"),
            ("5sigma_qga", round(DELTA_5SIGMA_QGA, 6), "exact"),
            ("g_factor", G_FACTOR, "exact"),
            ("B_uT", 50.0, "exact"),
            ("gamma_L_over_2pi_GHz_per_T", round(gamma_ghz_per_t, 9), "exact"),
            ("f_L_derived_MHz", round(larmor_mhz_derived, 9), "exact"),
            ("f_L_stated_MHz", LARMOR_MHZ_STATED, "exact"),
            ("A1_diag_mT", list(A1_DIAG_MT), "exact"),
            ("A2_iso_mT", A2_ISO_MT, "exact"),
            ("k_inv_us", K_INV_US, "exact"),
            ("tau_us", TAU_US, "exact"),
            ("theta_deg", THETA_DEG, "exact"),
            ("Phi_T_0G", PHI_T_0G, "exact"),
            ("Phi_T_0p5G", PHI_T_0P5G, "exact"),
            ("larmor_cycles", round(larmor_cycles, 12), "exact"),
            ("signal_mod1", signal, "exact"),
            ("phase_pred_qga", round(prediction, 6), "exact"),
            ("control_pred_qga", round(control_prediction, 6), "exact"),
            ("control_separation_qga", round(control_separation, 6), "exact"),
            ("phase_mean_qga", round(float(np.mean(samples)), 6), "simulado"),
            ("hits", f"{hits}/{N_ESTIMATES}", "simulado"),
            ("p_upper", p_value, "simulado"),
            ("decision", verdict, "simulado"),
            ("control_hits", f"{control_hits}/{N_ESTIMATES}", "simulado"),
            ("control_p_upper", control_p, "simulado"),
            ("control_decision", control_verdict, "simulado"),
            ("singlet_kept_fraction", round(float(kept), 6), "exact"),
            ("singlet_kept_count", int(np.count_nonzero(filtered > 0)), "exact"),
            ("singlet_filtered_sum", round(float(filtered.sum()), 6), "exact"),
            (
                "in_vivo_magnetoreception",
                "orientacion del rendimiento de singlete no verificada in vivo",
                "unverified",
            ),
        ],
    )
    assert abs(gamma_ghz_per_t - 27.992489872) < 1e-8
    assert abs(larmor_mhz_derived - 1.399624494) < 1e-8
    assert round(larmor_mhz_derived, 2) == LARMOR_MHZ_STATED
    assert abs(signal - 0.40) < 1e-12
    assert abs(prediction - 400.0) < 1e-6
    assert abs(control_separation - 50.0) < 1e-6
    assert abs(float(kept) - 1.0) < 1e-12
    assert abs(float(filtered.sum()) - 1.0) < 1e-12
    assert int(np.count_nonzero(filtered > 0)) == 2
    assert verdict == "confirmada" and control_verdict == "refutada"
    print("verify_exp_f5: OK")


if __name__ == "__main__":
    main()
