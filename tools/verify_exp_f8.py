"""F8 — Floquet discrete time crystal: verificacion del desplazamiento de
500 qga entre periodos consecutivos y del sector weight 2 en 4 qubits, con
resolucion angular 65536 shots.

Hipotesis H8 (propuesta, `unverified`): la duplicacion de periodo desplaza la
fase de lectura 500 qga entre periodos de conduccion consecutivos y eps<0.15
no destruye la senal en el modelo de juguete; las afirmaciones fisicas de DTC
permanecen sin verificar.
Teorema T8 (`exact` en el juguete): phi(n+1) = phi(n) + 500 mod 1000 qga.
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
    QGA_PER_TURN,
    SEED,
    SHOTS,
    circular_distance_qga,
    emit,
    falsify_angular,
    noisy_phase_samples,
    signal_phase_qga,
)

N_IONS = 10
T_DRIVE_US = 74.0
EPSILON = 0.1
EPSILON_LIMIT = 0.15
J0_OVER_2PI_KHZ = 0.1
W_T3 = np.pi
J0_T2 = 0.02
N_PERIODS = 100
DETECTION_FIDELITY = 0.98


def main() -> None:
    """Ejecuta la verificacion F8 y emite filas con etiqueta de evidencia."""
    rng = np.random.default_rng(SEED)
    drive_khz = 1000.0 / T_DRIVE_US
    odd_signal = 0.5
    even_signal = 0.0
    odd_prediction = signal_phase_qga(odd_signal)
    even_prediction = signal_phase_qga(even_signal)
    separation = circular_distance_qga(odd_prediction, even_prediction)
    phase_sequence = [
        int(round((n * separation) % QGA_PER_TURN)) for n in range(1, N_PERIODS + 1)
    ]
    control_signal = (odd_signal + 0.05) % 1.0
    control_prediction = signal_phase_qga(control_signal)
    control_separation = circular_distance_qga(odd_prediction, control_prediction)
    samples = noisy_phase_samples(odd_signal, rng)
    verdict, hits, p_value = falsify_angular(odd_prediction, samples)
    control_verdict, control_hits, control_p = falsify_angular(control_prediction, samples)
    even_verdict, even_hits, even_p = falsify_angular(even_prediction, samples)

    probs = np.full(16, 1.0 / 16.0)
    filtered, kept = filter_distribution(probs, 4, 2)
    kept_fraction = 6.0 / 16.0

    emit(
        "verify_exp_f8 — Floquet DTC / period doubling",
        [
            ("shots", SHOTS, "exact"),
            ("5sigma_qga", round(DELTA_5SIGMA_QGA, 6), "exact"),
            ("N_ions", N_IONS, "exact"),
            ("T_drive_us", T_DRIVE_US, "exact"),
            ("drive_kHz", round(drive_khz, 6), "exact"),
            ("epsilon", EPSILON, "exact"),
            ("epsilon_limit", EPSILON_LIMIT, "exact"),
            ("J0_over_2pi_kHz", J0_OVER_2PI_KHZ, "exact"),
            ("W_T3", round(float(W_T3), 6), "exact"),
            ("J0_t2", J0_T2, "exact"),
            ("n_periods", N_PERIODS, "exact"),
            ("detection_fidelity", DETECTION_FIDELITY, "exact"),
            ("odd_pred_qga", round(odd_prediction, 6), "exact"),
            ("even_pred_qga", round(even_prediction, 6), "exact"),
            ("odd_even_separation_qga", round(separation, 6), "exact"),
            ("phase_sequence_all_periods_ok", all(
                phase == (500 if n % 2 == 1 else 0)
                for n, phase in enumerate(phase_sequence, start=1)
            ), "exact"),
            ("control_pred_qga", round(control_prediction, 6), "exact"),
            ("control_separation_qga", round(control_separation, 6), "exact"),
            ("phase_mean_qga", round(float(np.mean(samples)), 6), "simulado"),
            ("hits", f"{hits}/{N_ESTIMATES}", "simulado"),
            ("p_upper", p_value, "simulado"),
            ("decision", verdict, "simulado"),
            ("control_hits", f"{control_hits}/{N_ESTIMATES}", "simulado"),
            ("control_p_upper", control_p, "simulado"),
            ("control_decision", control_verdict, "simulado"),
            ("even_hits", f"{even_hits}/{N_ESTIMATES}", "simulado"),
            ("even_p_upper", even_p, "simulado"),
            ("even_decision", even_verdict, "simulado"),
            ("sector_weight2_kept_fraction", round(kept_fraction, 6), "exact"),
            ("filter_distribution_kept", round(float(kept), 6), "exact"),
            ("sector_weight2_kept_count", int(np.count_nonzero(filtered > 0)), "exact"),
            ("sector_weight2_sum", round(float(filtered.sum()), 6), "exact"),
            (
                "physical_dtc_claim",
                "no verificada; solo modelo de juguete Floquet",
                "unverified",
            ),
        ],
    )
    assert EPSILON < EPSILON_LIMIT
    assert abs(odd_prediction - 500.0) < 1e-6
    assert abs(even_prediction - 0.0) < 1e-6
    assert abs(separation - 500.0) < 1e-6
    assert len(phase_sequence) == N_PERIODS
    assert all(
        phase == (500 if n % 2 == 1 else 0)
        for n, phase in enumerate(phase_sequence, start=1)
    )
    assert abs(control_separation - 50.0) < 1e-6
    assert abs(float(kept) - kept_fraction) < 1e-12
    assert abs(float(filtered.sum()) - 1.0) < 1e-12
    assert int(np.count_nonzero(filtered > 0)) == 6
    assert verdict == "confirmada"
    assert control_verdict == "refutada" and even_verdict == "refutada"
    print("verify_exp_f8: OK")


if __name__ == "__main__":
    main()
