"""F4 — FMO (APC fuera de alcance): verificacion de fase excitonica lineal en
Delta_epsilon*t y distribucion de peso 1 en 7 qubits, con resolucion angular
65536 shots.

Hipotesis H4 (propuesta, `unverified`): la fase de transferencia excitonica en
el modelo toy FMO es lineal en Delta_epsilon*t y resoluble a 5 sigma para
t=10 fs; el calculo usa solo parametros FMO de Adolphs-Renger, por lo que
APC queda fuera de alcance (`unverified` como identificacion).
Teorema T4 (`exact` identidad): propagador de un enlace = RBS(Jt);
phase_qga = 1000*(omega*t mod 1).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qang.geometric import state_from_bloch
from qang.sectors import filter_distribution
from tools.verify_exp_common import (
    DELTA_5SIGMA_QGA,
    N_ESTIMATES,
    QGA_PER_TURN,
    SEED,
    SHOTS,
    emit,
    falsify_angular,
    geometric_cone_qga,
    noisy_phase_samples,
    pancharatnam_qga,
    signal_phase_qga,
)


CM_INV_TO_RAD_PER_FS = 1.8836516e-4
CM_INV_TO_QGA_PER_FS = 0.0299792458


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
    """Ejecuta la verificacion F4 y emite filas con etiqueta de evidencia."""
    rng = np.random.default_rng(SEED)
    site_energies_cm = [410, 530, 210, 320, 480, 630, 440]
    coupling_12_cm = -87.7
    reorg_lambda_cm = 35.0
    tau_c_fs = 50.0
    temp_k = 300.0
    delta_eps_cm = site_energies_cm[1] - site_energies_cm[0]
    delta_qga_per_fs = delta_eps_cm * CM_INV_TO_QGA_PER_FS
    t_fs = 10.0
    phase_t10_qga = (delta_qga_per_fs * t_fs) % QGA_PER_TURN
    signal = phase_t10_qga / QGA_PER_TURN
    prediction = signal_phase_qga(signal)
    samples = noisy_phase_samples(signal, rng)
    verdict, hits, p_value = falsify_angular(prediction, samples)
    control_verdict, control_hits, control_p = falsify_angular(
        (prediction + 50.0) % 1000.0, samples
    )

    probs = np.zeros(128, dtype=float)
    weight1_indices = [1, 2, 4, 8, 16, 32, 64]
    probs[weight1_indices] = 1.0 / 7.0
    filtered, kept = filter_distribution(probs, 7, 1)
    kept_fraction = 7.0 / 128.0

    cone = geometric_cone_qga(0.25)
    panch = pancharatnam_loop_qga(0.25)

    emit(
        "verify_exp_f4 — FMO / APC",
        [
            ("shots", SHOTS, "exact"),
            ("5sigma_qga", round(DELTA_5SIGMA_QGA, 6), "exact"),
            ("site_energies_cm_inv", site_energies_cm, "exact"),
            ("coupling_site1_site2_cm_inv", coupling_12_cm, "exact"),
            ("reorganization_lambda_cm_inv", reorg_lambda_cm, "exact"),
            ("tau_c_fs", tau_c_fs, "exact"),
            ("T_K", temp_k, "exact"),
            ("cm_inv_to_rad_per_fs", CM_INV_TO_RAD_PER_FS, "exact"),
            ("cm_inv_to_qga_per_fs", CM_INV_TO_QGA_PER_FS, "exact"),
            ("delta_epsilon_cm_inv", delta_eps_cm, "exact"),
            ("delta_qga_per_fs", round(delta_qga_per_fs, 7), "exact"),
            ("t_fs", t_fs, "exact"),
            ("phase_t10_qga", round(phase_t10_qga, 6), "exact"),
            ("phase_pred_qga", round(prediction, 6), "exact"),
            ("phase_mean_qga", round(float(np.mean(samples)), 6), "simulado"),
            ("hits", f"{hits}/{N_ESTIMATES}", "simulado"),
            ("p_upper", p_value, "simulado"),
            ("decision", verdict, "simulado"),
            ("control_hits", f"{control_hits}/{N_ESTIMATES}", "simulado"),
            ("control_p_upper", control_p, "simulado"),
            ("control_decision", control_verdict, "simulado"),
            ("sector_weight1_kept_fraction", round(kept_fraction, 6), "exact"),
            ("filter_distribution_kept", round(float(kept), 6), "exact"),
            ("sector_weight1_kept_count", int(np.count_nonzero(filtered > 0)), "exact"),
            ("sector_weight1_sum", round(float(filtered.sum()), 6), "exact"),
            ("cone_phase_qga_qgz_0p25", round(cone, 6), "exact"),
            ("pancharatnam_qga_qgz_0p25", round(panch, 6), "exact"),
            (
                "apc_identification",
                "fuera de alcance: parametros FMO de Adolphs-Renger, no aloficocianina",
                "unverified",
            ),
        ],
    )
    assert delta_eps_cm == 120
    assert abs(delta_qga_per_fs - 3.5975095) < 1e-6
    assert abs(phase_t10_qga - 35.975094) < 1e-5
    assert abs(prediction - 35.975094) < 1e-5
    assert abs(kept_fraction - 7.0 / 128.0) < 1e-12
    assert abs(float(kept) - 1.0) < 1e-12
    assert abs(float(filtered.sum()) - 1.0) < 1e-12
    assert int(np.count_nonzero(filtered > 0)) == 7
    assert verdict == "confirmada" and control_verdict == "refutada"
    assert abs(cone - panch) < 1e-6
    print("verify_exp_f4: OK")


if __name__ == "__main__":
    main()
