"""F3 — flecha del tiempo: verificacion de qg_Z(tau) impar en tau para
H=omega Z/2 desde |+>, con resolucion angular 65536 shots.

Hipotesis H3 (propuesta, `unverified`): qg_Z(tau) es impar en tau para
H=omega Z/2 desde |+>, de modo que invertir tau desplaza la fase 71.945 qga
bajo la codificacion qg_Phi usada; no se sigue ninguna flecha fisica del
tiempo solo de la dinamica unitaria.
Teorema T3 (`exact`): qg_Z(tau) = -tanh(omega*tau).
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


def qg_z_tau(omega: float, tau: float) -> float:
    """Identidad exacta T3: qg_Z(tau) = -tanh(omega*tau)."""
    return -math.tanh(omega * tau)


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
    """Ejecuta la verificacion F3 y emite filas con etiqueta de evidencia."""
    rng_fwd = np.random.default_rng(SEED)
    rng_rev = np.random.default_rng(SEED + 1)
    omega = 1.0
    tau = 2.0
    qg_z_fwd = qg_z_tau(omega, tau)
    qg_z_rev = qg_z_tau(omega, -tau)
    signal_fwd = abs(qg_z_fwd)
    pred_fwd = signal_phase_qga(signal_fwd)
    pred_rev = signal_phase_qga(-signal_fwd)
    samples_fwd = noisy_phase_samples(signal_fwd, rng_fwd)
    verdict_fwd, hits_fwd, p_fwd = falsify_angular(pred_fwd, samples_fwd)
    control_verdict, control_hits, control_p = falsify_angular(
        (pred_fwd + 50.0) % 1000.0, samples_fwd
    )
    samples_rev = noisy_phase_samples(-signal_fwd, rng_rev)
    verdict_rev, hits_rev, p_rev = falsify_angular(pred_rev, samples_rev)
    separation = circular_distance_qga(pred_fwd, pred_rev)
    cone = geometric_cone_qga(0.5)
    panch = pancharatnam_loop_qga(0.5)

    emit(
        "verify_exp_f3 — flecha del tiempo / qg_Z(tau)",
        [
            ("shots", SHOTS, "exact"),
            ("5sigma_qga", round(DELTA_5SIGMA_QGA, 6), "exact"),
            ("omega_adim", omega, "exact"),
            ("tau", tau, "exact"),
            ("n_qubits", 1, "exact"),
            ("tanh_2", round(math.tanh(2.0), 8), "exact"),
            ("qg_Z_forward", round(qg_z_fwd, 8), "exact"),
            ("qg_Z_reversed", round(qg_z_rev, 8), "exact"),
            ("phase_pred_forward_qga", round(pred_fwd, 6), "exact"),
            ("phase_pred_reversed_qga", round(pred_rev, 6), "exact"),
            ("forward_reversed_separation_qga", round(separation, 6), "exact"),
            ("phase_mean_forward_qga", round(float(np.mean(samples_fwd)), 6), "simulado"),
            ("forward_hits", f"{hits_fwd}/{N_ESTIMATES}", "simulado"),
            ("forward_p_upper", p_fwd, "simulado"),
            ("forward_decision", verdict_fwd, "simulado"),
            ("phase_mean_reversed_qga", round(float(np.mean(samples_rev)), 6), "simulado"),
            ("reversed_hits", f"{hits_rev}/{N_ESTIMATES}", "simulado"),
            ("reversed_p_upper", p_rev, "simulado"),
            ("reversed_decision", verdict_rev, "simulado"),
            ("control_hits", f"{control_hits}/{N_ESTIMATES}", "simulado"),
            ("control_p_upper", control_p, "simulado"),
            ("control_decision", control_verdict, "simulado"),
            ("cone_phase_qga_qgz_0p5", round(cone, 6), "exact"),
            ("pancharatnam_qga_qgz_0p5", round(panch, 6), "exact"),
            (
                "physical_arrow_of_time",
                "no se sigue flecha fisica del tiempo solo de dinamica unitaria",
                "unverified",
            ),
        ],
    )
    assert abs(qg_z_fwd + math.tanh(2.0)) < 1e-12
    assert abs(qg_z_rev - math.tanh(2.0)) < 1e-12
    assert abs(pred_fwd - 964.02758) < 1e-6
    assert abs(pred_rev - 35.97242) < 1e-5
    assert abs(separation - 71.94484) < 1e-4
    assert verdict_fwd == "confirmada" and control_verdict == "refutada"
    assert verdict_rev == "confirmada"
    assert abs(cone - panch) < 1e-6
    print("verify_exp_f3: OK")


if __name__ == "__main__":
    main()
