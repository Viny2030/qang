"""F1 — firewall post-Page: verificacion de P*F=1/d_A^2, de la curva de Page
y del OTOC de Haar, con la resolucion angular fijada a 65536 shots.

Hipotesis H1 (propuesta, `unverified`): la lectura angular del correlador
compuesto no muestra una firma de firewall en un modelo de juguete; la
identidad P*F=1/4 es exacta y el OTOC de Haar es O(1/d^2).
Teorema T1 (`exact`): P_EPR*F_EPR = 1/d_A^2 y OTOC_Haar = -1/(d^2-1).
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


def page_renyi2(n_qubits: int) -> list[float]:
    """S2 = -log2((N+M)/(M*N+1)) con N=2^k, M=2^(n-k) (`exact`)."""
    out: list[float] = []
    for k in range(1, n_qubits):
        n_sub = 2**k
        m_env = 2 ** (n_qubits - k)
        out.append(-math.log2((n_sub + m_env) / (m_env * n_sub + 1)))
    return out


def pancharatnam_loop_qga(qg_z: float) -> float:
    """Lazo conico evaluated con pancharatnam_phase en cuatro estados."""
    theta = math.acos(max(-1.0, min(1.0, qg_z)))
    states = [
        state_from_bloch((math.sin(theta) * math.cos(a), math.sin(theta) * math.sin(a), qg_z))
        for a in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
    ]
    return pancharatnam_qga(states)


def main() -> None:
    """Ejecuta la verificacion y imprime cada resultado con su etiqueta."""
    rng = np.random.default_rng(SEED)
    d_a, d_d = 2, 4
    full_dim = 16
    pf = 1.0 / d_a**2
    otoc_haar = -1.0 / (full_dim**2 - 1)
    s2 = page_renyi2(8)
    signal = pf
    prediction = signal_phase_qga(signal)
    samples = noisy_phase_samples(signal, rng)
    verdict, hits, p_value = falsify_angular(prediction, samples)
    control_verdict, control_hits, control_p = falsify_angular(
        (prediction + 50.0) % 1000.0, samples
    )
    cone = geometric_cone_qga(0.0)
    panch = pancharatnam_loop_qga(0.0)
    singlet = np.array([0.0, 0.5, 0.5, 0.0])
    filtered, kept = filter_distribution(singlet, 2, 1)

    emit(
        "verify_exp_f1 — firewall post-Page / Yoshida-Kitaev",
        [
            ("shots", SHOTS, "exact"),
            ("5sigma_qga", round(DELTA_5SIGMA_QGA, 6), "exact"),
            ("P*F", pf, "exact"),
            ("OTOC_Haar", round(otoc_haar, 9), "exact"),
            ("S2_Page_k1..7", [round(v, 3) for v in s2], "exact"),
            ("phase_pred_qga", round(prediction, 6), "exact"),
            ("phase_mean_qga", round(float(np.mean(samples)), 6), "simulado"),
            ("hits", f"{hits}/{N_ESTIMATES}", "simulado"),
            ("p_upper", p_value, "simulado"),
            ("decision", verdict, "simulado"),
            ("control_hits", f"{control_hits}/{N_ESTIMATES}", "simulado"),
            ("control_p_upper", control_p, "simulado"),
            ("control_decision", control_verdict, "simulado"),
            ("cone_phase_qga", round(cone, 6), "exact"),
            ("pancharatnam_qga", round(panch, 6), "exact"),
            ("singlet_kept", round(kept, 6), "exact"),
            ("singlet_filtered_sum", round(float(filtered.sum()), 6), "exact"),
            ("firewall_interpretation", "sin evidencia", "unverified"),
        ],
    )
    assert abs(pf - 0.25) < 1e-12
    assert abs(cone - panch) < 1e-6
    assert abs(float(np.mean(samples)) - prediction) < DELTA_5SIGMA_QGA
    assert verdict == "confirmada" and control_verdict == "refutada"
    assert abs(float(filtered.sum()) - 1.0) < 1e-12
    print("verify_exp_f1: OK")


if __name__ == "__main__":
    main()
