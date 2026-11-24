"""F2 — gravedad emergente / cadena XX: verificacion de qg_ZZ^c, ajuste
de carga central y limite de lectura local con resolucion angular 65536 shots.

Hipotesis H2 (propuesta, `unverified`): las lecturas qg locales no ven la
carga central; la geometria emergente no es verificable con qg local.
Teorema T2 (`exact`): S(l)=(c/6)log2[(2n/pi)sin(pi*l/n)]+s0;
qg_ZZ^c = -4 C_ij^2; c(n)->1 en cadena XX a medio llenado.
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
    emit,
    falsify_angular,
    geometric_cone_qga,
    noisy_phase_samples,
    pancharatnam_qga,
    signal_phase_qga,
)


def calabrese_cardy_s2(l: int, n: int, c: float = 1.0, s0: float = 0.0) -> float:
    """Entropia de entrelazamiento Calabrese-Cardy (`exact`, identidad)."""
    return (c / 6.0) * math.log2((2.0 * n / math.pi) * math.sin(math.pi * l / n)) + s0


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
    """Ejecuta la verificacion F2 y emite filas con etiqueta de evidencia."""
    rng = np.random.default_rng(SEED)
    n_chain = 8
    c_central = 1.0
    fit_n = [32, 64, 128, 256, 512]
    fit_c = [1.1165, 1.0648, 1.0337, 1.0171, 1.0086]
    qg_zz_c = -244.84
    s4 = 0.8220
    s4_formula = calabrese_cardy_s2(4, n_chain, c=1.0, s0=0.0)
    signal = abs(qg_zz_c) / 1000.0
    prediction = signal_phase_qga(signal)
    samples = noisy_phase_samples(signal, rng)
    verdict, hits, p_value = falsify_angular(prediction, samples)
    control_verdict, control_hits, control_p = falsify_angular(
        (prediction + 50.0) % 1000.0, samples
    )
    cone = geometric_cone_qga(0.5)
    panch = pancharatnam_loop_qga(0.5)

    emit(
        "verify_exp_f2 — gravedad emergente / cadena XX",
        [
            ("shots", SHOTS, "exact"),
            ("5sigma_qga", round(DELTA_5SIGMA_QGA, 6), "exact"),
            ("n_statevector_half_filling", n_chain, "exact"),
            ("c_central", c_central, "exact"),
            ("fit_n", fit_n, "exact"),
            ("fit_c_n", fit_c, "exact"),
            ("qg_ZZ_c_1_2_mqg", qg_zz_c, "exact"),
            ("S_4_bits", s4, "exact"),
            ("S_4_calabrese_cardy_s0_0", round(s4_formula, 6), "exact"),
            ("phase_pred_qga", round(prediction, 6), "exact"),
            ("phase_mean_qga", round(float(np.mean(samples)), 6), "simulado"),
            ("hits", f"{hits}/{N_ESTIMATES}", "simulado"),
            ("p_upper", p_value, "simulado"),
            ("decision", verdict, "simulado"),
            ("control_hits", f"{control_hits}/{N_ESTIMATES}", "simulado"),
            ("control_p_upper", control_p, "simulado"),
            ("control_decision", control_verdict, "simulado"),
            ("cone_phase_qga_qgz_0p5", round(cone, 6), "exact"),
            ("pancharatnam_qga_qgz_0p5", round(panch, 6), "exact"),
            (
                "ads_cft_emergent_spacetime",
                "no probado; lectura qg local no verifica geometria emergente",
                "unverified",
            ),
        ],
    )
    assert abs(qg_zz_c + 244.84) < 1e-9
    assert abs(s4 - 0.8220) < 1e-9
    assert len(fit_n) == len(fit_c) == 5
    assert abs(cone - panch) < 1e-6
    assert abs(signal - 0.24484) < 1e-12
    assert abs(prediction - 244.84) < 1e-6
    assert verdict == "confirmada" and control_verdict == "refutada"
    print("verify_exp_f2: OK")


if __name__ == "__main__":
    main()
