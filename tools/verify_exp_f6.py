"""F6 — ADN / ZPE: verificacion de la fase de 1.11 THz en 1 ps y de su
separacion respecto de cero a 5 sigma, con resolucion angular 65536 shots.

Hipotesis H6 (propuesta, `unverified`): el modo de torsion C-G de 1.11 THz
acumula 110 qga en 1 ps y es distinguishable de cero a 5 sigma; un mecanismo
directo de transferencia de ZPE a carga en ADN NO esta establecido.
Teorema T6 (`exact`): phi_qga = 1000*(nu*t mod 1); ZPE = h*nu/2.
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

H_EV_S = 4.135667696e-15
NU_HZ = 1.11e12
T_PS = 1.0
DG_AT_KCAL_MOL = 0.57
DG_GC_KCAL_MOL = -0.11


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
    """Ejecuta la verificacion F6 y emite filas con etiqueta de evidencia."""
    rng = np.random.default_rng(SEED)
    zpe_ev = H_EV_S * NU_HZ / 2.0
    zpe_mev = zpe_ev * 1e3
    signal = round(((NU_HZ / 1e12) * T_PS) % 1.0, 12)
    prediction = signal_phase_qga(signal)
    control_signal = (signal + 0.05) % 1.0
    control_prediction = signal_phase_qga(control_signal)
    control_separation = circular_distance_qga(prediction, control_prediction)
    zero_prediction = signal_phase_qga(0.0)
    zero_separation = circular_distance_qga(prediction, zero_prediction)
    samples = noisy_phase_samples(signal, rng)
    verdict, hits, p_value = falsify_angular(prediction, samples)
    control_verdict, control_hits, control_p = falsify_angular(control_prediction, samples)
    zero_verdict, zero_hits, zero_p = falsify_angular(zero_prediction, samples)

    cone = geometric_cone_qga(0.75)
    panch = pancharatnam_loop_qga(0.75)

    emit(
        "verify_exp_f6 — ADN / ZPE / torsion C-G",
        [
            ("shots", SHOTS, "exact"),
            ("5sigma_qga", round(DELTA_5SIGMA_QGA, 6), "exact"),
            ("nu_THz", NU_HZ / 1e12, "exact"),
            ("t_ps", T_PS, "exact"),
            ("h_eV_s", H_EV_S, "exact"),
            ("ZPE_eV", zpe_ev, "exact"),
            ("ZPE_meV", round(zpe_mev, 9), "exact"),
            ("dG_AT_kcal_per_mol", DG_AT_KCAL_MOL, "exact"),
            ("dG_GC_kcal_per_mol", DG_GC_KCAL_MOL, "exact"),
            ("signal_mod1", signal, "exact"),
            ("phase_pred_qga", round(prediction, 6), "exact"),
            ("control_pred_qga", round(control_prediction, 6), "exact"),
            ("control_separation_qga", round(control_separation, 6), "exact"),
            ("zero_pred_qga", round(zero_prediction, 6), "exact"),
            ("zero_separation_qga", round(zero_separation, 6), "exact"),
            ("phase_mean_qga", round(float(np.mean(samples)), 6), "simulado"),
            ("hits", f"{hits}/{N_ESTIMATES}", "simulado"),
            ("p_upper", p_value, "simulado"),
            ("decision", verdict, "simulado"),
            ("control_hits", f"{control_hits}/{N_ESTIMATES}", "simulado"),
            ("control_p_upper", control_p, "simulado"),
            ("control_decision", control_verdict, "simulado"),
            ("zero_hits", f"{zero_hits}/{N_ESTIMATES}", "simulado"),
            ("zero_p_upper", zero_p, "simulado"),
            ("zero_decision", zero_verdict, "simulado"),
            ("cone_phase_qga_qgz_0p75", round(cone, 6), "exact"),
            ("pancharatnam_qga_qgz_0p75", round(panch, 6), "exact"),
            (
                "zpe_to_dna_charge_transfer",
                "mecanismo directo no establecido",
                "unverified",
            ),
        ],
    )
    assert abs(zpe_mev - 2.295295571) < 1e-8
    assert DG_AT_KCAL_MOL > 0.0 > DG_GC_KCAL_MOL
    assert abs(signal - 0.11) < 1e-12
    assert abs(prediction - 110.0) < 1e-6
    assert abs(control_separation - 50.0) < 1e-6
    assert abs(zero_separation - 110.0) < 1e-6
    assert zero_separation > DELTA_5SIGMA_QGA
    assert verdict == "confirmada"
    assert control_verdict == "refutada" and zero_verdict == "refutada"
    assert abs(cone - panch) < 1e-6
    print("verify_exp_f6: OK")


if __name__ == "__main__":
    main()
