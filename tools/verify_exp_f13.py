r"""F13 — benchmark metrológico B3: la misma fase angular bajo ruido en cinco stacks.

Hipótesis H13 (`unverified`): con 65_536 shots, la combinación de `qecho.invert_readout`
y el filtro de sector reduce el error angular frente a la lectura cruda cuando hay
despolarización, amortiguamiento de amplitud y error de lectura; la afirmación sobre
hardware real sigue sin probarse.

Teorema T13 (`exact`): para $|\psi\rangle=(|0\rangle+e^{i\varphi}|1\rangle)/\sqrt2$ el vector de Bloch es $(\cos\varphi,\sin\varphi,0)$; la despolarización con parámetro $p$ lo escala por $1-p$, el amortiguamiento de amplitud con $\gamma$ envía $(x,y,z)\to(\sqrt{1-\gamma}x,\sqrt{1-\gamma}y,\gamma+(1-\gamma)z)$, y la lectura afín $m'=km+1000(e_{10}-e_{01})$ se invierte con $k=1-e_{01}-e_{10}$, de modo que `azimuth_qga` recupera $\varphi$ aunque el sesgo crudo desplace el azimut más de $5\sigma$.

Circuito testigo: 1 qubit, `H` seguido de `RZ(phi)` con $\varphi=250$ qga, lectura X/Y, 65_536 shots, 600 estimaciones, umbral 5 sigma = 3.108 qga y control +50 qga. Los statevectors reales disponibles son Qiskit, PennyLane y Cirq; QuTiP, Braket e IonQ quedan como fallback analítico `unverified` en este entorno.
"""
from __future__ import annotations

import math
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qang.geometric import bloch_from_state
from qang.qecho import invert_readout, mse_raw_vs_inverted, readout_affine
from qang.qfalsify import verify_registration
from qang.qga import QGA_PER_TURN, angle_sigma_qga, azimuth_qga, radians_from_qga
from qang.sectors import filter_distribution
from tools.verify_exp_common import (
    DELTA_5SIGMA_QGA,
    E01,
    E10,
    N_ESTIMATES,
    SEED,
    SHOTS,
    SIGMA_QGA,
    circular_distance_qga,
    emit,
    falsify_angular,
    noisy_phase_samples,
    phase_qga_from_complex,
)

PHASE_QGA: float = 250.0
DEPOLARIZING_P: float = 0.02
AMPLITUDE_DAMPING_GAMMA: float = 0.05
N_CAL: int = SHOTS
CONTROL_OFFSET_QGA: float = 50.0
STACKS: tuple[str, ...] = ("qiskit", "pennylane", "cirq", "qutip", "braket", "ionq_noise_model")
REAL_STACKS: tuple[str, ...] = ("qiskit", "pennylane", "cirq")
NOISE_CHANNELS: tuple[str, ...] = ("depolarizing", "amplitude_damping", "readout_only")

PREDICTION: dict = {
    "exp": "F13",
    "domain": "librerias_cuanticas",
    "frameworks": list(STACKS),
    "real_statevectors_in_this_environment": list(REAL_STACKS),
    "mode": "real_libraries_where_installed_plus_analytic_fallback",
    "circuit": "H then RZ(phi), readout X/Y with affine error and qecho inversion",
    "phase_qga": PHASE_QGA,
    "depolarizing_p": DEPOLARIZING_P,
    "amplitude_damping_gamma": AMPLITUDE_DAMPING_GAMMA,
    "readout_e01": E01,
    "readout_e10": E10,
    "shots": SHOTS,
    "n_cal": N_CAL,
    "n_estimates": N_ESTIMATES,
    "delta_5sigma_qga": DELTA_5SIGMA_QGA,
    "control_offset_qga": CONTROL_OFFSET_QGA,
    "seed": SEED,
    "criterion": "corrected azimuth within 1e-3 qga, raw error above 5 sigma, 600/600 hits, control refutada and MSE_inverted below MSE_raw for every channel",
}
REGISTRATION_SHA256: str = "ecb0dad97b5fa18e8eb134e4ccb4d001eb7b6f1a7bf1dae20ae0fbb38e949dff"


def reference_bloch(phi: float) -> np.ndarray:
    """Vector de Bloch analítico de (|0> + exp(i phi)|1>)/sqrt(2)."""
    return np.array([math.cos(phi), math.sin(phi), 0.0], dtype=float)


def qiskit_bloch(phi: float) -> np.ndarray:
    """Statevector real de Qiskit para H y RZ."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.rz(phi, 0)
    return bloch_from_state(np.asarray(Statevector.from_instruction(circuit).data))


def pennylane_bloch(phi: float) -> np.ndarray:
    """Statevector real de PennyLane para H y RZ."""
    import pennylane as qml

    device = qml.device("default.qubit", wires=1)

    @qml.qnode(device)
    def circuit():
        qml.Hadamard(wires=0)
        qml.RZ(phi, wires=0)
        return qml.state()

    return bloch_from_state(np.asarray(circuit()))


def cirq_bloch(phi: float) -> np.ndarray:
    """Statevector real de Cirq para H y rz."""
    import cirq

    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit([cirq.H(qubit), cirq.rz(phi)(qubit)])
    return bloch_from_state(np.asarray(cirq.final_state_vector(circuit)))


def stack_bloch(stack: str, phi: float) -> tuple[np.ndarray, bool]:
    """Devuelve (vector de Bloch, bandera de statevector real)."""
    if stack == "qiskit":
        return qiskit_bloch(phi), True
    if stack == "pennylane":
        return pennylane_bloch(phi), True
    if stack == "cirq":
        return cirq_bloch(phi), True
    return reference_bloch(phi), False


def apply_noise(bloch: np.ndarray, channel: str) -> np.ndarray:
    """Aplica el modelo de ruido analítico al vector de Bloch."""
    if channel == "depolarizing":
        return (1.0 - DEPOLARIZING_P) * bloch
    if channel == "amplitude_damping":
        scale = math.sqrt(1.0 - AMPLITUDE_DAMPING_GAMMA)
        return np.array([scale * bloch[0], scale * bloch[1], AMPLITUDE_DAMPING_GAMMA + (1.0 - AMPLITUDE_DAMPING_GAMMA) * bloch[2]])
    if channel == "readout_only":
        return np.asarray(bloch, dtype=float).copy()
    raise ValueError(f"unknown channel {channel!r}")


def azimuth_from_bloch(bloch: np.ndarray, corrected: bool) -> float:
    """Azimut en qga con lectura X/Y cruda o invertida por qecho."""
    qg_x = float(bloch[0] * QGA_PER_TURN)
    qg_y = float(bloch[1] * QGA_PER_TURN)
    if corrected:
        qg_x = invert_readout(readout_affine(qg_x, E01, E10), E01, E10)
        qg_y = invert_readout(readout_affine(qg_y, E01, E10), E01, E10)
    else:
        qg_x = readout_affine(qg_x, E01, E10)
        qg_y = readout_affine(qg_y, E01, E10)
    return azimuth_qga(qg_x, qg_y)


def main() -> None:
    """Ejecuta F13 y emite filas con etiqueta de evidencia."""
    assert verify_registration(PREDICTION, REGISTRATION_SHA256)

    rng = np.random.default_rng(SEED)
    phi = radians_from_qga(PHASE_QGA)
    rows: list[tuple[str, object, str]] = [
        ("H13_claim", "qecho y filtro de sector bajan el error angular; hardware real no afirmado", "unverified"),
        ("T13_phase_qga", PHASE_QGA, "exact"),
        ("T13_depolarizing_p", DEPOLARIZING_P, "exact"),
        ("T13_amplitude_damping_gamma", AMPLITUDE_DAMPING_GAMMA, "exact"),
        ("T13_readout_e01_e10", [E01, E10], "exact"),
        ("T13_readout_gain_k", 1.0 - E01 - E10, "exact"),
        ("T13_qangphi_phase_qga", round(phase_qga_from_complex(complex(math.cos(phi), math.sin(phi))), 9), "exact"),
        ("registration_sha256", REGISTRATION_SHA256[:16], "exact"),
        ("shots", SHOTS, "exact"),
        ("sigma_qga", round(SIGMA_QGA, 6), "exact"),
        ("angle_sigma_qga", round(float(angle_sigma_qga(SHOTS)), 6), "exact"),
        ("5sigma_qga", DELTA_5SIGMA_QGA, "exact"),
    ]

    versions: dict[str, str] = {}
    for stack in STACKS:
        base, real = stack_bloch(stack, phi)
        if real:
            versions[stack] = version(stack)
        noisy = apply_noise(base, "depolarizing")
        raw = azimuth_from_bloch(noisy, False)
        corrected = azimuth_from_bloch(noisy, True)
        samples = noisy_phase_samples(PHASE_QGA / QGA_PER_TURN, rng)
        verdict, hits, p_value = falsify_angular(PHASE_QGA, samples)
        control_verdict, control_hits, control_p = falsify_angular(
            (PHASE_QGA + CONTROL_OFFSET_QGA) % QGA_PER_TURN, samples
        )
        rows.extend(
            [
                (f"{stack}_source", "real_statevector" if real else "analytic_fallback_sdk_absent", "exact" if real else "unverified"),
                (f"{stack}_raw_azimuth_qga", round(raw, 6), "simulado"),
                (f"{stack}_corrected_azimuth_qga", round(corrected, 6), "simulado"),
                (f"{stack}_raw_error_qga", round(circular_distance_qga(raw, PHASE_QGA), 6), "simulado"),
                (f"{stack}_corrected_error_qga", round(circular_distance_qga(corrected, PHASE_QGA), 6), "simulado"),
                (f"{stack}_hits", f"{hits}/{N_ESTIMATES}", "simulado"),
                (f"{stack}_p_upper", p_value, "simulado"),
                (f"{stack}_decision", verdict, "simulado"),
                (f"{stack}_control", f"{control_hits}/{N_ESTIMATES} {control_verdict}", "simulado"),
                (f"{stack}_control_p", control_p, "simulado"),
            ]
        )
        assert circular_distance_qga(corrected, PHASE_QGA) < 1e-3
        assert circular_distance_qga(raw, PHASE_QGA) > DELTA_5SIGMA_QGA
        assert hits == N_ESTIMATES and control_hits == 0
        assert verdict == "confirmada" and control_verdict == "refutada"

    rows.append(("real_library_versions", versions, "exact"))

    for channel in NOISE_CHANNELS:
        noisy = apply_noise(reference_bloch(phi), channel)
        raw = azimuth_from_bloch(noisy, False)
        corrected = azimuth_from_bloch(noisy, True)
        mse_raw, mse_inv, bias, sd_m = mse_raw_vs_inverted(
            float(noisy[2] * QGA_PER_TURN), E01, E10, SHOTS, N_CAL
        )
        rows.extend(
            [
                (f"T13_{channel}_raw_azimuth_qga", round(raw, 6), "simulado"),
                (f"T13_{channel}_corrected_azimuth_qga", round(corrected, 6), "simulado"),
                (f"T13_{channel}_raw_error_qga", round(circular_distance_qga(raw, PHASE_QGA), 6), "simulado"),
                (f"T13_{channel}_corrected_error_qga", round(circular_distance_qga(corrected, PHASE_QGA), 6), "simulado"),
                (f"T13_{channel}_readout_bias_mqg", round(float(bias), 6), "exact"),
                (f"T13_{channel}_sd_mqg", round(float(sd_m), 6), "exact"),
                (f"T13_{channel}_mse_raw_mqg2", round(float(mse_raw), 6), "simulado"),
                (f"T13_{channel}_mse_inverted_mqg2", round(float(mse_inv), 6), "simulado"),
                (f"T13_{channel}_mse_ratio", round(float(mse_raw / mse_inv), 6), "simulado"),
            ]
        )
        assert circular_distance_qga(corrected, PHASE_QGA) < 1e-9
        assert circular_distance_qga(raw, PHASE_QGA) > DELTA_5SIGMA_QGA
        assert mse_inv < mse_raw

    distribution = np.array([0.40, 0.20, 0.20, 0.20], dtype=float)
    filtered, kept = filter_distribution(distribution, 2, 1)
    counts = rng.multinomial(SHOTS, distribution)
    filtered_counts, kept_counts = filter_distribution(counts, 2, 1)
    expected_kept = 0.40
    tolerance = 5.0 * math.sqrt(expected_kept * (1.0 - expected_kept) / SHOTS)
    rows.extend(
        [
            ("T13_sector_kept_exact", round(float(kept), 9), "exact"),
            ("T13_sector_filtered_exact", np.round(filtered, 9).tolist(), "exact"),
            ("T13_sector_kept_sampled", round(float(kept_counts), 9), "simulado"),
            ("T13_sector_filtered_sampled", np.round(filtered_counts, 9).tolist(), "simulado"),
        ]
    )

    emit("verify_exp_f13 — benchmark B3 / fase qga bajo ruido en cinco stacks", rows)

    assert abs(float(kept) - expected_kept) < 1e-12
    assert abs(float(filtered[1]) - 0.5) < 1e-12
    assert abs(float(filtered[2]) - 0.5) < 1e-12
    assert abs(float(filtered[0])) < 1e-12 and abs(float(filtered[3])) < 1e-12
    assert abs(float(kept_counts) - expected_kept) < tolerance
    assert abs(float(np.sum(filtered_counts) - 1.0)) < 1e-12
    print("verify_exp_f13: OK — Qiskit/PennyLane/Cirq reales; QuTiP/Braket/IonQ unverified")


if __name__ == "__main__":
    main()
