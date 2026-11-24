r"""F12 — interoperabilidad: la misma fase angular leída con Qiskit, PennyLane y Cirq.

Hipótesis H12 (`unverified`): la fase de una compuerta paramétrica se lee idéntica en qga en los tres frameworks reales instalados; la ejecución en hardware sigue sin afirmarse.

Teorema T12 (`exact`): para $|+\rangle$ seguido de $RZ(\varphi)$, el vector de Bloch es $(\cos\varphi,\sin\varphi,0)$ y `azimuth_qga` es invariante bajo $q_X,q_Y\to\lambda q_X,\lambda q_Y$ con $\lambda>0$; `qecho.invert_readout` recupera el sesgo afín. Las tres convenciones usan $RZ(\varphi)=\exp(-i\varphi Z/2)$ y coinciden hasta fase global.

Circuito testigo: 1 qubit, $\varphi=125$ qga, 65_536 shots, 600 estimaciones, umbral 5 sigma = 3.108 qga y control +50 qga. Los statevectors se obtienen con `qiskit.quantum_info.Statevector`, `pennylane.state` y `cirq.final_state_vector`.
"""
from __future__ import annotations

import cmath
import math
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qang.geometric import bloch_from_state
from qang.qecho import invert_readout, readout_affine
from qang.qfalsify import verify_registration
from qang.qga import QGA_PER_TURN, azimuth_qga
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

PHASE_QGA: float = 125.0
CONTROL_OFFSET_QGA: float = 50.0
SCALE_LAMBDA: float = 0.7
FRAMEWORKS: tuple[str, ...] = ("qiskit", "pennylane", "cirq")

PREDICTION: dict = {
    "exp": "F12",
    "domain": "librerias_cuanticas",
    "frameworks": list(FRAMEWORKS),
    "mode": "real_libraries_statevector",
    "real_libraries_required": True,
    "circuit": "H then RZ(phi), readout X/Y",
    "phase_qga": PHASE_QGA,
    "shots": SHOTS,
    "n_estimates": N_ESTIMATES,
    "delta_5sigma_qga": DELTA_5SIGMA_QGA,
    "control_offset_qga": CONTROL_OFFSET_QGA,
    "seed": SEED,
    "criterion": "three real-library azimuths agree within 1e-3 qga; 600/600 hits and control refutada",
}
REGISTRATION_SHA256: str = "44e9bbfc85602d6891654b22a2d89f8b962ad6e3157232fd17ff9a282e128682"


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


def main() -> None:
    """Ejecuta F12 y emite filas con etiqueta de evidencia."""
    assert verify_registration(PREDICTION, REGISTRATION_SHA256)

    rng = np.random.default_rng(SEED)
    phi = 2.0 * math.pi * PHASE_QGA / QGA_PER_TURN
    adapters = (qiskit_bloch, pennylane_bloch, cirq_bloch)
    blochs = [adapter(phi) for adapter in adapters]
    azimuths = [azimuth_qga(float(b[0]), float(b[1])) for b in blochs]
    versions = {name: version(name) for name in FRAMEWORKS}
    max_deviation = float(max(np.max(np.abs(b - blochs[0])) for b in blochs))
    scaled = azimuth_qga(SCALE_LAMBDA * float(blochs[0][0]), SCALE_LAMBDA * float(blochs[0][1]))
    m = QGA_PER_TURN * float(blochs[0][2])
    m_recovered = invert_readout(readout_affine(m, E01, E10), E01, E10)

    rows: list[tuple[str, object, str]] = [
        ("H12_claim", "las tres librerias concuerdan en statevector; hardware real no afirmado", "unverified"),
        ("T12_phase_qga", PHASE_QGA, "exact"),
        ("T12_qangphi_phase_qga", round(phase_qga_from_complex(cmath.exp(2j * math.pi * PHASE_QGA / QGA_PER_TURN)), 9), "exact"),
        ("real_library_versions", versions, "exact"),
        ("max_bloch_deviation", max_deviation, "simulado"),
    ]
    for name, bloch, azimuth in zip(FRAMEWORKS, blochs, azimuths):
        rows.extend(
            [
                (f"{name}_bloch", np.round(bloch, 9).tolist(), "simulado"),
                (f"{name}_azimuth_qga", round(azimuth, 9), "simulado"),
            ]
        )
    rows.extend(
        [
            ("T12_scaled_azimuth_qga", round(scaled, 9), "exact"),
            ("T12_readout_inversion_m_qg", round(m_recovered, 9), "exact"),
            ("registration_sha256", REGISTRATION_SHA256[:16], "exact"),
            ("shots", SHOTS, "exact"),
            ("sigma_qga", round(SIGMA_QGA, 6), "exact"),
            ("5sigma_qga", DELTA_5SIGMA_QGA, "exact"),
        ]
    )

    samples = noisy_phase_samples(PHASE_QGA / QGA_PER_TURN, rng)
    verdict, hits, p_value = falsify_angular(PHASE_QGA, samples)
    control_verdict, control_hits, control_p = falsify_angular(
        (PHASE_QGA + CONTROL_OFFSET_QGA) % QGA_PER_TURN, samples
    )
    rows.extend(
        [
            ("mean_azimuth_qga", round(float(np.mean(samples)), 6), "simulado"),
            ("hits", f"{hits}/{N_ESTIMATES}", "simulado"),
            ("p_upper", p_value, "simulado"),
            ("decision", verdict, "simulado"),
            ("control", f"{control_hits}/{N_ESTIMATES} {control_verdict}", "simulado"),
            ("control_p", control_p, "simulado"),
        ]
    )

    emit("verify_exp_f12 — interoperabilidad / fase qga en tres librerias reales", rows)

    for azimuth in azimuths:
        assert circular_distance_qga(azimuth, PHASE_QGA) < 1e-3
    assert max_deviation < 1e-6
    assert circular_distance_qga(scaled, PHASE_QGA) < 1e-9
    assert abs(m_recovered - m) < 1e-9
    assert hits == N_ESTIMATES
    assert control_hits == 0
    assert verdict == "confirmada" and control_verdict == "refutada"
    print("verify_exp_f12: OK — statevector real; hardware unverified")


if __name__ == "__main__":
    main()
