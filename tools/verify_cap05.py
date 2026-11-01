"""Independent check of the additions of 2026-11-01 to chapter 5 of the book (the azimuthal phase and the radius).

Covers only the new paragraphs: the phase atan2(qg_Y, qg_X) under channels that shorten the vector, in QuTiP and
Qiskit, and the qga values of the 30° example. qang is not used. Prints PASA/FALLA per claim and exits with 1 if
any fails.

    .venv-int/Scripts/python tools/verify_cap05.py
"""

import math
import sys

import numpy as np
import qutip
from qiskit.quantum_info import DensityMatrix, Kraus, Pauli

RESULTS = []
QGA = 1000 / 360


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASA' if ok else 'FALLA'}] {name}" + (f" — {detail}" if detail else ""))


th, ph, g = 1.0, 0.7, 0.3
psi = np.array([math.cos(th / 2), np.exp(1j * ph) * math.sin(th / 2)])
I2, X, Y, Z = np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])
channels = {
    "despolarizante": [math.sqrt(1 - 3 * g / 4) * I2] + [math.sqrt(g / 4) * P for P in (X, Y, Z)],
    "desfase": [math.sqrt(1 - g / 2) * I2, math.sqrt(g / 2) * Z],
    "T1": [np.array([[1, 0], [0, math.sqrt(1 - g)]]), np.array([[0, math.sqrt(g)], [0, 0]])],
    "inversión de bit (0,15)": [math.sqrt(1 - g / 2) * I2, math.sqrt(g / 2) * X],
}


def phase_qiskit(kraus):
    rho = DensityMatrix(np.outer(psi, psi.conj())).evolve(Kraus(kraus))
    return math.atan2(float(np.real(rho.expectation_value(Pauli("Y")))), float(np.real(rho.expectation_value(Pauli("X")))))


def phase_qutip(kraus):
    rho = qutip.Qobj(np.outer(psi, psi.conj()))
    out = sum(qutip.Qobj(K) * rho * qutip.Qobj(K).dag() for K in kraus)
    return math.atan2(qutip.expect(qutip.sigmay(), out), qutip.expect(qutip.sigmax(), out))


for name in ("despolarizante", "desfase", "T1"):
    a, b = phase_qiskit(channels[name]), phase_qutip(channels[name])
    check(f"{name}: la fase atan2(qg_Y, qg_X) se conserva (Qiskit y QuTiP)", abs(a - ph) < 1e-12 and abs(b - ph) < 1e-12,
          f"{a:.6f} / {b:.6f}")
a, b = phase_qiskit(channels["inversión de bit (0,15)"]), phase_qutip(channels["inversión de bit (0,15)"])
check("inversión de bit 0,15: la fase 0,70 rad se lee 0,53 rad (Qiskit y QuTiP)",
      round(a, 2) == 0.53 and round(b, 2) == 0.53, f"{a:.4f} / {b:.4f}")
check("30° = 83,3 qga y 46,1° = 128,1 qga", round(30 * QGA, 1) == 83.3 and round(46.1 * QGA, 1) == 128.1)
check("el ejemplo: r = 0,8 a 30° da qg_Z = 0,693 y arccos lee 46,1°",
      round(0.8 * math.cos(math.radians(30)), 3) == 0.693
      and round(math.degrees(math.acos(0.8 * math.cos(math.radians(30)))), 1) == 46.1)

print(f"\n{sum(RESULTS)} de {len(RESULTS)} afirmaciones PASAN")
sys.exit(0 if all(RESULTS) else 1)
