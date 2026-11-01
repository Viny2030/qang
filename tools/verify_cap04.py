"""Independent check of the additions of 2026-11-01 to chapter 4 of the book (the azimuthal phase and the qga scale).

Covers only the new azimuthal-phase subsection: the delta-method variance of atan2(qg_Y, qg_X), the bound used by
the author's working version, and the variance of qg_X. Sympy for the formula; sampling on states prepared in QuTiP
and Qiskit. qang is not used. Prints PASA/FALLA per claim and exits with 1 if any fails.

    .venv-int/Scripts/python tools/verify_cap04.py
"""

import math
import sys

import numpy as np
import qutip
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASA' if ok else 'FALLA'}] {name}" + (f" — {detail}" if detail else ""))


def delta_var(th, ph, n):
    return (1 - 0.5 * math.sin(th) ** 2 * math.sin(2 * ph) ** 2) / (n * math.sin(th) ** 2)


# 1. the delta-method formula, symbolically
t, f, n = sp.symbols("theta phi N", positive=True)
x, y = sp.sin(t) * sp.cos(f), sp.sin(t) * sp.sin(f)
X, Y = sp.symbols("X Y", real=True)
phi_hat = sp.atan2(Y, X)
gx, gy = sp.diff(phi_hat, X).subs({X: x, Y: y}), sp.diff(phi_hat, Y).subs({X: x, Y: y})
var = gx ** 2 * (1 - x ** 2) / n + gy ** 2 * (1 - y ** 2) / n
target = (1 - sp.Rational(1, 2) * sp.sin(t) ** 2 * sp.sin(2 * f) ** 2) / (n * sp.sin(t) ** 2)
pts = [(0.4, 0.3), (1.2, 0.785), (1.5708, 0.785), (2.3, 2.0), (0.9, 4.0)]
err = max(abs(float((var - target).subs({t: a, f: b, n: 1}))) for a, b in pts)
check("sympy: Var(atan2) por método delta = [1 − ½ sin²θ sin²2φ]/(N sin²θ)", err < 1e-12, f"error máx {err:.1e}")
check("la cota 1/(N sin²θ) duplica la varianza en el ecuador a φ = 45°",
      abs((1 / 1) / delta_var(math.pi / 2, math.pi / 4, 1) - 2) < 1e-12)
check("la cota es exacta sobre los ejes (φ = 0, 90°)",
      abs(delta_var(1.0, 0.0, 1) * math.sin(1.0) ** 2 - 1) < 1e-12
      and abs(delta_var(1.0, math.pi / 2, 1) * math.sin(1.0) ** 2 - 1) < 1e-12)


# 2. sampling: N shots in each transverse basis, probabilities from two libraries
def probs_qutip(th, ph):
    st = (math.cos(th / 2) * qutip.basis(2, 0) + np.exp(1j * ph) * math.sin(th / 2) * qutip.basis(2, 1)).unit()
    return qutip.expect(qutip.sigmax(), st), qutip.expect(qutip.sigmay(), st)


def probs_qiskit(th, ph):
    qc = QuantumCircuit(1)
    qc.ry(th, 0)
    qc.rz(ph, 0)
    sv = Statevector(qc)
    return float(np.real(sv.expectation_value(Pauli("X")))), float(np.real(sv.expectation_value(Pauli("Y"))))


rng = np.random.default_rng(11)
N, R = 2000, 100000
for name, src in (("QuTiP", probs_qutip), ("Qiskit", probs_qiskit)):
    worst = 0.0
    for th, ph in ((math.pi / 2, math.pi / 4), (math.pi / 2, 0.0), (0.6, 1.0), (0.3, 0.3)):
        qx, qy = src(th, ph)
        kx, ky = rng.binomial(N, (1 + qx) / 2, R), rng.binomial(N, (1 + qy) / 2, R)
        est = np.arctan2(2 * ky / N - 1, 2 * kx / N - 1)
        v = np.var(np.angle(np.exp(1j * (est - ph))))
        worst = max(worst, abs(v / delta_var(th, ph, N) - 1))
    check(f"{name}: la varianza muestreada de φ̂ coincide con la fórmula exacta (4 estados, N = 2000)", worst < 0.03,
          f"desvío relativo máx {worst:.3f}")

# 3. qg_X has variance (1 − qg_X²)/N ≤ 1/N
qx, _ = probs_qutip(1.1, 0.4)
kx = rng.binomial(N, (1 + qx) / 2, R)
v = np.var(2 * kx / N - 1)
check("Var(q̂g_X) = (1 − qg_X²)/N ≤ 1/N", abs(v * N / (1 - qx ** 2) - 1) < 0.02 and (1 - qx ** 2) <= 1)
check("en el ecuador y sobre un eje la desviación de φ es 159,15/√N qga",
      round(math.sqrt(delta_var(math.pi / 2, 0.0, 1)) * 1000 / (2 * math.pi), 2) == 159.15)

print(f"\n{sum(RESULTS)} de {len(RESULTS)} afirmaciones PASAN")
sys.exit(0 if all(RESULTS) else 1)
