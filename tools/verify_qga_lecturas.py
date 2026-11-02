"""Independent check of the qga sentences added on 2026-11-01 to chapters 9, 16, 19 and 23 of the book.

The angular qga scale of Chapter 1 has 1000 qga per turn. Each check recomputes the number the text quotes, with
numpy, scipy, sympy and QuTiP; qang is not used. Prints PASA/FALLA per claim and exits with 1 if any fails.

    .venv-int/Scripts/python tools/verify_qga_lecturas.py
"""

import math
import sys

import numpy as np
import qutip
import sympy as sp
from scipy.optimize import brentq

RESULTS = []
QGA = 1000 / (2 * math.pi)


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASA' if ok else 'FALLA'}] {name}" + (f" — {detail}" if detail else ""))


# chapter 9: the mixing angle is a quarter of the qang angle; its spread with N = 1000
sigma = 1 / (4 * math.sqrt(1000.5))
check("cap. 9: σ = 1/(4√1000,5) = 0,0079 rad = 1,26 qga; θ ≤ π/4 es θ ≤ 125 qga",
      round(sigma, 4) == 0.0079 and round(sigma * QGA, 2) == 1.26 and round(math.pi / 4 * QGA) == 125)
th = 0.3
qz = math.cos(2 * th) ** 2 - math.sin(2 * th) ** 2
check("cap. 9: en el máximo qg_Z = cos 4θ, así que θ_qga = ϑ_qga/4", abs(math.acos(qz) * QGA / 4 - th * QGA) < 1e-9)

# chapter 12: the Bell pair's Schmidt angle and its qubit's qang angle
alpha = math.pi / 4
qz_local = math.cos(alpha) ** 2 - math.sin(alpha) ** 2          # cos²α − sin²α for one qubit of the resource
check("cap. 12: el par de Bell tiene α = 125 qga y su qubit (qg_Z = 0) está en 250 qga",
      round(alpha * QGA) == 125 and abs(qz_local) < 1e-12 and round(math.acos(qz_local) * QGA) == 250
      and abs(math.acos(qz_local) - 2 * alpha) < 1e-12)

# chapter 14: the polar errors of the devices as angles, and the success law in qga
errs = [math.acos(1 - d / 1000) for d in (34, 138)]
check("cap. 14: 34 y 138 m-qg junto al polo son 0,26 y 0,53 rad, 41,6 y 84,6 qga",
      [round(e, 2) for e in errs] == [0.26, 0.53] and [round(e * QGA, 1) for e in errs] == [41.6, 84.6])
check("cap. 14: éxito = cos²(φ/2) = cos²(π φ_qga/1000); la paradoja (éxito 0) en 500 qga",
      max(abs(math.cos(p / 2) ** 2 - math.cos(math.pi * p * QGA / 1000) ** 2) for p in np.linspace(0, math.pi, 13)) < 1e-12
      and abs(math.cos(math.pi * 500 / 1000) ** 2) < 1e-12)

# chapter 16: the Shor-Preskill threshold as a reading and an angle
h = lambda q: -q * math.log2(q) - (1 - q) * math.log2(1 - q)  # noqa: E731
Q = brentq(lambda q: 1 - 2 * h(q), 0.05, 0.2)
ang = 2 * math.asin(math.sqrt(Q))
check("cap. 16: 1 − 2h(Q) = 0 en Q = 11,0 %, m = 780 m-qg, 2 arcsin √Q = 0,676 rad = 38,7° = 107,6 qga",
      round(100 * Q, 1) == 11.0 and round(1000 * (1 - 2 * Q)) == 780 and round(ang, 3) == 0.676
      and round(math.degrees(ang), 1) == 38.7 and round(ang * QGA, 1) == 107.6, f"{ang:.4f} rad")
check("cap. 16: el 0,687 rad (39,38°) del .docx no es 2 arcsin √0,110", abs(2 * math.asin(math.sqrt(0.110)) - 0.687) > 0.01,
      f"2 arcsin √0,110 = {2 * math.asin(math.sqrt(0.110)):.4f} rad")

# chapter 19: ergotropy of a pure state = ω sin²(π θ_qga/1000), checked against the passive state in QuTiP
worst = 0.0
for a in (0, 100, 250, 400, 500):
    t = 2 * math.pi * a / 1000
    st = math.cos(t / 2) * qutip.basis(2, 0) + math.sin(t / 2) * qutip.basis(2, 1)
    H = qutip.Qobj(np.diag([0.0, 1.0]))                                   # ω = 1, ground state |0>
    rho = st * st.dag()
    passive = qutip.basis(2, 0) * qutip.basis(2, 0).dag()               # pure state: passive = ground
    W = qutip.expect(H, rho) - qutip.expect(H, passive)
    worst = max(worst, abs(W - math.sin(math.pi * a / 1000) ** 2))
check("cap. 19 (QuTiP): W = ω sin²(π θ_qga/1000); 0 en 0 qga, ½ en 250, 1 en 500", worst < 1e-12, f"error máx {worst:.1e}")

# chapter 23: the qga angle is a flat Fisher coordinate; changing units rescales the gradient by a constant
a = sp.symbols("a", positive=True)
p0 = sp.cos(sp.pi * a / 1000) ** 2
F = sp.diff(p0, a) ** 2 / (p0 * (1 - p0))
check("cap. 23 (sympy): F = 4π²·10⁻⁶ qga⁻², constante",
      max(abs(float(F.subs(a, v)) - 4 * math.pi ** 2 * 1e-6) for v in (30, 170, 260, 410)) < 1e-15)
rng = np.random.default_rng(3)
g_rad = rng.normal(size=5000)                                 # gradients w.r.t. radians at random initializations
g_qga = g_rad * (2 * math.pi / 1000)                          # the same gradients w.r.t. qga
check("cap. 23: Var(∂C/∂θ_qga) / Var(∂C/∂θ) = (2π/1000)² para cualquier muestra: la meseta no cambia de escala relativa",
      abs(np.var(g_qga) / np.var(g_rad) - (2 * math.pi / 1000) ** 2) < 1e-15
      and abs(np.var(g_qga) / np.mean(g_qga) ** 2 - np.var(g_rad) / np.mean(g_rad) ** 2) < 1e-6)

# chapter 33: the angle's spread in qga, independent of p0 (binomial sampling)
rng33 = np.random.default_rng(33)
N33, worst33 = 2000, 0.0
for p in (0.05, 0.3, 0.7, 0.95):
    k = rng33.binomial(N33, p, 40000)
    ang = 2 * np.arccos(np.sqrt(k / N33)) * QGA
    worst33 = max(worst33, abs(np.std(ang) / (QGA / math.sqrt(N33)) - 1))
check("cap. 33: σ del ángulo = 159,15/√N qga para p0 = 0,05 … 0,95", worst33 < 0.05, f"desvío relativo máx {worst33:.3f}")

# chapter 34: the Wigner angle of 0.6c x̂ then 0.8c ŷ, recomputed from 4x4 boosts, in qga
def boost(v):
    v = np.asarray(v, float)
    b2 = v @ v
    g = 1 / math.sqrt(1 - b2)
    L = np.eye(4)
    L[0, 0] = g
    L[0, 1:] = L[1:, 0] = g * v
    L[1:, 1:] += (g - 1) * np.outer(v, v) / b2
    return L


Lc = boost([0, 0.8, 0]) @ boost([0.6, 0, 0])
u = Lc[1:, 0] / Lc[0, 0]                                     # velocity of the composed boost
R = np.linalg.inv(boost(u)) @ Lc                             # what is left is a rotation
omega = math.acos((np.trace(R[1:, 1:]) - 1) / 2)
check("cap. 34: Ω_W = 0,3303 rad (18,92°) = 52,6 qga", round(omega, 4) == 0.3303 and round(omega * QGA, 1) == 52.6,
      f"{omega:.5f} rad")
check("cap. 34: una fase de π son 500 qga (la escala qga es periódica: 1000 qga por vuelta)", round(math.pi * QGA) == 500)

print(f"\n{sum(RESULTS)} de {len(RESULTS)} afirmaciones PASAN")
sys.exit(0 if all(RESULTS) else 1)
