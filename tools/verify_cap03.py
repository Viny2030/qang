"""Independent check of the qga additions to chapter 3 of the book (reflections and Grover on the angular scale).

Covers only the sentences added on 2026-11-01 with the angular qga scale of Chapter 1 (1000 qga per turn); the
rest of the chapter has not been through the T1-T9 review yet. Uses numpy, sympy and QuTiP, not qang. Prints
PASA/FALLA per claim and exits with 1 if any fails.

    .venv-int/Scripts/python tools/verify_cap03.py
"""

import math
import sys

import numpy as np
import qutip
import sympy as sp

RESULTS = []
QGA = 1000 / (2 * math.pi)          # qga per radian


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASA' if ok else 'FALLA'}] {name}" + (f" — {detail}" if detail else ""))


def ry_state(theta):
    return (-1j * theta / 2 * qutip.sigmay()).expm() * qutip.basis(2, 0)


# composition of two rotations: additive in qga, not in m-qg
t1, t2 = math.acos(0.3), math.acos(-0.5)
q12 = qutip.expect(qutip.sigmaz(), ry_state(t1 + t2))
check("θ1 = 201,5 qga, θ2 = 333,3 qga, suma 534,8 qga",
      [round(t1 * QGA, 1), round(t2 * QGA, 1), round((t1 + t2) * QGA, 1)] == [201.5, 333.3, 534.8])
check("QuTiP: R_y(θ1 + θ2)|0⟩ da qg_Z = −0,976 = cos(2π·534,8/1000)",
      round(q12, 3) == -0.976 and abs(q12 - math.cos(2 * math.pi * (t1 + t2) * QGA / 1000)) < 1e-12, f"{q12:.4f}")

# transition probability: gamma in qga and the Fubini-Study distance gamma/2
gamma = abs(t1 - t2)
P = abs(ry_state(t1).overlap(ry_state(t2))) ** 2
check("QuTiP: P = 0,838 = cos²(π γ_qga/1000), γ = 131,8 qga",
      round(P, 3) == 0.838 and round(gamma * QGA, 1) == 131.8 and abs(P - math.cos(math.pi * gamma * QGA / 1000) ** 2) < 1e-12)
check("distancia de Fubini–Study arccos|⟨ψ1|ψ2⟩| = γ/2", abs(math.acos(math.sqrt(P)) - gamma / 2) < 1e-12)

# Grover with a = 0.1: the flag's Bloch angle in qga
t = math.asin(math.sqrt(0.1))
angles = [(2 * k + 1) * 2 * t * QGA for k in range(4)]
check("a = 0,1: t = 51,2 qga, 2t = 102,4 qga, paso 4t = 204,8 qga",
      [round(t * QGA, 1), round(2 * t * QGA, 1), round(4 * t * QGA, 1)] == [51.2, 102.4, 204.8])
check("en k = 2 la bandera está en 512,1 qga, 12 qga más allá del polo (500 qga)",
      round(angles[2], 1) == 512.1 and round(angles[2] - 500) == 12)
reads = [round(1000 * qutip.expect(qutip.sigmaz(), ry_state(a / QGA))) for a in angles]
check("QuTiP: las lecturas de k = 0..3 son +800, −352, −997, −206 m-qg", reads == [800, -352, -997, -206], f"{reads}")

# constant Fubini-Study arc per iteration: Grover in the 2D plane |bad>, |good>
N, M = 10, 1
s = np.ones(N) / math.sqrt(N)
oracle = np.eye(N)
oracle[0, 0] = -1
diff = 2 * np.outer(s, s) - np.eye(N)
psi, arcs = s.copy(), []
for _ in range(4):
    new = diff @ oracle @ psi
    arcs.append(math.acos(min(1.0, abs(psi @ new))))
    psi = new
check("Grover con N = 10, M = 1: cada iteración recorre el mismo arco 2t de Fubini–Study",
      max(abs(a - 2 * math.asin(math.sqrt(M / N))) for a in arcs) < 1e-12, f"{arcs[0]:.6f} rad")

# Fisher information about the flag's initial angle at depth m: m² rad⁻² = m²·4π²·10⁻⁶ qga⁻²
th, mm = sp.symbols("theta m", positive=True)
p0 = sp.cos(mm * th / 2) ** 2
F = sp.diff(p0, th) ** 2 / (p0 * (1 - p0))
vals = [float(F.subs({mm: m, th: v})) for m in (1, 3, 5) for v in (0.2, 0.9, 1.7)]
check("sympy: la información sobre θ0 a profundidad m es m², sin depender de θ0",
      max(abs(v - m ** 2) for v, m in zip(vals, [1, 1, 1, 3, 3, 3, 5, 5, 5])) < 1e-9)
check("resolución 159,15/(m√N) qga: 15,9 qga sin iterar y 3,2 qga a profundidad 5 (N = 100)",
      round(QGA / 10, 1) == 15.9 and round(QGA / (5 * 10), 1) == 3.2)

# k* in qga
for a, kstar in ((0.1, 2), (0.3, 1)):
    tt = math.asin(math.sqrt(a))
    check(f"a = {a}: k* = round(π/4t − ½) = round(125/t_qga − ½) = {kstar}",
          round(math.pi / (4 * tt) - 0.5) == kstar == round(125 / (tt * QGA) - 0.5))
check("sympy: π/(4t) = 125/t_qga con t = 2π t_qga/1000",
      sp.simplify(sp.pi / (4 * 2 * sp.pi * sp.Symbol("x") / 1000) - 125 / sp.Symbol("x")) == 0)
check("la reflexión de Householder es una rotación de π = 500 qga", round(math.pi * QGA) == 500)

print(f"\n{sum(RESULTS)} de {len(RESULTS)} afirmaciones PASAN")
sys.exit(0 if all(RESULTS) else 1)
