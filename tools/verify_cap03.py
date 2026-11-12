"""Independent check of the qga additions to chapter 3 of the book (reflections and Grover on the angular scale).

Covers only the sentences added on 2026-11-01 with the angular qga scale of Chapter 1 (1000 qga per turn) and those
added on 2026-11-12 with the derived quantities qg_L, qg_SNR, qg_kappa and qg_tau; the rest of the chapter has not been
through the T1-T9 review yet. Uses numpy, sympy and QuTiP; qang only in the last comparison. Prints
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

# ---------------------------------------------------------------- magnitudes derivadas (2026-11-12)
print("\nMagnitudes derivadas en Grover: qg_L, qg_SNR, qg_κ, qg_τ")
s1 = (math.sqrt(0.725) * qutip.basis(2, 0) + math.sqrt(0.275) * qutip.basis(2, 1))   # q = 0.45, phase 0
s2 = (math.sqrt(0.725) * qutip.basis(2, 0) - math.sqrt(0.275) * qutip.basis(2, 1))   # q = 0.45, phase pi
ov = abs(s1.overlap(s2))
check("q = 0,45 con fases 0 y π: |⟨ψ1|ψ2⟩| = 0,45, qg_L = 1,104 rad = 175,7 qga, Bhattacharyya en Z = 0",
      round(ov, 6) == 0.45 and round(math.acos(ov), 3) == 1.104 and round(math.acos(ov) * QGA, 1) == 175.7
      and abs(math.acos(min(1.0, math.sqrt(0.725 * 0.725) + math.sqrt(0.275 * 0.275)))) < 1e-7)
check("qg_L(0,3; −0,5) = 0,414 rad = 65,9 qga", round(gamma / 2, 3) == 0.414 and round(gamma / 2 * QGA, 1) == 65.9)

rows = []
for k in range(4):
    zk = qutip.expect(qutip.sigmaz(), ry_state((2 * k + 1) * 2 * t))
    snr = abs(zk) / math.sqrt(1 - zk ** 2)
    kap = 1 / (math.log(2) * (1 - zk ** 2))
    rows.append((round(1000 * zk, 1), round((2 * k + 1) * 2 * t * QGA, 1), round((1 - zk) / 2, 4), round(snr, 2), round(kap, 2)))
expected = [(800.0, 102.4, 0.1, 1.33, 4.01), (-352.0, 307.2, 0.676, 0.38, 1.65),
            (-997.1, 512.1, 0.9986, 13.15, 250.83), (-206.4, 716.9, 0.6032, 0.21, 1.51)]
check("QuTiP: tabla tbl-grover-derivadas (qg_k, ángulo, P_k, qg_SNR, qg_κ)", rows == expected, f"{rows}")
z_after = qutip.expect(qutip.sigmaz(), ry_state((500 + 12.1) / QGA))
z_before = qutip.expect(qutip.sigmaz(), ry_state((500 - 12.1) / QGA))
check("qg_SNR y qg_κ valen lo mismo 12 qga antes y después del polo", abs(z_after - z_before) < 1e-12)
n_sign = [math.ceil((3 / (abs(z) / math.sqrt(1 - z * z))) ** 2 - 1e-9)
          for z in (math.cos(5 * 2 * t), math.cos(3 * 2 * t))]
check("disparos para el signo a 3σ: 1 en k = 2 y 64 en k = 1", n_sign == [1, 64], f"{n_sign}")

# depolarizing noise shortens the flag's qg vector by V = 1 - p per iteration (QuTiP channel)
def depolarize(rho, p):
    return (1 - p) * rho + p * qutip.qeye(2) / 2


def tau_after(p, k):
    rho = ry_state(0.7) * ry_state(0.7).dag()
    step = (-1j * 0.3 * qutip.sigmay()).expm()
    for _ in range(k):
        rho = depolarize(step * rho * step.dag(), p)
    r2 = sum(qutip.expect(op, rho) ** 2 for op in (qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()))
    return math.sqrt(max(0.0, 1 - r2))


check("QuTiP: con despolarización, qg_τ = √(1 − V^{2k}), V = 1 − p (p = 0,025, k = 10)",
      abs(tau_after(0.025, 10) - math.sqrt(1 - 0.975 ** 20)) < 1e-12)
check("V = 0,975: qg_τ = 0,63 tras 10 iteraciones y 0,80 tras 20",
      round(tau_after(0.025, 10), 2) == 0.63 and round(tau_after(0.025, 20), 2) == 0.80)
check("en k = 1/p, qg_τ > 0,93 (p = 0,01 y p = 0,05)", tau_after(0.01, 100) > 0.93 and tau_after(0.05, 20) > 0.93,
      f"{tau_after(0.01, 100):.4f}, {tau_after(0.05, 20):.4f}")

from qang import algebra, probability  # noqa: E402  final comparison only
q2 = algebra.chebyshev_trajectory(0.8, 2)
check("qang coincide: chebyshev_trajectory, quantum_snr, information_curvature, grover_fisher_information, geodesic_distance",
      round(1000 * q2, 1) == -997.1 and round(probability.quantum_snr(q2), 2) == 13.15
      and round(probability.information_curvature(q2), 1) == 250.8
      and round(algebra.grover_fisher_information(0.8, 5), 1) == 69.4
      and abs(algebra.geodesic_distance(0.3, -0.5) - gamma / 2) < 1e-12)

# ---------------------------------------------------------------- versión Integral del capítulo 3 (2026-11-12)
print("\nLa trayectoria como oscilación armónica")
check("sympy: 1 − qg0² = 4a(1 − a) con qg0 = 1 − 2a",
      sp.simplify(1 - (1 - 2 * sp.Symbol("a")) ** 2 - 4 * sp.Symbol("a") * (1 - sp.Symbol("a"))) == 0)
check("resolución 2000√(a(1−a))/(m√N): 60 y 12 m-qg con a = 0,1, N = 100, m = 1 y 5",
      round(2000 * math.sqrt(0.09) / 10) == 60 and round(2000 * math.sqrt(0.09) / 50) == 12)
losses = [1 - abs(ry_state(math.pi + d / QGA).overlap(qutip.basis(2, 1))) ** 2 for d in (12.1, 25, 50)]
check("QuTiP: quedar a 12,1 / 25 / 50 qga del polo cuesta 0,14 % / 0,6 % / 2,4 % de éxito",
      [round(100 * x, 2) for x in losses[:1]] == [0.14] and round(100 * losses[1], 1) == 0.6 and round(100 * losses[2], 1) == 2.4,
      f"{[round(100 * x, 3) for x in losses]}")
dd = sp.Symbol("d")
check("sympy: cos²(δ/2) = 1 − δ²/4 + O(δ⁴)", sp.series(sp.cos(dd / 2) ** 2, dd, 0, 4).removeO() == 1 - dd ** 2 / 4)
z2 = math.cos(5 * 2 * t)
d_sol = math.acos(abs(ry_state(5 * 2 * t).overlap(qutip.basis(2, 1))))
check("QuTiP: distancia de Fubini–Study de la bandera en k = 2 al polo |1⟩: 6,04 qga, la mitad del exceso",
      round(d_sol * QGA, 2) == 6.04 and abs(d_sol * QGA - (5 * 2 * t * QGA - 500) / 2) < 1e-9)


def h2(p):
    return 0.0 if p in (0.0, 1.0) else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


ent = [round(h2((1 + math.cos((2 * k + 1) * 2 * t)) / 2), 3) for k in range(4)]
check("qg_S en k = 0..3: 0,469, 0,909, 0,016, 0,969 bits", ent == [0.469, 0.909, 0.016, 0.969], f"{ent}")
check("paso de 204,8 qga por iteración con a = 0,1", round(4 * t * QGA, 1) == 204.8)
# neutral-meson mixing with equal widths: two-level rotation, asymmetry = cos(dm t)
dm = 0.7
ok = True
for tt in (0.3, 1.1, 2.9, 4.0):
    U = (-1j * dm * tt / 2 * qutip.sigmax()).expm()
    psi = U * qutip.basis(2, 0)
    unmixed, mixed = abs(psi[0, 0]) ** 2, abs(psi[1, 0]) ** 2
    ok &= abs((unmixed - mixed) / (unmixed + mixed) - math.cos(dm * tt)) < 1e-12
check("QuTiP: oscilación de dos niveles con anchuras iguales: la asimetría de mezcla es cos(Δm t), un qg_Z", ok)
check("dos estados ortogonales están a π/2 = 250 qga", round(math.acos(abs(qutip.basis(2, 0).overlap(qutip.basis(2, 1)))) * QGA) == 250)
check("qang coincide: geodesic_distance(qg_2, −1) = 6,04 qga", round(algebra.geodesic_distance(q2, -1.0) * QGA, 2) == 6.04)

print(f"\n{sum(RESULTS)} de {len(RESULTS)} afirmaciones PASAN")
sys.exit(0 if all(RESULTS) else 1)
