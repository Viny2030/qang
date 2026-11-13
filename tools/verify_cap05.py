"""Independent check of the additions of 2026-11-01 and 2026-11-13 to chapter 5 of the book.

2026-11-01: the phase atan2(qg_Y, qg_X) under channels that shorten the vector, in QuTiP and Qiskit, and the qga
values of the 30° example. 2026-11-13: the derived quantities inside the sphere (qg_tau as concurrence, the
three-term duality, the split of the shot noise, the shots for a sign of a shortened vector, qg_L against the Bures
angle, and one basis against three for the deficit). qang is used only for the final comparison. Prints PASA/FALLA
per claim and exits with 1 if any fails.

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

# ---------------------------------------------------------------- magnitudes derivadas dentro de la esfera (2026-11-13)
print("\nMagnitudes derivadas dentro de la esfera")
import sympy as sp  # noqa: E402
from math import comb  # noqa: E402

from qiskit.quantum_info import Statevector, concurrence, partial_trace, state_fidelity  # noqa: E402

sx, sy, sz, rr = sp.symbols("x y z r", real=True)
rho_s = sp.Matrix([[1 + sz, sx - sp.I * sy], [sx + sp.I * sy, 1 - sz]]) / 2
check("sympy: 1 − r² = 4 det ρ, de modo que qg_τ = 2√det ρ y el one-tangle es qg_τ²",
      sp.simplify(4 * rho_s.det() - (1 - sx ** 2 - sy ** 2 - sz ** 2)) == 0)
check("sympy: 1 − qg_Z² = qg_τ² + V² (la varianza de un disparo en Z se reparte)",
      sp.simplify((1 - sz ** 2) - ((1 - sx ** 2 - sy ** 2 - sz ** 2) + (sx ** 2 + sy ** 2))) == 0)


def bloch(rho):
    return [float(np.real(np.trace(rho @ P))) for P in (X, Y, Z)]


rng5 = np.random.default_rng(5)
worst_c = worst_q = worst_t = 0.0
for _ in range(200):
    v = rng5.normal(size=4) + 1j * rng5.normal(size=4)
    v /= np.linalg.norm(v)
    c = concurrence(Statevector(v))
    for q in (0, 1):
        red = partial_trace(Statevector(v), [1 - q]).data
        b = bloch(red)
        tau = math.sqrt(max(0.0, 1 - sum(t * t for t in b)))
        worst_c = max(worst_c, abs(tau - c))
        worst_t = max(worst_t, abs(b[2] ** 2 + b[0] ** 2 + b[1] ** 2 + c ** 2 - 1))
        qt = qutip.Qobj(v.reshape(4, 1), dims=[[2, 2], [1, 1]]).ptrace(1 - q)   # qutip orders qubits the other way
        worst_q = max(worst_q, abs(2 * math.sqrt(max(0.0, float(np.real(np.linalg.det(qt.full()))))) - c))
check("200 estados puros de dos qubits: qg_τ de cada qubit es la concurrencia (Qiskit)", worst_c < 1e-9, f"{worst_c:.1e}")
check("los mismos estados en QuTiP: 2√det ρ_q es la concurrencia", worst_q < 1e-7, f"{worst_q:.1e}")
check("P² + V² + C² = 1 en los mismos estados (Jakob–Bergou, con C = qg_τ)", worst_t < 1e-9, f"{worst_t:.1e}")
check("Bell: qg_τ = 1; W de 3 qubits: qg_τ = √(8/9) = 0,943", round(math.sqrt(1 - 1 / 9), 3) == 0.943)
a6 = 0.6
b6 = math.sqrt(1 - a6 ** 2)
c6 = concurrence(Statevector(np.array([0, a6, b6, 0])))
check("0,6|01⟩ + 0,8|10⟩: qg_Z = ∓0,28 y qg_τ = √(1 − 0,28²) = 0,96 = 2|ab| = concurrencia, sólo con Z",
      round(abs(a6 ** 2 - b6 ** 2), 2) == 0.28 and round(math.sqrt(1 - (a6 ** 2 - b6 ** 2) ** 2), 2) == 0.96
      and abs(c6 - 0.96) < 1e-9)
qz_e, qx_e = 0.9 * math.cos(math.radians(60)), 0.9 * math.sin(math.radians(60))
tau_e = math.sqrt(1 - 0.81)
check("60°, r = 0,9: P² + V² + qg_τ² = 0,2025 + 0,6075 + 0,19 = 1, con qg_τ = 0,436",
      round(qz_e ** 2, 4) == 0.2025 and round(qx_e ** 2, 4) == 0.6075 and round(tau_e ** 2, 2) == 0.19
      and round(tau_e, 3) == 0.436 and abs(qz_e ** 2 + qx_e ** 2 + tau_e ** 2 - 1) < 1e-12)
vals = {m: (math.sqrt(1 - (m / 1000) ** 2), (m / 1000) / math.sqrt(1 - (m / 1000) ** 2),
            1 / (math.log(2) * (1 - (m / 1000) ** 2))) for m in (800, 990)}
check("peso definido, 800 m-qg: qg_τ = 0,6, qg_SNR = r/qg_τ = 1,33, qg_κ = 1/(ln2·qg_τ²) = 4,01",
      [round(v, 2) for v in vals[800]] == [0.6, 1.33, 4.01])
check("peso definido, 990 m-qg: qg_τ = 0,141, qg_SNR = 7,02, qg_κ = 72,5",
      round(vals[990][0], 3) == 0.141 and round(vals[990][1], 2) == 7.02 and round(vals[990][2], 1) == 72.5)
nsign = [math.ceil(9 * (1 - (r_ * 0.5) ** 2) / (r_ * 0.5) ** 2 - 1e-9) for r_ in (1, 0.9, 0.8, 0.5)]
check("signo a 3σ a 60°: 27, 36, 48 y 135 disparos con r = 1, 0,9, 0,8 y 0,5", nsign == [27, 36, 48, 135], str(nsign))

q_ref = math.cos(math.radians(30))
q_sh = 0.8 * q_ref
qgl = abs(math.acos(q_ref) - math.acos(q_sh)) / 2
bures = math.acos(math.sqrt(0.9))
check("30°, r = 0,8: qg_L = 0,141 rad = 22,4 qga; ángulo de Bures = 0,322 rad; la lectura en Z ve el 44 %",
      round(qgl, 3) == 0.141 and round(qgl * 1000 / (2 * math.pi), 1) == 22.4 and round(bures, 3) == 0.322
      and round(100 * qgl / bures) == 44)
psi30 = np.array([math.cos(math.radians(15)), math.sin(math.radians(15))])
pure30 = np.outer(psi30, psi30)
mixed30 = 0.8 * pure30 + 0.2 * I2 / 2
f_qk = state_fidelity(DensityMatrix(pure30), DensityMatrix(mixed30))
f_qt = qutip.fidelity(qutip.Qobj(pure30), qutip.Qobj(mixed30)) ** 2
check("la fidelidad del ejemplo es (1 + r)/2 = 0,9 (Qiskit y QuTiP)", abs(f_qk - 0.9) < 1e-6 and abs(f_qt - 0.9) < 1e-6,
      f"{f_qk:.9f} / {f_qt:.9f}")
check("N = (z/(2 qg_L))² = 114 disparos en Z para ver ese acortamiento a 3σ", math.ceil((3 / (2 * qgl)) ** 2) == 114)
blind = abs(math.acos(0.0) - math.acos(0.8 * 0.0)) / 2
pole = abs(math.acos(1.0) - math.acos(0.8)) / 2
check("en el polo qg_L iguala al ángulo de Bures; en el ecuador vale 0", abs(pole - bures) < 1e-12 and blind == 0.0)
worst_b = 1.0
for _ in range(500):
    pair = []
    for _k in range(2):
        g = rng5.normal(size=(2, 2)) + 1j * rng5.normal(size=(2, 2))
        m = g @ g.conj().T
        pair.append(m / np.trace(m).real)
    za, zb = (float(np.real(np.trace(m @ Z))) for m in pair)
    ang = math.acos(min(1.0, math.sqrt(state_fidelity(DensityMatrix(pair[0]), DensityMatrix(pair[1]), validate=False))))
    worst_b = min(worst_b, ang - abs(math.acos(za) - math.acos(zb)) / 2)
check("500 pares de estados mixtos: qg_L ≤ ángulo de Bures (la medición no aumenta la distinguibilidad)",
      worst_b > -1e-7, f"margen mínimo {worst_b:.2e}")


def unbiased_sq_var(q, n):
    """Exact mean and variance, over the binomial, of the unbiased square (n q̂² − 1)/(n − 1)."""
    p = (1 + q) / 2
    m1 = m2 = 0.0
    for k in range(n + 1):
        w = comb(n, k) * p ** k * (1 - p) ** (n - k)
        u = (n * (2 * k / n - 1) ** 2 - 1) / (n - 1)
        m1 += w * u
        m2 += w * u * u
    return m1, m2 - m1 * m1


mz, vz = unbiased_sq_var(0.8, 300)
m3, v3 = unbiased_sq_var(0.8, 100)
m0, v0 = unbiased_sq_var(0.0, 100)
check("déficit en peso definido a 800 m-qg, 300 disparos: error 0,055 sólo en Z y 0,098 en tres bases (binomial exacta)",
      round(math.sqrt(vz), 3) == 0.055 and round(math.sqrt(v3 + 2 * v0), 3) == 0.098
      and abs(mz - 0.64) < 1e-9 and abs(m3 + 2 * m0 - 0.64) < 1e-9,
      f"{math.sqrt(vz):.4f} / {math.sqrt(v3 + 2 * v0):.4f}")
ratios = []
for q_ in (0.0, 0.3, 0.8, 0.99):
    ratios.append((unbiased_sq_var(q_, 100)[1] + 2 * v0) / unbiased_sq_var(q_, 300)[1])
check("300 disparos, lecturas de 0, 300, 800 y 990 m-qg: la varianza con tres bases es al menos 3 veces la de una (3,1 a 800)",
      min(ratios) >= 3.0 and round(ratios[2], 1) == 3.1, ", ".join(f"{x:.2f}" for x in ratios))

from qang import algebra as _al, probability as _pb  # noqa: E402  final comparison only

check("qang coincide: dissipative_deficit, quantum_snr, shots_for_sign, information_curvature y geodesic_distance",
      abs(_pb.dissipative_deficit(qx_e, 0.0, qz_e) - tau_e) < 1e-12
      and abs(_pb.quantum_snr(0.8) - vals[800][1]) < 1e-12
      and [_pb.shots_for_sign(r_ * 0.5) for r_ in (1, 0.9, 0.8, 0.5)] == nsign
      and abs(_pb.information_curvature(0.8) - vals[800][2]) < 1e-12
      and abs(_al.geodesic_distance(q_ref, q_sh) - qgl) < 1e-12)

print(f"\n{sum(RESULTS)} de {len(RESULTS)} afirmaciones PASAN")
sys.exit(0 if all(RESULTS) else 1)
