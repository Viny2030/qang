"""Check of appendix G of the book: relativistic energy and antimatter as qang readings.

Every number the appendix quotes is computed here, without qang except for the final comparison with qang.qga:
the relativistic angle (cos θ_r = 1/γ), the reading of the Dirac matrix β = γ⁰ on free eigenstates (⟨β⟩ = mc²/E,
negative on the negative-energy branch), its invariance under a change of representation, the action of C, P, T
and CPT on that reading, the "Pythagoras qubit" (p0 = 1/γ²), collinear boosts in rapidity and in qga, and the
shot noise of the relativistic angle, sampled in Qiskit and QuTiP. Prints PASA/FALLA per claim and the tables.

    .venv-int/Scripts/python tools/verify_apendice_g.py
"""

import math
import sys

import numpy as np
import qutip
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Pauli, Statevector

RESULTS = []
QGA = 1000 / (2 * math.pi)


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASA' if ok else 'FALLA'}] {name}" + (f" — {detail}" if detail else ""))


# ---- 1. the relativistic angle -------------------------------------------------------------------------------
b, g = sp.symbols("beta gamma", positive=True)
gam = 1 / sp.sqrt(1 - b ** 2)
check("sympy: (1/γ)² + β² = 1 y tan θ_r = βγ = p/(mc)", sp.simplify((1 / gam) ** 2 + b ** 2 - 1) == 0
      and sp.simplify(b / (1 / gam) - b * gam) == 0)
m_, E_ = sp.symbols("m E", positive=True)
T_ = E_ - m_
check("sympy: p1 = (E − mc²)/2E = T/2E y p0 = (E + mc²)/2E = cos²(θ_r/2)",
      sp.simplify((1 - m_ / E_) / 2 - T_ / (2 * E_)) == 0 and sp.simplify((1 + m_ / E_) / 2 - (E_ + m_) / (2 * E_)) == 0)

print("\nTabla G1: β, γ, θ_r (°), θ_r (qga), lectura 1000/γ (m-qg), T/mc²")
table = []
for beta in (0.0, 0.5, 0.6, 0.8, 0.99, 0.999):
    gg = 1 / math.sqrt(1 - beta ** 2)
    th = math.acos(1 / gg)
    row = (beta, gg, math.degrees(th), th * QGA, 1000 / gg, gg - 1)
    table.append(row)
    print("  β=%.3f  γ=%.4f  θ_r=%.2f°  %.1f qga  m=%.1f  T/mc²=%.4f" % row)
check("tabla G1: β = 0,6 → γ = 1,25, 36,87°, 102,4 qga, +800 m-qg",
      round(table[2][1], 2) == 1.25 and round(table[2][2], 2) == 36.87 and round(table[2][3], 1) == 102.4
      and round(table[2][4]) == 800)
check("tabla G1: β = 0,8 → γ = 1,667, 53,13°, 147,6 qga, +600 m-qg",
      round(table[3][1], 3) == 1.667 and round(table[3][2], 2) == 53.13 and round(table[3][3], 1) == 147.6
      and round(table[3][4]) == 600)
check("tabla G1: β = 0,99 → γ = 7,089, 81,89°, 227,5 qga, +141,1 m-qg",
      round(table[4][1], 3) == 7.089 and round(table[4][2], 2) == 81.89 and round(table[4][3], 1) == 227.5
      and round(table[4][4], 1) == 141.1)
check("límite β → 1: 90° = 250 qga, lectura 0 (el ecuador): la luz no tiene masa en reposo",
      round(math.pi / 2 * QGA) == 250)

# ---- 2. the Dirac matrix β as a qubit Z --------------------------------------------------------------------
I2, Z2 = np.eye(2), np.zeros((2, 2))
sx, sy, sz = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])
BETA = np.block([[I2, Z2], [Z2, -I2]])
ALPHA = [np.block([[Z2, s], [s, Z2]]) for s in (sx, sy, sz)]
G0 = BETA
GI = [BETA @ a for a in ALPHA]                      # γ^i = β α_i
check("β = σz ⊗ I: la matriz de Dirac es la Z del primer qubit de la codificación en dos qubits",
      np.allclose(BETA, np.kron(sz, I2)))

rng = np.random.default_rng(0)
worst, worst_hf = 0.0, 0.0
states = []
for _ in range(300):
    mass, p = rng.uniform(0.1, 2.0), rng.normal(size=3) * rng.uniform(0, 3)
    H = sum(a * pi for a, pi in zip(ALPHA, p)) + mass * BETA
    w, V = np.linalg.eigh(H)
    for k in range(4):
        v, E = V[:, k], w[k]
        worst = max(worst, abs(np.real(v.conj() @ BETA @ v) - mass / E))
        states.append((mass, p, E, v))
    # Hellmann–Feynman: dE/dm = <β>
    h = 1e-6
    w2 = np.linalg.eigvalsh(sum(a * pi for a, pi in zip(ALPHA, p)) + (mass + h) * BETA)
    worst_hf = max(worst_hf, np.max(np.abs((w2 - w) / h - np.array([mass / x for x in w]))))
check("numpy: ⟨β⟩ = mc²/E en 1200 autoestados libres (rama positiva y negativa)", worst < 1e-12, f"error máx {worst:.1e}")
check("Hellmann–Feynman: ∂E/∂(mc²) = ⟨β⟩ (diferencia finita)", worst_hf < 1e-5, f"error máx {worst_hf:.1e}")

# invariance: chiral representation (a unitary change of basis) gives the same expectation of the same operator
U = np.block([[I2, -I2], [I2, I2]]) / math.sqrt(2)
worst_c = max(abs(np.real((U @ v).conj() @ (U @ BETA @ U.conj().T) @ (U @ v)) - mass / E) for mass, p, E, v in states)
check("la lectura no depende de la representación (quiral): ⟨β⟩ = ψ̄ψ/ψ†ψ", worst_c < 1e-12, f"error máx {worst_c:.1e}")

# Qiskit: the β reading is qg_Z of the first qubit (Qiskit orders qubits right to left: Z on qubit 1)
worst_q = 0.0
for mass, p, E, v in states[:40]:
    zq = float(np.real(Statevector(np.ascontiguousarray(v)).expectation_value(Pauli("ZI"))))
    worst_q = max(worst_q, abs(zq - mass / E))
check("Qiskit: qg_Z del qubit de β en la codificación de dos qubits = mc²/E (40 estados)", worst_q < 1e-12,
      f"error máx {worst_q:.1e}")

# ---- 3. antimatter: C, P, T, CPT on the reading ---------------------------------------------------------------
def C(v):        # charge conjugation ψ -> iγ²ψ*
    return 1j * GI[1] @ v.conj()


def P(v):        # parity at the same momentum label: ψ -> γ⁰ψ (the momentum flips, the β reading is what we test)
    return G0 @ v


def T(v):        # time reversal ψ -> iγ¹γ³ψ*
    return 1j * GI[0] @ GI[2] @ v.conj()


def reading(v):
    return float(np.real(v.conj() @ BETA @ v) / np.real(v.conj() @ v))


worst_C = max(abs(reading(C(v)) + reading(v)) for *_, v in states)
worst_P = max(abs(reading(P(v)) - reading(v)) for *_, v in states)
worst_T = max(abs(reading(T(v)) - reading(v)) for *_, v in states)
worst_CPT = max(abs(reading(C(P(T(v)))) + reading(v)) for *_, v in states)
check("C invierte la lectura: m → −m (u de energía positiva pasa a la rama negativa)", worst_C < 1e-12)
check("P y T conservan la lectura", worst_P < 1e-12 and worst_T < 1e-12)
worst_d = max(abs(np.real(C(v).conj() @ C(v)) - np.real(v.conj() @ v)) for *_, v in states)
check("C conserva la densidad ψ†ψ (iγ² es unitaria)", worst_d < 1e-12, f"error máx {worst_d:.1e}")
check("CPT invierte la lectura, porque C la invierte", worst_CPT < 1e-12)
# C maps a positive-energy eigenstate of H(p) to a negative-energy eigenstate of H(-p)
mass, p, E, v = next(s for s in states if s[2] > 0)
Hm = sum(a * pi for a, pi in zip(ALPHA, -p)) + mass * BETA
cv = C(v)
check("C lleva un autoestado de energía +E de H(p) a uno de energía −E de H(−p)",
      np.allclose(Hm @ cv, -E * cv), f"E = {E:.4f}")
a = 147.6
check("en el ángulo del qubit de β, θ_qga → 500 − θ_qga (reflexión); a + 500 da la misma lectura",
      abs(math.cos(2 * math.pi * (500 - a) / 1000) + math.cos(2 * math.pi * a / 1000)) < 1e-12
      and abs(math.cos(2 * math.pi * (500 + a) / 1000) - math.cos(2 * math.pi * (500 - a) / 1000)) < 1e-12)

# ---- 4. the Pythagoras qubit: p0 = 1/γ², p1 = β² (angle 2θ_r) -------------------------------------------------
worst_py = 0.0
for beta in (0.3, 0.6, 0.8, 0.95):
    th = math.acos(math.sqrt(1 - beta ** 2))
    st = (-1j * th * qutip.sigmay()).expm() * qutip.basis(2, 0)          # R_y(2θ_r)
    p0 = abs(st.full()[0, 0]) ** 2
    worst_py = max(worst_py, abs(p0 - (1 - beta ** 2)))
check("QuTiP: R_y(2θ_r)|0⟩ da p0 = (mc²/E)² = 1/γ² y p1 = β² (el 'qubit de Pitágoras')", worst_py < 1e-12)
check("β = 0,6: qubit de β p0 = 0,9, qubit de Pitágoras p0 = 0,64 (204,8 qga)",
      round((1 + 0.8) / 2, 3) == 0.9 and round(0.8 ** 2, 2) == 0.64 and round(2 * math.acos(0.8) * QGA, 1) == 204.8)

# ---- 5. rapidity adds, qga does not ----------------------------------------------------------------------------
eta = math.atanh(0.6)
btot = math.tanh(2 * eta)
th_tot = math.acos(math.sqrt(1 - btot ** 2))
check("dos impulsos colineales de 0,6c: β = 0,8824, γ = 2,125, θ_r = 61,93° = 172,0 qga (no 204,8)",
      round(btot, 4) == 0.8824 and round(1 / math.sqrt(1 - btot ** 2), 3) == 2.125
      and round(math.degrees(th_tot), 2) == 61.93 and round(th_tot * QGA, 1) == 172.0)
check("cos θ_r = 1/cosh η: la lectura del qubit de β es 1000·sech η", abs(math.sqrt(1 - btot ** 2) - 1 / math.cosh(2 * eta)) < 1e-12)

# ---- 6. shot noise of the relativistic angle and of β ---------------------------------------------------------
sampler = StatevectorSampler(seed=np.random.default_rng(7))
N, R = 1000, 400
worst_t, worst_b = 0.0, 0.0
for beta in (0.3, 0.6, 0.9):
    th = math.acos(math.sqrt(1 - beta ** 2))
    qc = QuantumCircuit(1)
    qc.ry(th, 0)
    qc.measure_all()
    est_t, est_b = [], []
    for r in sampler.run([qc] * R, shots=N).result():
        k0 = r.data.meas.get_counts().get("0", 0)
        t_hat = math.acos(2 * k0 / N - 1)
        est_t.append(t_hat)
        est_b.append(math.sin(t_hat))
    worst_t = max(worst_t, abs(np.std(est_t) * math.sqrt(N) - 1))
    worst_b = max(worst_b, abs(np.std(est_b) * math.sqrt(N) / math.sqrt(1 - beta ** 2) - 1))
check("Qiskit: σ(θ_r) = 1/√N (159,15/√N qga) para cualquier velocidad", worst_t < 0.08, f"desvío máx {worst_t:.3f}")
check("Qiskit: σ(β) = 1/(γ√N): la velocidad se resuelve mejor cuanto más rápida", worst_b < 0.08, f"desvío máx {worst_b:.3f}")
check("con N = 1000: σ(θ_r) = 5,03 qga; a β = 0,99, σ(β) = 0,0045", round(QGA / math.sqrt(1000), 2) == 5.03
      and round(math.sqrt(1 - 0.99 ** 2) / math.sqrt(1000), 4) == 0.0045)

# ---- 7. against qang.qga ---------------------------------------------------------------------------------------
import os  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qang import qga  # noqa: E402

check("qang.qga: milliqang_from_qga(102,4 qga) ≈ +800 m-qg y qga_from_milliqang(600) = 147,6 qga",
      abs(qga.milliqang_from_qga(math.acos(0.8) * QGA) - 800) < 1e-9 and round(qga.qga_from_milliqang(600), 1) == 147.6)

print(f"\n{sum(RESULTS)} de {len(RESULTS)} afirmaciones PASAN")
sys.exit(0 if all(RESULTS) else 1)
