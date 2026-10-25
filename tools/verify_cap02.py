"""Independent check of chapter 2 of the book (native gates, closed algebra, universal operator).

    .venv-int/Scripts/python tools/verify_cap02.py

Recomputes the chapter's central claims with Cirq, PennyLane and QuTiP, which are not the library the chapter
used (§166 checked against Qiskit). Prints PASA/FALLA per claim and exits 1 if any claim fails.
"""

import math
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import cirq  # noqa: E402
import pennylane as qml  # noqa: E402
import qutip  # noqa: E402

from qang import algebra, probability, pulse_area  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASA ' if ok else 'FALLA'}  {name}" + (f"  [{detail}]" if detail else ""))


def phase_equal(a, b, tol=1e-10):
    """Distance between two unitaries up to a global phase."""
    k = np.argmax(np.abs(b))
    ph = a.flat[k] / b.flat[k]
    if abs(abs(ph) - 1) > 1e-8:
        return math.inf
    return float(np.max(np.abs(a - ph * b)))


# ---------------------------------------------------------------- U3 in qg against three SDKs

def u3_cirq(theta, phi, lam):
    # cirq's QasmUGate is OpenQASM U(theta, phi, lam), angles in half-turns
    return cirq.unitary(cirq.circuits.qasm_output.QasmUGate(theta / math.pi, phi / math.pi, lam / math.pi))


def u3_pennylane(theta, phi, lam):
    return qml.matrix(qml.U3(theta, phi, lam, wires=0))


def u3_qutip(theta, phi, lam):
    # Rz(phi) Ry(theta) Rz(lam), built from QuTiP operators; equals U3 up to a global phase
    rz = lambda a: (-1j * a / 2 * qutip.sigmaz()).expm()  # noqa: E731
    ry = (-1j * theta / 2 * qutip.sigmay()).expm()
    return (rz(phi) * ry * rz(lam)).full()


rng = np.random.default_rng(2)
# Cirq's QasmUGate is Rz(φ)Ry(θ)Rz(λ): it differs from Qiskit's U by the global phase e^{-i(φ+λ)/2}
SDKS = {"Cirq": (u3_cirq, True),"PennyLane": (u3_pennylane, False), "QuTiP": (u3_qutip, True)}
for sdk, (f, up_to_phase) in SDKS.items():
    worst = 0.0
    for _ in range(2000):
        q = rng.uniform(-1, 1)
        phi, lam = rng.uniform(-math.pi, math.pi, 2)
        ours = algebra.u3_qang(q, phi, lam)
        ref = f(math.acos(q), phi, lam)
        worst = max(worst, phase_equal(ours, ref) if up_to_phase else float(np.max(np.abs(ours - ref))))
    exact = "salvo fase global" if up_to_phase else "exacta, fase incluida"
    check(f"u3_qang(q, φ, λ) = U3(arccos q, φ, λ) de {sdk} en 2000 compuertas ({exact})", worst < 1e-12,
          f"máx. {worst:.1e}; el libro cita 3,5e-14 contra Qiskit")

# I, X, Z, H as points (q, φ, λ) of the table, against each SDK's own gate, up to a global phase
STD = {"I": (1, 0, 0, cirq.I, qml.Identity, qutip.qeye(2)),
       "X": (-1, 0, math.pi, cirq.X, qml.PauliX, qutip.sigmax()),
       "Z": (1, 0, math.pi, cirq.Z, qml.PauliZ, qutip.sigmaz()),
       "H": (0, 0, math.pi, cirq.H, qml.Hadamard, (qutip.sigmax() + qutip.sigmaz()) / math.sqrt(2))}
for g, (q, phi, lam, cg, pg, qg) in STD.items():
    ours = algebra.u3_qang(q, phi, lam)
    d = max(phase_equal(ours, cirq.unitary(cg)), phase_equal(ours, qml.matrix(pg(wires=0))),
            phase_equal(ours, qg.full()))
    m = probability.milliqang_from_p0((1 + q) / 2)
    check(f"{g} = U3_qg({q}, {phi}, {lam:.4g}), lectura {m:+.0f} m-qg (Cirq, PennyLane, QuTiP)", d < 1e-12,
          f"máx. {d:.1e}")

# ---------------------------------------------------------------- transition matrix

worst, ds = 0.0, 0.0
for _ in range(500):
    q = rng.uniform(-1, 1)
    phi, lam = rng.uniform(-math.pi, math.pi, 2)
    t_ref = np.abs(u3_pennylane(math.acos(q), phi, lam)) ** 2
    worst = max(worst, float(np.max(np.abs(algebra.transition_matrix(q) - t_ref))))
    ds = max(ds, float(np.max(np.abs(t_ref.sum(0) - 1))), float(np.max(np.abs(t_ref.sum(1) - 1))))
check("matriz de transición |U_ij|² = T(q) (PennyLane, fases al azar)", worst < 1e-12, f"máx. {worst:.1e}")
check("T(q) doblemente estocástica", ds < 1e-12, f"máx. {ds:.1e}")
t03 = algebra.transition_matrix(0.3)
check("transition_matrix(0.3) = [[0.65, 0.35], [0.35, 0.65]] y p0 = 0.65 es +300 m-qg",
      np.allclose(t03, [[0.65, 0.35], [0.35, 0.65]]) and abs(probability.milliqang_from_p0(0.65) - 300) < 1e-9)

# Born rule on |0> through Cirq's simulator: p0 after the gate is (1+q)/2 whatever the phases
worst = 0.0
for _ in range(200):
    q = rng.uniform(-1, 1)
    phi, lam = rng.uniform(-math.pi, math.pi, 2)
    qb = cirq.LineQubit(0)
    psi = cirq.Simulator().simulate(cirq.Circuit(cirq.MatrixGate(algebra.u3_qang(q, phi, lam)).on(qb))).final_state_vector
    worst = max(worst, abs(abs(psi[0]) ** 2 - (1 + q) / 2))
check("p0 tras U3_qg|0⟩ = (1+q)/2 en el simulador de Cirq (ecuación de probabilidad)", worst < 1e-6,
      f"máx. {worst:.1e}, precisión float32 de Cirq")


# capacity of the binary symmetric channel = 1 - qg_S
def h2(p):
    return 0.0 if p in (0.0, 1.0) else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


worst = 0.0
for q in np.linspace(-1, 1, 41):
    t = algebra.transition_matrix(q)
    # mutual information of the channel with uniform input (optimal for a BSC), from QuTiP-free numpy
    pin = np.array([0.5, 0.5])
    joint = pin[:, None] * t
    pout = joint.sum(0)
    mi = sum(joint[i, j] * math.log2(joint[i, j] / (pin[i] * pout[j])) for i in range(2) for j in range(2)
             if joint[i, j] > 0)
    worst = max(worst, abs(mi - (1 - h2((1 + q) / 2))))
check("capacidad del canal = 1 − qg_S (información mutua con entrada uniforme)", worst < 1e-12, f"máx. {worst:.1e}")

# preparation: p0 = 0.75 and +500 m-qg ask for the same gate R_y(60°)
th = probability.theta_from_milliqang(probability.milliqang_from_p0(0.75))
dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def ry_probs(t):
    qml.RY(t, wires=0)
    return qml.probs(wires=0)


check("p0 = 0,75 ↔ +500 m-qg ↔ R_y(60°), y R_y(60°)|0⟩ da p0 = 0,75 en PennyLane",
      abs(math.degrees(th) - 60) < 1e-9 and abs(float(ry_probs(th)[0]) - 0.75) < 1e-12,
      f"θ = {math.degrees(th):.6f}°")

# ---------------------------------------------------------------- diagonal Gell-Mann matrices

# d = 3 against PennyLane's GellMann (indices 3 and 8 are the diagonal ones)
gm3 = np.array([np.real(np.diag(qml.matrix(qml.GellMann(wires=0, index=i)))) for i in (3, 8)])
check("cartan_generators(3) = diagonales de λ3 y λ8 de PennyLane (qml.GellMann)",
      np.allclose(algebra.cartan_generators(3), gm3), str(np.round(algebra.cartan_generators(3), 4).tolist()))
# d = 2: the single generator is sigma_z of QuTiP
check("d = 2: λ1 = σ_z (QuTiP) y cartan_values(p) = qg_Z",
      np.allclose(algebra.cartan_generators(2)[0], np.real(np.diag(qutip.sigmaz().full())))
      and abs(algebra.cartan_values([0.8, 0.2])[0] - 0.6) < 1e-12)
ok, worst = True, 0.0
for d in range(2, 9):
    g = algebra.cartan_generators(d)
    ok &= np.allclose(g @ g.T, 2 * np.eye(d - 1)) and np.allclose(g.sum(1), 0)
    for _ in range(50):
        p = rng.dirichlet(np.ones(d))
        # expectation values from QuTiP: <λ_k> = Tr(ρ λ_k) with ρ diagonal
        rho = qutip.Qobj(np.diag(p))
        vals = [qutip.expect(qutip.Qobj(np.diag(row)), rho) for row in g]
        worst = max(worst, float(np.max(np.abs(algebra.populations_from_cartan(vals) - p))))
check("d = 2..8: Tr(λj λk) = 2δjk, trazas nulas y p = 1/d + ½ Σ ⟨λk⟩ diag(λk) (⟨λk⟩ de QuTiP)",
      ok and worst < 1e-12, f"máx. {worst:.1e}")

# ---------------------------------------------------------------- minimal pulse area

# R_X(θ) = −R_X(θ − 2π) in Cirq and PennyLane
worst = 0.0
for t in rng.uniform(-4 * math.pi, 4 * math.pi, 200):
    worst = max(worst, float(np.max(np.abs(cirq.unitary(cirq.rx(t)) + cirq.unitary(cirq.rx(t - 2 * math.pi))))),
                float(np.max(np.abs(qml.matrix(qml.RX(t, 0)) + qml.matrix(qml.RX(t - 2 * math.pi, 0))))))
check("R_X(θ) = −R_X(θ − 2π) (Cirq y PennyLane)", worst < 1e-6, f"máx. {worst:.1e}")

# R_ZZ(θ + π) = R_ZZ(θ)(−i Z⊗Z) in PennyLane (IsingZZ) and Cirq (ZZPowGate with global_shift −½)
ZZ = np.kron(np.diag([1, -1]), np.diag([1, -1]))
rzz_c = lambda t: cirq.unitary(cirq.ZZPowGate(exponent=t / math.pi, global_shift=-0.5))  # noqa: E731
rzz_p = lambda t: qml.matrix(qml.IsingZZ(t, wires=[0, 1]))  # noqa: E731
worst = 0.0
for t in rng.uniform(-2 * math.pi, 2 * math.pi, 200):
    for f in (rzz_c, rzz_p):
        worst = max(worst, float(np.max(np.abs(f(t + math.pi) - f(t) @ (-1j * ZZ)))))
check("R_ZZ(θ + π) = R_ZZ(θ)(−i Z⊗Z) (Cirq y PennyLane)", worst < 1e-6, f"máx. {worst:.1e}")
t, k = pulse_area.minimal_area_angle(3.5, math.pi)
d = float(np.max(np.abs(rzz_p(3.5) - rzz_p(t) @ np.linalg.matrix_power(-1j * ZZ, k))))
check("minimal_area_angle(3.5, π) = (0.358, 1) y R_ZZ(3.5) = R_ZZ(0.358)(−i Z⊗Z) en PennyLane",
      round(t, 3) == 0.358 and k == 1 and d < 1e-12, f"θ' = {t:.4f}, k = {k}")


# damping qg_Z = cos θ e^{−θ²σ²/2}: Gauss–Hermite average of cos(θ(1+ε)) read from QuTiP's RX state
xs, ws = np.polynomial.hermite_e.hermegauss(60)
ws = ws / ws.sum()


def qgz_rx_qutip(theta, sigma):
    out = 0.0
    for x, w in zip(xs, ws):
        a = theta * (1 + sigma * x)
        psi = (-1j * a / 2 * qutip.sigmax()).expm() * qutip.basis(2, 0)
        out += w * qutip.expect(qutip.sigmaz(), psi)
    return out


worst = 0.0
for th_ in np.linspace(0, 2 * math.pi, 25):
    worst = max(worst, abs(qgz_rx_qutip(th_, 0.05) - pulse_area.rx_qg_z_random_amplitude(th_, 0.05)))
check("qg_Z = cos θ · e^(−θ²σ²/2) frente al promedio sobre ε de RX(θ(1+ε)) en QuTiP (σ = 0,05)", worst < 1e-10,
      f"máx. {worst:.1e}")

# Monte Carlo with Cirq's simulator, the §160 P1 check, at the X gate
eps = rng.normal(0, 0.05, 4000)
qb = cirq.LineQubit(0)
sim = cirq.Simulator()
vals = [abs(sim.simulate(cirq.Circuit(cirq.rx(math.pi * (1 + e)).on(qb))).final_state_vector[0]) ** 2 * 2 - 1
        for e in eps]
mc = float(np.mean(vals))
err = float(np.std(vals) / math.sqrt(len(vals)))
check("X con σ = 0,05: Monte Carlo en Cirq da −0,988 (rx_qg_z_random_amplitude)",
      abs(mc - pulse_area.rx_qg_z_random_amplitude(math.pi, 0.05)) < 4 * err + 1e-6,
      f"MC {mc:.5f} ± {err:.5f}, fórmula {pulse_area.rx_qg_z_random_amplitude(math.pi, 0.05):.5f}")

# the m-qg numbers of the text
e_x = 1000 * (qgz_rx_qutip(math.pi, 0.05) - math.cos(math.pi))
check("corrimiento de una X con σ = 0,05: 12,3 m-qg (QuTiP)", round(e_x, 1) == 12.3, f"{e_x:.3f}")
e_324 = 1000 * (qgz_rx_qutip(math.radians(324), 0.05) - math.cos(math.radians(324)))
e_m36 = 1000 * (qgz_rx_qutip(math.radians(-36), 0.05) - math.cos(math.radians(-36)))
check("R_X(324°): −31,7 m-qg; su representante R_X(−36°): −0,4 m-qg (QuTiP)",
      round(e_324, 1) == -31.7 and round(e_m36, 1) == -0.4, f"{e_324:.3f} / {e_m36:.3f}")
check("R_X(324°) y R_X(−36°) son la misma compuerta salvo fase global (Cirq)",
      phase_equal(cirq.unitary(cirq.rx(math.radians(324))), cirq.unitary(cirq.rx(math.radians(-36)))) < 1e-6)
sig = probability.shot_noise_milliqang(math.pi / 2, 10_000)
check("resolución de 10⁴ disparos en el ecuador: 10 m-qg", abs(sig - 10) < 1e-9, f"{sig:.2f}")
s = math.sqrt(1 - pulse_area.rx_qg_z_random_amplitude(math.pi, 0.05) ** 2)
print(f"       (nota: en la lectura real de la X, −987,7 m-qg, la resolución de 10⁴ disparos es "
      f"{1000 * s / 100:.1f} m-qg, no 10)")

# first-order systematic error 1000 ε θ sin θ
th_, e_ = 2.0, 1e-4
num = 1000 * (pulse_area.rx_qg_z_systematic_amplitude(th_, e_) - math.cos(th_))
check("error fijo ε: −1000 ε θ sin θ m-qg a primer orden", abs(num + 1000 * e_ * th_ * math.sin(th_)) < 1e-5,
      f"{num:.6f} frente a {-1000 * e_ * th_ * math.sin(th_):.6f}; el texto da el módulo")

# ---------------------------------------------------------------- W state of the interop section

w = np.zeros(8)
w[[1, 2, 4]] = 1 / math.sqrt(3)
rho = qutip.ket2dm(qutip.Qobj(w, dims=[[2, 2, 2], [1, 1, 1]]))
qz = [qutip.expect(qutip.tensor(*[qutip.sigmaz() if j == i else qutip.qeye(2) for j in range(3)]), rho)
      for i in range(3)]
p = np.abs(w) ** 2
ent = -sum(x * math.log2(x) for x in p if x > 0) / 3
check("estado W: qg_Z = 1/3 por qubit (333 m-qg) y entropía del registro log₂3/3 = 0,528 (QuTiP)",
      np.allclose(qz, 1 / 3) and round(ent, 3) == 0.528, f"qg_Z = {qz[0]:.4f}, H/n = {ent:.4f}")
sm = probability.shot_noise_milliqang(math.acos(1 / 3), 20_000)
check("σ_m = 1000 sin θ/√N = 6,7 m-qg con 20 000 disparos; 316 m-qg queda a 2,6σ",
      round(sm, 1) == 6.7 and round((1000 / 3 - 316) / sm, 1) == 2.6, f"σ_m = {sm:.3f}, z = {(1000/3-316)/sm:.2f}")

print(f"\n{sum(RESULTS)}/{len(RESULTS)} afirmaciones PASAN")
sys.exit(0 if all(RESULTS) else 1)
