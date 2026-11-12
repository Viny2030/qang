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

# ---------------------------------------------------------------- magnitudes derivadas en el muestreo (2026-11-12)
print("\nMagnitudes derivadas en el muestreo")
zz = sp.Symbol("z")
pz = (1 + zz) / 2
F_qg = sum(sp.diff(q, zz) ** 2 / q for q in (pz, 1 - pz))
check("sympy: F_qg = 1/(1 − qg_Z²) y F_θ = 1 con qg_Z = cos θ",
      sp.simplify(F_qg - 1 / (1 - zz ** 2)) == 0
      and abs(float((F_qg * sp.diff(sp.cos(sp.Symbol("t")), sp.Symbol("t")) ** 2).subs(zz, sp.cos(sp.Symbol("t"))).subs(sp.Symbol("t"), 0.7)) - 1) < 1e-12)
zb = 50 / 52
check("50 de 50: media de Bayes–Haar 0,962, qg_SNR = 3,50, qg_κ = 19,1",
      round(zb, 3) == 0.962 and round(zb / math.sqrt(1 - zb ** 2), 2) == 3.50
      and round(1 / (math.log(2) * (1 - zb ** 2)), 1) == 19.1)

th = lambda p: 2 * math.acos(math.sqrt(p))
qgl = abs(th(0.75) - th(0.70)) / 2
se = math.sqrt(1 / 1000 + 1 / 1000) / 2
check("750 y 700 de 1000: qg_L = 0,056 rad = 8,9 qga, error 0,0224, 2,5σ",
      round(qgl, 3) == 0.056 and round(qgl * 1000 / (2 * math.pi), 1) == 8.9 and round(se, 4) == 0.0224 and round(qgl / se, 1) == 2.5)
check("qg_L es la mitad de la h de Cohen, |2 arcsin√p1 − 2 arcsin√p2|/2",
      abs(qgl - abs(2 * math.asin(math.sqrt(0.75)) - 2 * math.asin(math.sqrt(0.70))) / 2) < 1e-12)
rng = np.random.default_rng(4)
R = 200000
a = rng.binomial(1000, 0.75, R) / 1000
b = rng.binomial(1000, 0.70, R) / 1000
sim = ((2 * np.arccos(np.sqrt(a)) - 2 * np.arccos(np.sqrt(b))) / 2).std()
check("muestreo (200 000 pares): la desviación de qg_L es ½√(1/N1 + 1/N2) = 0,0224", abs(sim - se) / se < 0.01, f"{sim:.5f}")
worst = 0.0
for p1, p2 in ((0.5, 0.45), (0.9, 0.86), (0.2, 0.25)):
    a = rng.binomial(1000, p1, R) / 1000
    b = rng.binomial(1000, p2, R) / 1000
    worst = max(worst, abs(((2 * np.arccos(np.sqrt(a)) - 2 * np.arccos(np.sqrt(b))) / 2).std() - se) / se)
check("la misma desviación en el ecuador y cerca de los polos (dentro de 2 %)", worst < 0.02, f"{worst:.3f}")

# E[r_hat^2] = r^2 + (3 - r^2)/N, exactly, from E[q_hat^2] = q^2 + (1 - q^2)/N on each axis
qx, qy, qz, Nn = sp.symbols("qx qy qz N", positive=True)
lhs = sum(q ** 2 + (1 - q ** 2) / Nn for q in (qx, qy, qz))
r2s = qx ** 2 + qy ** 2 + qz ** 2
check("sympy: E[r̂²] = r² + (3 − r²)/N", sp.simplify(lhs - (r2s + (3 - r2s) / Nn)) == 0)
# exact binomial expectation of q_hat^2 for one axis
from scipy.stats import binom as _binom
ks = np.arange(101)
q = 0.72
e2 = float(np.sum(_binom.pmf(ks, 100, (1 + q) / 2) * (2 * ks / 100 - 1) ** 2))
check("binomial exacta: E[q̂²] = q² + (1 − q²)/N (q = 0,72, N = 100)", abs(e2 - (q * q + (1 - q * q) / 100)) < 1e-12)
check("estado puro, 100 disparos por eje: E[r̂²] = 1,02", abs(1 + (3 - 1) / 100 - 1.02) < 1e-12)
rvec = np.array([0.54, 0.0, 0.72])
Nsh = 100
# exact expectations over the three binomials (101^3 terms), no sampling
qh = 2 * ks / Nsh - 1
pm = [_binom.pmf(ks, Nsh, (1 + x) / 2) for x in rvec]
W = pm[0][:, None, None] * pm[1][None, :, None] * pm[2][None, None, :]
n2, u2 = qh ** 2, (Nsh * qh ** 2 - 1) / (Nsh - 1)
grid_naive = n2[:, None, None] + n2[None, :, None] + n2[None, None, :]
grid_unb = u2[:, None, None] + u2[None, :, None] + u2[None, None, :]


class _Mean:
    def __init__(self, v):
        self.v = float(v)

    def mean(self):
        return self.v


r2_naive, r2_unb = _Mean((W * grid_naive).sum()), _Mean((W * grid_unb).sum())
tau_naive = float((W * np.sqrt(np.clip(1 - grid_naive, 0, None))).sum())
tau_unb = float((W * np.sqrt(np.clip(1 - grid_unb, 0, None))).sum())
check("sumas binomiales exactas, r = 0,9 (qg_τ = 0,436): E[r̂²] directo 0,832, corregido 0,810",
      round(float(rvec @ rvec), 2) == 0.81 and round(math.sqrt(1 - 0.81), 3) == 0.436
      and round(r2_naive.mean(), 3) == 0.832 and round(r2_unb.mean(), 3) == 0.810,
      f"{r2_naive.mean():.4f}, {r2_unb.mean():.4f}")
check("qg_τ estimado: 0,376 con el radio directo y 0,404 con el corregido, los dos por debajo de 0,436",
      round(tau_naive, 3) == 0.376 and round(tau_unb, 3) == 0.404, f"{tau_naive:.4f}, {tau_unb:.4f}")
check("el corregido reduce el sesgo de qg_τ a cerca de la mitad", 0.4 < (0.4359 - tau_unb) / (0.4359 - tau_naive) < 0.65,
      f"{(0.4359 - tau_unb) / (0.4359 - tau_naive):.2f}")
check("N_ejecución = N/(1 − P_descarte): 2500 → 3125 con 20 % de descarte", round(2500 / 0.8) == 3125)

from qang import proportions as _pr, statistics as _st  # noqa: E402  final comparison only
est, err = _st.radius2_estimate([(77, 100), (50, 100), (86, 100)])
check("qang coincide: cohens_h/2 = qg_L y radius2_estimate usa los cuadrados insesgados",
      abs(_pr.cohens_h(0.75, 0.70) / 2 - qgl) < 1e-12
      and abs(est - sum(_st.qg2_unbiased(k, 100) for k in (77, 50, 86))) < 1e-12)

print(f"\n{sum(RESULTS)} de {len(RESULTS)} afirmaciones PASAN")
sys.exit(0 if all(RESULTS) else 1)
