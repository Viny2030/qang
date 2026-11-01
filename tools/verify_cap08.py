"""Independent check of the additions of 2026-11-01 to chapter 8 of the book (Fisher information in qga, Schottky).

Covers only the new sentences: the per-shot Fisher information on the qga scale (sympy, and a sampling check of the
angle's spread in Qiskit) and the Schottky optimum written as a reading and an angle (scipy). qang is not used.
Prints PASA/FALLA per claim and exits with 1 if any fails.

    .venv-int/Scripts/python tools/verify_cap08.py
"""

import math
import sys

import numpy as np
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from scipy.optimize import brentq

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASA' if ok else 'FALLA'}] {name}" + (f" — {detail}" if detail else ""))


a = sp.symbols("a", positive=True)                      # angle in qga
p0 = sp.cos(sp.pi * a / 1000) ** 2
F = sp.diff(p0, a) ** 2 / (p0 * (1 - p0))
vals = [float(F.subs(a, v)) for v in (40, 125, 250, 400)]
check("sympy: la información de un disparo en qga es 4π²·10⁻⁶ qga⁻² en todo el círculo",
      max(abs(v - 4 * math.pi ** 2 * 1e-6) for v in vals) < 1e-15, f"{vals[0]:.6e}")

# sampling in Qiskit: spread of the estimated angle (in qga) for N = 1000 shots at three angles
sampler = StatevectorSampler(seed=np.random.default_rng(5))     # a generator: an int seed repeats every pub
N, R = 1000, 400
worst = 0.0
for deg in (40, 90, 140):
    qc = QuantumCircuit(1)
    qc.ry(math.radians(deg), 0)
    qc.measure_all()
    res = sampler.run([qc] * R, shots=N).result()
    est = []
    for r in res:
        c = r.data.meas.get_counts()
        p = c.get("0", 0) / N
        est.append(2 * math.acos(math.sqrt(p)) * 1000 / (2 * math.pi))
    worst = max(worst, abs(np.std(est) / (1000 / (2 * math.pi) / math.sqrt(N)) - 1))
check("Qiskit: la desviación del ángulo estimado es 159,15/√N qga a 40°, 90° y 140° (N = 1000)", worst < 0.1,
      f"desvío relativo máx {worst:.3f}")

q = brentq(lambda x: x * math.atanh(x) - 1, 0.5, 0.99)
th = math.degrees(math.acos(q))
check("Schottky: qg* = 0,8336, m* = +833,6 m-qg, θ* = 33,53° = 93,2 qga",
      round(q, 4) == 0.8336 and round(1000 * q, 1) == 833.6 and round(th, 2) == 33.53 and round(th * 1000 / 360, 1) == 93.2,
      f"θ* = {th:.4f}° = {th * 1000 / 360:.3f} qga")

print(f"\n{sum(RESULTS)} de {len(RESULTS)} afirmaciones PASAN")
sys.exit(0 if all(RESULTS) else 1)
