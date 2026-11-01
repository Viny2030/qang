"""Independent check of the additions of 2026-11-01 to chapter 6 of the book (depolarized angle, complementarity).

Covers only the new paragraphs: the depolarization example (naive and separated angle), the identity
P_Z² + V² = P², and the purity-corrected entropic bound qg_S(Z) + qg_S(Y) ≥ 1 + H((1+P)/2) on random mixed states in
QuTiP and Qiskit. qang is not used. Prints PASA/FALLA per claim and exits with 1 if any fails.

    .venv-int/Scripts/python tools/verify_cap06.py
"""

import math
import sys

import numpy as np
import qutip
from qiskit.quantum_info import DensityMatrix, Pauli, random_density_matrix

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASA' if ok else 'FALLA'}] {name}" + (f" — {detail}" if detail else ""))


def H(p):
    return 0.0 if p <= 0 or p >= 1 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


# the light of the chapter: (qg_X, qg_Y, qg_Z) = (0.3, 0, 0.6), through diag(1, d, d, d) with d = 0.7
v = np.array([0.3, 0.0, 0.6])
P0, d = np.linalg.norm(v), 0.7
v1 = d * v
check("P pasa de 0,671 a 0,470 y qg_Z de 0,60 a 0,42", round(P0, 3) == 0.671 and round(np.linalg.norm(v1), 3) == 0.470
      and round(v1[2], 2) == 0.42)
naive0, naive1 = math.degrees(math.acos(v[2])), math.degrees(math.acos(v1[2]))
check("arccos(qg_Z) pasa de 53,1° a 65,2°: 12° de ensanchamiento falso",
      round(naive0, 1) == 53.1 and round(naive1, 1) == 65.2 and round(naive1 - naive0) == 12)
sep0, sep1 = math.degrees(math.acos(v[2] / P0)), math.degrees(math.acos(v1[2] / np.linalg.norm(v1)))
check("arccos(qg_Z/P) = 26,57° antes y después; 73,8 qga",
      round(sep0, 2) == round(sep1, 2) == 26.57 and round(sep0 * 1000 / 360, 1) == 73.8)
check("con los valores redondeados 0,42/0,470 daría 26,67°: el 26,56° del .docx viene de no redondear",
      round(math.degrees(math.acos(0.42 / 0.470)), 2) == 26.67)

# complementarity and the entropic bound on random mixed states, two libraries
worst_id, worst_gap = 0.0, math.inf
for k in range(200):
    if k % 2:
        rho = qutip.rand_dm(2, seed=k)
        x, y, z = (qutip.expect(op, rho) for op in (qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()))
    else:
        rho = random_density_matrix(2, seed=k)
        x, y, z = (float(np.real(DensityMatrix(rho).expectation_value(Pauli(p)))) for p in "XYZ")
    P = math.sqrt(x * x + y * y + z * z)
    worst_id = max(worst_id, abs(z * z + (x * x + y * y) - P * P))
    worst_gap = min(worst_gap, H((1 + z) / 2) + H((1 + y) / 2) - 1 - H((1 + P) / 2))
check("QuTiP y Qiskit: P_Z² + V² = P² en 200 estados mixtos", worst_id < 1e-12, f"error máx {worst_id:.1e}")
check("QuTiP y Qiskit: qg_S(Z) + qg_S(Y) ≥ 1 + H((1+P)/2) en 200 estados mixtos", worst_gap >= -1e-12,
      f"holgura mínima {worst_gap:.2e}")

print(f"\n{sum(RESULTS)} de {len(RESULTS)} afirmaciones PASAN")
sys.exit(0 if all(RESULTS) else 1)
