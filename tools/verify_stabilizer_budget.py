"""Independent check of qang.sectors.stabilizer_shot_budget (2026 context, arXiv:2607.01072:
"Optimizing Symmetry Informed Probabilistic Error Cancellation").

    .venv-int/Scripts/python tools/verify_stabilizer_budget.py

Recomputes the DP against brute force over random instances (standard library + NumPy only,
no SDK involved: this is a discrete-optimization helper, not a physics claim). Prints PASA/FALLA
per claim and exits 1 if any claim fails. Status in qang.measures: sin_verificar.
"""

import os
import sys
from itertools import combinations

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from qang.sectors import stabilizer_shot_budget  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASA ' if ok else 'FALLA'}  {name}" + (f"  [{detail}]" if detail else ""))


def brute_force(w, c, budget):
    n = len(w)
    best = 0.0
    for k in range(n + 1):
        for combo in combinations(range(n), k):
            if sum(c[i] for i in combo) <= budget:
                best = max(best, sum(w[i] for i in combo))
    return best


def main():
    rng = np.random.default_rng(2026)

    ok = True
    worst = 0.0
    for _ in range(300):
        n = int(rng.integers(1, 10))
        w = rng.uniform(0, 5, n)
        c = rng.integers(0, 6, n)
        budget = int(rng.integers(0, 15))
        _, got = stabilizer_shot_budget(w, c, budget)
        want = brute_force(w, c, budget)
        worst = max(worst, abs(got - want))
        if abs(got - want) > 1e-9:
            ok = False
    check("[1] DP exacta == fuerza bruta (300 instancias aleatorias)", ok, f"peor diferencia {worst:.2e}")

    chosen, total = stabilizer_shot_budget([5.0, 3.0], [1, 1], 0)
    check("[2] presupuesto 0 => no elige ningun estabilizador", chosen == [] and total == 0.0)

    chosen, total = stabilizer_shot_budget([5.0, 3.0, 2.0], [0, 0, 1], 0)
    check("[3] comprobaciones gratis (costo 0) siempre entran", sorted(chosen) == [0, 1], str(chosen))

    exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
