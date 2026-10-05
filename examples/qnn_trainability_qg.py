"""
Trainability of weight-conserving QNNs under T1, with and without qang (§112)

Gradients of random weight-conserving circuits shrink as the circuit grows
(a barren-plateau question, here inside a Hamming-weight sector), and
amplitude damping adds its own suppression: without the filter the readout
of a decayed register drifts towards |0...0>. Under equal T1 the filtered
expectation equals the noiseless one, so the filtered gradient is the
noiseless gradient; the price is the discarded shots. This study measures
both sides: the size of the gradient, and the shots needed to resolve it.

Circuits: qang.qml.WeightQNN(n, k, layers=L) with random angles and a random
real input state in the weight-k sector; cost = qg_Z of qubit 0; gradient
with respect to the first angle by central differences on exact
probabilities. Grid: n = 4, 6, 8; k = 1 and n/2; L = n and 2n layers (3
sublayers each); equal T1 gamma = 0.02 per qubit per sublayer (kept
fraction K = (1 - gamma)^(3 k L)); 200 random draws per cell; seed 112.

Readouts: noiseless; under T1 with qang (filter); under T1 without qang.
Shot cost: to resolve the sign of the difference D = f(theta + 0.3) -
f(theta - 0.3) at two standard errors, S = 4 [(1 - f+^2) + (1 - f-^2)] /
(D^2 r), with r = K for the filter (kept shots) and r = 1 without it; the
median over draws is reported.

Pre-registered predictions (committed before the run; code checked on one
cell with 5 draws):
  T1  with qang the gradient equals the noiseless gradient in every draw
      (to 1e-9).
  T2  weight 1: without qang Var(gradient) / Var(noiseless gradient) = K^2
      (to 1e-6 relative), since the raw excitation probabilities are the
      noiseless ones times K.
  T3  weight n/2: without qang the variance ratio is below K in every cell
      (the raw gradient is suppressed at least as much as the kept fraction).
  T4  the noiseless gradient variance at weight n/2 falls faster from n = 4
      to n = 8 than at weight 1 (same L = n).
  T5  the median shot cost to resolve the gradient is lower with qang than
      without, in every cell.

OMP_NUM_THREADS=1 python examples/qnn_trainability_qg.py

Findings:

FINDINGS_PLACEHOLDER
"""

import json
import sys

import numpy as np

from qang.qml import WeightQNN, kept_fraction

GAMMA = 0.02
DRAWS = 200
SHIFT = 0.3
H = 1e-5


def cell(n, k, L, rng, draws=DRAWS):
    m = WeightQNN(n, k, layers=L)
    idx = m.idx[k]
    K = kept_fraction(GAMMA, k, m.depth)
    g = {"noiseless": [], "qang": [], "raw": []}
    cost = {"qang": [], "raw": []}
    for _ in range(draws):
        th = rng.uniform(-np.pi, np.pi, m.n_theta)
        psi = np.zeros((1, m.dim))
        psi[0, idx] = rng.normal(size=len(idx))
        psi /= np.linalg.norm(psi)

        def f(t, gamma, qang):
            th2 = th.copy()
            th2[0] = t
            return float(m.qg_z(m.probs(th2, psi, gamma), qang)[0, 0])

        t0 = th[0]
        g["noiseless"].append((f(t0 + H, None, False) - f(t0 - H, None, False)) / (2 * H))
        g["qang"].append((f(t0 + H, GAMMA, True) - f(t0 - H, GAMMA, True)) / (2 * H))
        g["raw"].append((f(t0 + H, GAMMA, False) - f(t0 - H, GAMMA, False)) / (2 * H))
        for label, q, r in (("qang", True, K), ("raw", False, 1.0)):
            fp, fm = f(t0 + SHIFT, GAMMA, q), f(t0 - SHIFT, GAMMA, q)
            D = fp - fm
            cost[label].append(4 * ((1 - fp**2) + (1 - fm**2)) / (max(D * D, 1e-300) * r))
    g = {a: np.array(b) for a, b in g.items()}
    var0 = float(np.var(g["noiseless"]))
    return {"n": n, "k": k, "L": L, "depth": m.depth, "K": K,
            "var noiseless": var0, "var qang": float(np.var(g["qang"])), "var raw": float(np.var(g["raw"])),
            "max |qang - noiseless|": float(np.max(np.abs(g["qang"] - g["noiseless"]))),
            "ratio raw": float(np.var(g["raw"]) / var0) if var0 > 0 else float("nan"),
            "median shots qang": float(np.median(cost["qang"])), "median shots raw": float(np.median(cost["raw"]))}


def run(draws=DRAWS, sizes=(4, 6, 8)):
    rng = np.random.default_rng(112)
    rows = []
    for n in sizes:
        for k in (1, n // 2):
            for L in (n, 2 * n):
                rows.append(cell(n, k, L, rng, draws))
                print(json.dumps(rows[-1]), flush=True)
    return rows


def verdict(rows):
    w1 = [r for r in rows if r["k"] == 1]
    wh = [r for r in rows if r["k"] == r["n"] // 2 and r["k"] > 1]
    v = lambda n, k: next(r["var noiseless"] for r in rows if r["n"] == n and r["k"] == k and r["L"] == n)  # noqa: E731
    ns = sorted({r["n"] for r in rows})
    return {
        "T1": all(r["max |qang - noiseless|"] < 1e-9 for r in rows),
        "T2": all(abs(r["ratio raw"] / r["K"] ** 2 - 1) < 1e-6 for r in w1),
        "T3": all(r["ratio raw"] < r["K"] for r in wh),
        "T4": v(ns[-1], ns[-1] // 2) / v(ns[0], ns[0] // 2) < v(ns[-1], 1) / v(ns[0], 1),
        "T5": all(r["median shots qang"] < r["median shots raw"] for r in rows),
    }


def main():
    rows = run()
    v = verdict(rows)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return rows, v


if __name__ == "__main__":
    main()
