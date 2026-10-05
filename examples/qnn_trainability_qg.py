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

200 random draws per cell, equal T1 gamma = 0.02 per qubit per sublayer:

  n  k  L    K       Var(grad)      Var ratio without   K^2        median shots to resolve
                     noiseless      qang / noiseless               with qang / without
  4  1  4   0.785    3.4e-01        0.616               0.616      132 / 156
  4  1  8   0.616    3.1e-01        0.379               0.379      150 / 191
  4  2  4   0.616    3.7e-01        0.472               0.379      187 / 304
  4  2  8   0.379    3.2e-01        0.195               0.144      462 / 906
  6  1  6   0.695    9.4e-02        0.483               0.483      413 / 465
  6  1  12  0.483    1.0e-01        0.234               0.234      638 / 731
  6  3  6   0.336    7.1e-02        0.261               0.113      4215 / 5570
  6  3  12  0.113    6.5e-02        0.062               0.013      9748 / 10405
  8  1  8   0.616    6.6e-02        0.379               0.379      919 / 932
  8  1  16  0.379    3.5e-02        0.144               0.144      2394 / 2512
  8  4  8   0.144    1.8e-02        0.176               0.021      35163 / 26649
  8  4  16  0.021    1.3e-02        0.028               0.0004     258192 / 118611

  * T1, T2, T4 pass; T3 and T5 fail.
  * T1: with qang the gradient is the noiseless gradient in every draw (to
    7.5e-11): the filter removes the T1 suppression of the gradient exactly.
  * T2: at weight 1 the raw gradient is exactly K times the noiseless one
    (variance ratio = K^2).
  * T4: from 4 to 8 qubits the noiseless gradient variance falls by 5.2x at
    weight 1 and by 20x at weight n/2: the larger sector flattens faster.
  * T3 FAILS: at weight n/2 the raw gradient is suppressed less than K
    (variance ratio 0.028 against K = 0.021 at n = 8, L = 16, and above K^2
    everywhere): shots that decayed to lower weights keep part of the
    gradient signal.
  * T5 FAILS at the largest cells: the filter needs fewer shots to resolve the
    gradient in 10 of 12 cells, but at n = 8, k = 4 (K = 0.14 and 0.021) it
    needs 1.3-2.2x more than the raw readout. There the discarded shots cost
    more than the suppression they remove. Caveat: the raw readout resolves
    the gradient of the biased cost, whose sign need not be the noiseless
    one; that agreement was not measured.
  Verdict. Under T1 the filter gives back the noiseless gradient exactly,
  which the raw readout cannot (its gradient shrinks by K at weight 1), and it
  is the cheaper way to resolve gradients while the kept fraction is above
  about 0.3. In large sectors at long depth (K < 0.15) the shot cost of
  filtering is higher than that of the raw readout. The intrinsic flattening
  of the gradient with the sector size (20x from 4 to 8 qubits at half
  filling) is untouched by either readout.
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
