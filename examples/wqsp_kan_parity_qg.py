"""
Odd + even weighted-QSP activations in the one-layer KAN (§136)

§135 found that a WQSP activation of odd total weight is an odd function of
its feature, so the one-layer KAN could not use even dependences (x^2) and got
179 of 189 against 185 for qang's QNN. Here each feature gets two
activations, one odd (WQSP weights (1, 2)) and one even (weights (1, 1, 1, 1),
standard QSP of degree 4, so Re P(-x) = Re P(x)):

    score(x) = sum_i [a_i Re P_i^odd(x_i) + c_i Re P_i^even(x_i)] + b      (41 parameters)

Design note (before any training on the datasets): the even WQSP (2, 2) was
tried first and cannot represent T2(x) = 2x^2 - 1 (least-squares residual
15.1 on 101 points), so it cannot fix the parity problem; (1, 1, 1, 1) and
(2,) represent T2 exactly. (1, 1, 1, 1) is used.

Same datasets, splits, loss, optimizer and 5 seeded starts as §135.

Pre-registered predictions (committed before training):
  P1  the odd + even model gets at least 181 of 189 (2 more than §135's 179).
  P2  it is within 2 inputs of the QNN's 185 (at least 183).

python examples/wqsp_kan_parity_qg.py

Findings:

FINDINGS_PLACEHOLDER
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
import qnn_full_pipeline_qg as P  # noqa: E402
import wqsp_kan_qg as K  # noqa: E402
from qang import qsp  # noqa: E402

ODD, EVEN = (1, 2), (1, 1, 1, 1)
OUT = os.path.join(HERE, "data", "wqsp_kan_parity.json")


class ParityKAN(K.WQSPKAN):
    """One odd and one even WQSP activation per feature."""

    def __init__(self, n_features):
        self.d, self.ws = n_features, (ODD, EVEN)
        self.k = [len(w) + 1 for w in self.ws]

    def score(self, p, X):
        off, s = 0, 0.0
        acts = []
        for w, k in zip(self.ws, self.k):
            ph = p[off: off + self.d * k].reshape(self.d, k)
            off += self.d * k
            acts.append(np.stack([qsp.wqsp_poly(X[:, i], ph[i], w).real for i in range(self.d)], axis=1))
        for A in acts:
            s = s + A @ p[off: off + self.d]
            off += self.d
        return s + p[off]

    def fit(self, X, y, starts=K.STARTS, seed=0):
        from scipy.optimize import minimize

        rng = np.random.default_rng(seed)
        n_ph = self.d * sum(self.k)
        best = None
        for _ in range(starts):
            p0 = np.concatenate([rng.uniform(-np.pi, np.pi, n_ph), rng.normal(0, 1, 2 * self.d), [0.0]])
            r = minimize(self.loss, p0, args=(X, y), method="L-BFGS-B")
            if best is None or r.fun < best.fun:
                best = r
        self.params_ = best.x
        return self


def run():
    correct, total, per = 0, 0, {}
    for ds in Q.DATASETS:
        Xtr, Xte, ytr, yte = P.data(ds)
        m = ParityKAN(Xtr.shape[1]).fit(Xtr, ytr, seed=135)
        c = int(np.sum(m.predict(Xte) == yte))
        per[ds] = [c, len(yte)]
        correct += c
        total += len(yte)
    out = {"correct": correct, "inputs": total, "per dataset": per, "parameters": 4 * (3 + 5 + 2) + 1}
    print(json.dumps(out), flush=True)
    return out


def main():
    out = run()
    v = {"P1": out["correct"] >= 181, "P2": abs(out["correct"] - K.QNN_REFERENCE) <= 2}
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"results": out, "verdict": v}, fh, indent=1)
    return out, v


if __name__ == "__main__":
    main()
