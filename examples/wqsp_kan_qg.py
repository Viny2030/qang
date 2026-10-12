"""
Weighted-QSP activations in a one-layer Kolmogorov-Arnold network (§135)

Sarkar et al. (arXiv:2609.21567) use weighted QSP polynomials as the learnable
activations of Kolmogorov-Arnold networks. Here the smallest such model is
trained on the four datasets of the QNN sections (iris, cancer, wine, digits;
4 features in [-1, 1]; same splits, 189 test inputs):

    score(x) = sum_i a_i Re P_i(x_i) + b,

where P_i is a WQSP polynomial of feature i (qang.qsp; Re P is the qg_Z read
by a Hadamard test). Two activation types:
  WQSP  weights (1, 2): 3 phases per feature, degree 3.
  QSP   weights (1, 1, 1): 4 phases per feature, degree 3.
Logistic loss, scipy L-BFGS from 5 seeded starts, exact (noiseless) Re P.
Reference: qang's weight-1 QNN trained noiselessly gets 185 of 189 (§124).

Pre-registered predictions (committed before training):
  K1  the WQSP model gets at least 180 of 189 test inputs right (95%).
  K2  the WQSP model is within 2 inputs of the QNN's 185.
  K3  the WQSP model (3 phases per feature) is within 2 inputs of the QSP
      model (4 phases per feature).

python examples/wqsp_kan_qg.py

Findings:

FINDINGS_PLACEHOLDER
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import qnn_classifier_qg as Q  # noqa: E402
import qnn_full_pipeline_qg as P  # noqa: E402
from qang import qsp  # noqa: E402

TYPES = {"WQSP": (1, 2), "QSP": (1, 1, 1)}
STARTS = 5
QNN_REFERENCE = 185
OUT = os.path.join(HERE, "data", "wqsp_kan.json")


class WQSPKAN:
    def __init__(self, n_features, weights):
        self.d, self.w = n_features, tuple(weights)
        self.k = len(self.w) + 1                       # phases per feature

    def _split(self, p):
        ph = p[: self.d * self.k].reshape(self.d, self.k)
        return ph, p[self.d * self.k: self.d * self.k + self.d], p[-1]

    def features(self, X, ph):
        return np.stack([qsp.wqsp_poly(X[:, i], ph[i], self.w).real for i in range(self.d)], axis=1)

    def score(self, p, X):
        ph, a, b = self._split(p)
        return self.features(X, ph) @ a + b

    def loss(self, p, X, y):
        s = self.score(p, X)
        return float(np.mean(np.logaddexp(0, s) - y * s))

    def fit(self, X, y, starts=STARTS, seed=0):
        from scipy.optimize import minimize

        rng = np.random.default_rng(seed)
        best = None
        for _ in range(starts):
            p0 = np.concatenate([rng.uniform(-np.pi, np.pi, self.d * self.k), rng.normal(0, 1, self.d), [0.0]])
            r = minimize(self.loss, p0, args=(X, y), method="L-BFGS-B")
            if best is None or r.fun < best.fun:
                best = r
        self.params_ = best.x
        return self

    def predict(self, X):
        return (self.score(self.params_, X) > 0).astype(int)


def run():
    out = {}
    for name, w in TYPES.items():
        correct, total = 0, 0
        per = {}
        for ds in Q.DATASETS:
            Xtr, Xte, ytr, yte = P.data(ds)
            m = WQSPKAN(Xtr.shape[1], w).fit(Xtr, ytr, seed=135)
            c = int(np.sum(m.predict(Xte) == yte))
            per[ds] = [c, len(yte)]
            correct += c
            total += len(yte)
        out[name] = {"correct": correct, "inputs": total, "per dataset": per,
                     "parameters": Xtr.shape[1] * (len(w) + 2) + 1}
        print(json.dumps({name: out[name]}), flush=True)
    return out


def verdict(out):
    w, q = out["WQSP"]["correct"], out["QSP"]["correct"]
    return {"K1": w >= 180, "K2": abs(w - QNN_REFERENCE) <= 2, "K3": abs(w - q) <= 2}


def main():
    out = run()
    v = verdict(out)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"results": out, "verdict": v}, fh, indent=1)
    return out, v


if __name__ == "__main__":
    main()
