"""
A multiclass readout with no pull: robust without the filter (§123)

F7 (section 121): at weight 1 under equal T1 the raw logits are
K L + (1 - K) v, with v = W^T 1 + b (head readout) or v = b (qubit readout).
If v is the same for every class, the pull is a constant and the raw argmax
equals the noiseless one for every input, with no filter. That happens
exactly when the readout is linear in the excitation probabilities with no
bias: logits = W^T (qg_Z - 1) for the head readout (that is -2 W^T p), and
logits = a p_c for the qubit readout. Decayed shots then scale every logit
by K and change no decision. This "pull-free" readout gives up the class
biases; the study measures what that costs and what it buys.

Setting: MultiClassQNN, 5 qubits, digits 0-4 (5 classes), PCA to 4, seeds
1230-1239, 120 epochs, trained noiselessly; standard readout against
pull-free readout (same angles' initialization), both "qubit" and "head".
Noise: equal T1 gamma = 0.08 (K = 0.472); unequal T1 with spread 0.5
(gamma_q = 0.08 (1 + 0.5 u_q)); readings from exact probabilities and from
1000 shots. The exact invariance under equal T1 is checked in the tests.

Pre-registered predictions (committed before the run):
  P1  the pull-free noiseless accuracy is within 2 points of the standard
      one on average, both readouts.
  P2  head readout, equal T1, without qang: pull-free beats standard by at
      least 10 points on average.
  P3  unequal T1, without qang: pull-free is within 2 points of its own
      noiseless accuracy on average, both readouts.
  P4  head readout, unequal T1: pull-free without qang is within 1 point of
      standard with qang, on average.
  P5  1000 shots, equal T1: pull-free without qang is within 2 points of
      standard with qang on average, both readouts.

python examples/qnn_pull_free_qg.py            # all seeds
python examples/qnn_pull_free_qg.py 1230       # one seed, JSON rows

Findings:

FINDINGS_PLACEHOLDER
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qnn_seed_spread_qg as S  # noqa: E402
from qang.qml import MultiClassQNN  # noqa: E402

SEEDS = tuple(range(1230, 1240))
GAMMA, SPREAD, SHOTS, EPOCHS = 0.08, 0.5, 1000, 120


class PullFreeQNN(MultiClassQNN):
    """MultiClassQNN whose logits are linear in the excitation probabilities with
    no bias: a p_c (qubit readout) or W^T (qg_Z - 1) (head readout)."""

    def _logits(self, R, head):
        C = self.n_classes
        if self.class_readout == "qubit":
            return head[0] * (1 - R[:, :C]) / 2
        W = head[: self.n * C].reshape(self.n, C)
        return (R - 1) @ W

    def fit(self, X, y, epochs=120, lr=0.1, gamma=None, dephasing=0.0, qang=True, seed=0, h=1e-4):
        y = np.asarray(y, int)
        Y = np.eye(self.n_classes)[y]
        psi = self.encode(X)
        rng = np.random.default_rng(seed)
        nt, C = self.n_theta, self.n_classes
        head0 = (np.concatenate([[4.0], np.zeros(C)]) if self.class_readout == "qubit"
                 else np.concatenate([rng.normal(0, 0.5, self.n * C), np.zeros(C)]))
        p = np.concatenate([rng.uniform(-np.pi, np.pi, nt), head0])
        m = np.zeros_like(p)
        v = np.zeros_like(p)

        def feats(th):
            return self.qg_z(self.probs(th, psi, gamma, dephasing), qang)

        for t in range(1, epochs + 1):
            th, head = p[:nt], p[nt:]
            R = feats(th)
            Z = self._logits(R, head)
            Z = Z - Z.max(axis=1, keepdims=True)
            Pr = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
            G = (Pr - Y) / len(y)
            g = np.zeros_like(p)
            if self.class_readout == "qubit":
                g[nt] = np.sum(G * (1 - R[:, :C]) / 2)
            else:
                g[nt:nt + self.n * C] = ((R - 1).T @ G).ravel()
            for k in range(nt):
                e = np.zeros(nt)
                e[k] = h
                dZ = (self._logits(feats(th + e), head) - self._logits(feats(th - e), head)) / (2 * h)
                g[k] = np.sum(G * dZ)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g**2
            p = p - lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        self.params_ = p
        return self


def unequal():
    return GAMMA * (1 + SPREAD * np.linspace(-1, 1, 5))


def run(seed, epochs=EPOCHS):
    Xtr, Xte, ytr, yte = S.data(seed)
    rows = []
    for ro in ("qubit", "head"):
        for kind, cls in (("standard", MultiClassQNN), ("pull-free", PullFreeQNN)):
            m = cls(5, 5, readout=ro).fit(Xtr, ytr, epochs, seed=seed)
            sc = lambda **kw: m.score(Xte, yte, **kw)  # noqa: E731
            rows.append({
                "seed": seed, "readout": ro, "model": kind, "noiseless": sc(),
                "T1 qang": sc(gamma=GAMMA, qang=True), "T1 raw": sc(gamma=GAMMA, qang=False),
                "unequal qang": sc(gamma=unequal(), qang=True), "unequal raw": sc(gamma=unequal(), qang=False),
                "shots qang": sc(gamma=GAMMA, qang=True, shots=SHOTS, seed=seed),
                "shots raw": sc(gamma=GAMMA, qang=False, shots=SHOTS, seed=seed),
            })
    return rows


KEYS = ("noiseless", "T1 qang", "T1 raw", "unequal qang", "unequal raw", "shots qang", "shots raw")


def summary(rows):
    return {(ro, k): {c: float(np.mean([r[c] for r in rows if r["readout"] == ro and r["model"] == k])) for c in KEYS}
            for ro in ("qubit", "head") for k in ("standard", "pull-free")}


def verdict(rows):
    s = summary(rows)
    R = ("qubit", "head")
    return {
        "P1": all(abs(s[(ro, "pull-free")]["noiseless"] - s[(ro, "standard")]["noiseless"]) <= 0.02 for ro in R),
        "P2": s[("head", "pull-free")]["T1 raw"] - s[("head", "standard")]["T1 raw"] >= 0.10,
        "P3": all(s[(ro, "pull-free")]["noiseless"] - s[(ro, "pull-free")]["unequal raw"] <= 0.02 for ro in R),
        "P4": abs(s[("head", "pull-free")]["unequal raw"] - s[("head", "standard")]["unequal qang"]) <= 0.01,
        "P5": all(abs(s[(ro, "pull-free")]["shots raw"] - s[(ro, "standard")]["shots qang"]) <= 0.02 for ro in R),
    }


def main(seeds=SEEDS, emit_json=False):
    rows = []
    for sd in seeds:
        r = run(sd)
        rows += r
        print(f"seed {sd} done", flush=True)
        if emit_json:
            print(json.dumps(r), flush=True)
    print(f"{'readout':<8}{'model':<11}" + "".join(f"{k:>14}" for k in KEYS))
    for (ro, k), v in summary(rows).items():
        print(f"{ro:<8}{k:<11}" + "".join(f"{v[c]:>14.3f}" for c in KEYS))
    v = verdict(rows)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return rows, v


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(seeds=tuple(args) if args else SEEDS, emit_json=bool(args))
