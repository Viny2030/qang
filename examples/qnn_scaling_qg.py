"""
QNN with and without qang at 5, 6 and 8 qubits, weight 1 and 2 (§111)

§94/§96 showed that the filter's advantage in the qg_Z readout grows with
depth and weight. This study asks whether that reaches classification
accuracy as the network grows. The RBS layers are now built directly in each
weight sector (WeightQNN.block_unitaries), which makes 8-qubit training
practical.

Models: qang.qml.WeightQNN with n = 5, 6, 8 qubits (n - 1 PCA features scaled
to [-1, 1]), 3 layers of the default brick sublayers; weight 1 (qg_Z
readout) and weight 2 (dual encoding, qg_ZZ readout, the best weight-2 model
of §90). Datasets: breast cancer, wine (classes 0/1), digits (3 vs 8), 200
samples each as in §80; seeds 1110-1112 (70/30 split and initialization per
seed); 120 epochs; exact probabilities.
Noise: equal T1 gamma = 0.08 per qubit per sublayer (kept fraction
(1 - 0.08)^(weight x depth)); unequal T1 with the same mean and spread 0.5.
Conditions: trained noiselessly and run under the noise with qang and
without it; trained under equal T1 without qang (noise-aware).

Pre-registered predictions (committed before the run; code checked on one
dataset, one seed, 5 qubits):
  SC1  trained noiselessly, run under equal T1: with qang the accuracy equals
       the noiseless accuracy in every run.
  SC2  the mean loss without qang (noiseless accuracy minus T1 accuracy
       without qang) is larger at 8 qubits than at 5, for both weights.
  SC3  at every n the mean loss without qang is larger at weight 2 than at
       weight 1.
  SC4  noise-aware training without qang comes within 3 points of the
       filtered noiselessly trained model, in every (n, weight) cell (mean).
  SC5  under unequal T1 the mean accuracy with qang is at least the one
       without, in every cell.

OMP_NUM_THREADS=1 python examples/qnn_scaling_qg.py [seeds]

Findings:

FINDINGS_PLACEHOLDER
"""

import json
import sys
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
from qang.qml import WeightQNN, kept_fraction  # noqa: E402

SEEDS = (1110, 1111, 1112)
SIZES = (5, 6, 8)
GAMMA, SPREAD, EPOCHS = 0.08, 0.5, 120
DATASETS = ("cancer", "wine", "digits")
MODELS = {1: dict(weight=1), 2: dict(weight=2, encoding="dual", readout="zz")}


def prepare(name, n, seed):
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    Xa, ya, _ = Q.load(name, np.random.default_rng(seed))
    Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.3, stratify=ya, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n - 1, random_state=seed).fit(sc.transform(Xtr))
    Ztr, Zte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    lo, hi = Ztr.min(0), Ztr.max(0)
    f = lambda Z: np.clip(2 * (Z - lo) / (hi - lo) - 1, -1, 1)  # noqa: E731
    return f(Ztr), f(Zte), ytr, yte


def run(seed, sizes=SIZES, datasets=DATASETS, epochs=EPOCHS):
    rows = []
    for n in sizes:
        unequal = GAMMA * (1 + SPREAD * np.linspace(-1, 1, n))
        for w, kw in MODELS.items():
            for name in datasets:
                Xtr, Xte, ytr, yte = prepare(name, n, seed)
                m = WeightQNN(n, **kw)
                p = m.fit(Xtr, ytr, epochs, seed=seed).params_
                pna = m.fit(Xtr, ytr, epochs, gamma=GAMMA, qang=False, seed=seed).params_
                rows.append({
                    "n": n, "weight": w, "dataset": name, "kept": kept_fraction(GAMMA, w, m.depth),
                    "noiseless": m.score(Xte, yte, p),
                    "T1 with qang": m.score(Xte, yte, p, gamma=GAMMA, qang=True),
                    "T1 without qang": m.score(Xte, yte, p, gamma=GAMMA, qang=False),
                    "unequal with qang": m.score(Xte, yte, p, gamma=unequal, qang=True),
                    "unequal without qang": m.score(Xte, yte, p, gamma=unequal, qang=False),
                    "noise-aware without qang": m.score(Xte, yte, pna, gamma=GAMMA, qang=False),
                })
                print(json.dumps(rows[-1]), flush=True)
    return rows


def cells(rows):
    out = {}
    for n in sorted({r["n"] for r in rows}):
        for w in sorted({r["weight"] for r in rows}):
            rr = [r for r in rows if r["n"] == n and r["weight"] == w]
            if rr:
                out[(n, w)] = {k: float(np.mean([r[k] for r in rr])) for k in rr[0] if k not in ("n", "weight", "dataset")}
    return out


def verdict(rows):
    c = cells(rows)
    loss = {k: v["noiseless"] - v["T1 without qang"] for k, v in c.items()}
    ns = sorted({k[0] for k in c})
    return {
        "SC1": all(abs(r["T1 with qang"] - r["noiseless"]) < 1e-12 for r in rows),
        "SC2": all(loss[(ns[-1], w)] > loss[(ns[0], w)] for w in (1, 2)),
        "SC3": all(loss[(n, 2)] > loss[(n, 1)] for n in ns),
        "SC4": all(abs(v["noise-aware without qang"] - v["T1 with qang"]) <= 0.03 for v in c.values()),
        "SC5": all(v["unequal with qang"] >= v["unequal without qang"] for v in c.values()),
    }


def main(seeds=SEEDS):
    rows = []
    for sd in seeds:
        rows += run(sd)
    for k, v in cells(rows).items():
        print(k, {a: round(b, 3) for a, b in v.items()})
    v = verdict(rows)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return rows, v


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(seeds=tuple(args) if args else SEEDS)
