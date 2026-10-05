"""
Multiclass QNN with and without qang: each qubit a class (§110)

qang.qml.MultiClassQNN (weight 1, 5 qubits) classifies C classes with two
readouts: "qubit" (class c is qubit c; logits a p_c + b_c with p_c the
excitation probability of qubit c) and "head" (a linear softmax head on the
five qg_Z). The filter keeps the shots with exactly one excitation; under
equal T1 the filtered p_c are exactly noiseless (F4), while the raw p_c all
shrink by the kept fraction and the decayed shots pile up in |00000>.

Datasets (features scaled to [-1, 1], PCA to 4 components where needed):
iris (3 classes), wine (3 classes), digits 0-4 (5 classes). Seeds 1100-1104
(70/30 split and initialization per seed), 120 epochs; 15 runs per readout.
Noise: equal T1 gamma = 0.08 per qubit per sublayer; unequal T1 with the same
mean and spread s = 0.5 (gamma_q = 0.08 (1 + 0.5 u_q), u_q evenly over
[-1, 1]). Readings with qang (filter) and without it from exact
probabilities. Conditions: trained noiselessly and run under the noise; and
trained under equal T1 without qang (noise-aware).

Pre-registered predictions (committed before the run; code checked on iris,
one seed):
  MC1  equal T1, trained noiselessly: with qang the accuracy equals the
       noiseless accuracy in every run, both readouts.
  MC2  head readout, trained noiselessly, equal T1: without qang the mean
       accuracy is at least 10 points below the noiseless one.
  MC3  the mean loss without qang under equal T1 is smaller for the qubit
       readout than for the head readout.
  MC4  head readout trained under equal T1 without qang: within 2 points of
       the filtered noiselessly trained model (mean).
  MC5  unequal T1: mean accuracy with qang >= without qang, both readouts.

OMP_NUM_THREADS=1 python examples/qnn_multiclass_qg.py

Findings:

15 runs per readout (3 datasets x 5 seeds), exact probabilities, mean test
accuracy:

  readout   noiseless   equal T1            unequal T1          noise-aware,
                        qang / without      qang / without      without qang
  qubit     0.918       0.918 / 0.865       0.914 / 0.847       0.911
  head      0.953       0.953 / 0.863       0.949 / 0.831       0.936

  per dataset (equal T1, with qang / without qang):
  iris   (3 classes)  qubit 0.938 / 0.947    head 0.956 / 0.889
  wine   (3 classes)  qubit 0.963 / 0.944    head 0.963 / 0.941
  digits (5 classes)  qubit 0.853 / 0.704    head 0.942 / 0.758

  * MC1, MC3, MC4, MC5 pass; MC2 fails.
  * MC1: with qang the multiclass accuracy equals the noiseless one in all 30
    runs (exact, both readouts).
  * MC2 fails narrowly: without qang the head readout loses 9.1 points on
    average, not the predicted 10 or more. It loses 18.4 on 5-class digits.
  * MC3: the qubit readout loses less without qang (5.3 points), because
    under equal T1 the raw class probabilities all shrink by the same factor
    and only the offsets break the argmax; on 3-class iris raw is even 0.9
    points ahead, and on digits it loses 14.9.
  * MC4: trained under the noise without qang, the head readout comes within
    1.7 points of the filtered model (0.936 against 0.953).
  * MC5: under unequal T1 qang is ahead with both readouts (+6.7 and +11.8
    points).
  Verdict. The filter extends to multiclass with each qubit a class: exact
  under equal T1, and the gain grows with the number of classes (5 classes:
  +15 to +18 points). The cheapest route is again noiseless training plus
  the filter.
"""

import json
import sys

import numpy as np

from qang.qml import MultiClassQNN

SEEDS = (1100, 1101, 1102, 1103, 1104)
GAMMA = 0.08
SPREAD = 0.5
EPOCHS = 120


def load(name):
    from sklearn import datasets

    if name == "iris":
        d = datasets.load_iris()
        return d.data, d.target
    if name == "wine":
        d = datasets.load_wine()
        return d.data, d.target
    d = datasets.load_digits()
    m = d.target < 5
    return d.data[m], d.target[m]


def prepare(X, y, seed):
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    if X.shape[1] > 4:
        sc = StandardScaler().fit(Xtr)
        pca = PCA(4, random_state=seed).fit(sc.transform(Xtr))
        Xtr, Xte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    mm = MinMaxScaler((-1, 1)).fit(Xtr)
    return np.clip(mm.transform(Xtr), -1, 1), np.clip(mm.transform(Xte), -1, 1), ytr, yte


def unequal(n=5):
    return GAMMA * (1 + SPREAD * np.linspace(-1, 1, n))


def run(seed, datasets=("iris", "wine", "digits"), epochs=EPOCHS):
    rows = []
    for name in datasets:
        X, y = load(name)
        Xtr, Xte, ytr, yte = prepare(X, y, seed)
        C = int(y.max()) + 1
        for ro in ("qubit", "head"):
            m = MultiClassQNN(5, C, readout=ro)
            p = m.fit(Xtr, ytr, epochs, seed=seed).params_
            pna = m.fit(Xtr, ytr, epochs, gamma=GAMMA, qang=False, seed=seed).params_
            rows.append({
                "dataset": name, "readout": ro, "classes": C,
                "noiseless": m.score(Xte, yte, p),
                "T1 with qang": m.score(Xte, yte, p, gamma=GAMMA, qang=True),
                "T1 without qang": m.score(Xte, yte, p, gamma=GAMMA, qang=False),
                "unequal T1 with qang": m.score(Xte, yte, p, gamma=unequal(), qang=True),
                "unequal T1 without qang": m.score(Xte, yte, p, gamma=unequal(), qang=False),
                "noise-aware without qang": m.score(Xte, yte, pna, gamma=GAMMA, qang=False),
            })
    return rows


def summary(rows):
    out = {}
    for ro in ("qubit", "head"):
        rr = [r for r in rows if r["readout"] == ro]
        out[ro] = {k: float(np.mean([r[k] for r in rr])) for k in rr[0] if k not in ("dataset", "readout", "classes")}
    return out


def verdict(rows):
    s = summary(rows)
    return {
        "MC1": all(abs(r["T1 with qang"] - r["noiseless"]) < 1e-12 for r in rows),
        "MC2": s["head"]["noiseless"] - s["head"]["T1 without qang"] >= 0.10,
        "MC3": (s["qubit"]["noiseless"] - s["qubit"]["T1 without qang"]) < (s["head"]["noiseless"] - s["head"]["T1 without qang"]),
        "MC4": abs(s["head"]["noise-aware without qang"] - s["head"]["T1 with qang"]) <= 0.02,
        "MC5": all(s[ro]["unequal T1 with qang"] >= s[ro]["unequal T1 without qang"] for ro in ("qubit", "head")),
    }


def main(seeds=SEEDS, emit_json=False):
    rows = []
    for sd in seeds:
        r = run(sd)
        rows += r
        print(f"seed {sd} done", flush=True)
        if emit_json:
            print(json.dumps(r), flush=True)
    s = summary(rows)
    for ro, d in s.items():
        print(ro, {k: round(v, 4) for k, v in d.items()})
    for name in ("iris", "wine", "digits"):
        for ro in ("qubit", "head"):
            rr = [r for r in rows if r["dataset"] == name and r["readout"] == ro]
            print(name, ro, {k: round(float(np.mean([r[k] for r in rr])), 3) for k in rr[0] if k not in ("dataset", "readout", "classes")})
    v = verdict(rows)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return rows, v


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(seeds=tuple(args) if args else SEEDS, emit_json=bool(args))
