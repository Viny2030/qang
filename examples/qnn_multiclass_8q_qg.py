"""
Multiclass QNN with and without qang at 8 qubits: does the gain grow with the
number of classes? (§114)

qang.qml.MultiClassQNN (weight 1) on 8 qubits, class c on qubit c, with the
two readouts of §110: "qubit" (logits a p_c + b_c, p_c the excitation
probability of qubit c) and "head" (linear softmax on the eight qg_Z). The
filter keeps the shots with exactly one excitation; under equal T1 the
filtered class scores are exactly noiseless (F4).

Data: digits restricted to classes 0..C-1 for C = 3, 5, 8, 100 samples per
class drawn per seed, 70/30 stratified split, standardization, PCA to 7
components, scaling to [-1, 1]. Seeds 1140-1144 (sampling, split and
initialization per seed), 120 epochs, depth 9 sublayers (3 layers). Noise:
equal T1 gamma = 0.08 per qubit per sublayer; unequal T1 with the same mean
and spread s = 0.5 (gamma_q = 0.08 (1 + 0.5 u_q), u_q evenly over [-1, 1]).
Readings with qang (filter) and without it from exact probabilities.
Conditions: trained noiselessly and run under the noise; trained under equal
T1 without qang (noise-aware).

Pre-registered predictions (committed before the run; code checked on seed 0
with 3 epochs, noiseless only):
  Q1  equal T1, trained noiselessly: with qang the accuracy equals the
      noiseless accuracy in every run, both readouts.
  Q2  head readout, equal T1: the mean loss without qang grows with the
      number of classes, loss(C=8) > loss(C=5) > loss(C=3).
  Q3  head readout, C = 8, equal T1: without qang the mean accuracy is at
      least 15 points below the noiseless one.
  Q4  the mean loss without qang under equal T1 is smaller for the qubit
      readout than for the head readout, at every C.
  Q5  trained under equal T1 without qang (noise-aware): within 3 points of
      the filtered noiselessly trained model (mean), in every (C, readout).
  Q6  unequal T1: mean accuracy with qang >= without qang, in every
      (C, readout).

python examples/qnn_multiclass_8q_qg.py            # all seeds
python examples/qnn_multiclass_8q_qg.py 1140 1141  # some seeds, JSON per seed

Findings:

FINDINGS_PLACEHOLDER
"""

import json
import sys

import numpy as np

from qang.qml import MultiClassQNN

N = 8
CLASSES = (3, 5, 8)
SEEDS = (1140, 1141, 1142, 1143, 1144)
GAMMA = 0.08
SPREAD = 0.5
EPOCHS = 120
PER_CLASS = 100


def prepare(C, seed, per_class=PER_CLASS):
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    d = load_digits()
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(d.target == c)[0], per_class, replace=False) for c in range(C)])
    X, y = d.data[idx], d.target[idx]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    pca = PCA(N - 1, random_state=seed).fit(sc.transform(Xtr))
    Xtr, Xte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    mm = MinMaxScaler((-1, 1)).fit(Xtr)
    return np.clip(mm.transform(Xtr), -1, 1), np.clip(mm.transform(Xte), -1, 1), ytr, yte


def unequal(n=N):
    return GAMMA * (1 + SPREAD * np.linspace(-1, 1, n))


def run(seed, classes=CLASSES, epochs=EPOCHS, noisy=True):
    rows = []
    for C in classes:
        Xtr, Xte, ytr, yte = prepare(C, seed)
        for ro in ("qubit", "head"):
            m = MultiClassQNN(N, C, readout=ro)
            p = m.fit(Xtr, ytr, epochs, seed=seed).params_
            row = {"classes": C, "readout": ro, "seed": seed, "noiseless": m.score(Xte, yte, p)}
            if noisy:
                pna = m.fit(Xtr, ytr, epochs, gamma=GAMMA, qang=False, seed=seed).params_
                row.update({
                    "T1 with qang": m.score(Xte, yte, p, gamma=GAMMA, qang=True),
                    "T1 without qang": m.score(Xte, yte, p, gamma=GAMMA, qang=False),
                    "unequal T1 with qang": m.score(Xte, yte, p, gamma=unequal(), qang=True),
                    "unequal T1 without qang": m.score(Xte, yte, p, gamma=unequal(), qang=False),
                    "noise-aware without qang": m.score(Xte, yte, pna, gamma=GAMMA, qang=False),
                })
            rows.append(row)
    return rows


KEYS = ("noiseless", "T1 with qang", "T1 without qang", "unequal T1 with qang", "unequal T1 without qang",
        "noise-aware without qang")


def summary(rows):
    out = {}
    for C in sorted({r["classes"] for r in rows}):
        for ro in ("qubit", "head"):
            rr = [r for r in rows if r["classes"] == C and r["readout"] == ro]
            if rr:
                out[(C, ro)] = {k: float(np.mean([r[k] for r in rr])) for k in KEYS}
    return out


def verdict(rows):
    s = summary(rows)
    loss = {k: v["noiseless"] - v["T1 without qang"] for k, v in s.items()}
    Cs = sorted({C for C, _ in s})
    return {
        "Q1": all(abs(r["T1 with qang"] - r["noiseless"]) < 1e-12 for r in rows),
        "Q2": all(loss[(a, "head")] < loss[(b, "head")] for a, b in zip(Cs, Cs[1:])),
        "Q3": loss[(max(Cs), "head")] >= 0.15,
        "Q4": all(loss[(C, "qubit")] < loss[(C, "head")] for C in Cs),
        "Q5": all(abs(v["noise-aware without qang"] - v["T1 with qang"]) <= 0.03 for v in s.values()),
        "Q6": all(v["unequal T1 with qang"] >= v["unequal T1 without qang"] for v in s.values()),
    }


def report(rows):
    print(f"{'C':>2} {'readout':<7}" + "".join(f"{k:>26}" for k in KEYS))
    for (C, ro), v in summary(rows).items():
        print(f"{C:>2} {ro:<7}" + "".join(f"{v[k]:>26.3f}" for k in KEYS))
    v = verdict(rows)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return v


def main(seeds=SEEDS, emit_json=False):
    rows = []
    for sd in seeds:
        r = run(sd)
        rows += r
        print(f"seed {sd} done", flush=True)
        if emit_json:
            print(json.dumps(r), flush=True)
    return rows, report(rows)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(seeds=tuple(args) if args else SEEDS, emit_json=bool(args))
