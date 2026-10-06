"""
Why the loss without qang varies so much between seeds (§121)

In sections 110 and 114 the multiclass head readout lost anywhere from 2 to
34 points without the filter, depending on the seed. Theory (F7): at weight 1
under equal T1, every decayed shot lands in |0...0>, where every qg_Z = +1
(and every excitation probability p_c = 0). So the raw readout is
qg_Z_raw = K qg_Z + (1 - K) 1 with K the kept fraction, and the raw logits are

    raw_c = K L_c + (1 - K) v_c,

with L_c the noiseless logits and v a constant vector set by training alone:
v = W^T 1 + b for the head readout, v = b for the qubit readout. Without the
filter every input is pulled toward the class with the largest v_c, by an
amount that depends on the seed through W and b, not on the input. A
seed-level "pull index" should predict the loss:

    pull = (1 - K) / K * (max v - min v) / (median noiseless margin on the
           training set, top logit minus second).

Setting: MultiClassQNN, 5 qubits, digits 0-4 (5 classes), PCA to 4, seeds
1210-1229 (split and initialization per seed, 20 runs per readout), 120
epochs, trained noiselessly; equal T1 gamma = 0.08 (K = 0.472).

Pre-registered predictions (committed before the run; F7 checked in the
tests on random parameters):
  V1  head readout: Spearman correlation between the pull index and the
      accuracy lost without qang >= 0.6 over the 20 seeds.
  V2  qubit readout: the same correlation >= 0.6.
  V3  head readout: the seeds in the top third of the pull index lose at
      least 3 times as much as those in the bottom third, on average.
  V4  head readout: the loss without qang spans at least 15 points between
      the best and the worst seed (the spread of section 114 at 5 qubits).
  V5  with qang every run equals its noiseless accuracy (both readouts).

python examples/qnn_seed_spread_qg.py            # all seeds
python examples/qnn_seed_spread_qg.py 1210       # one seed, JSON rows

Findings (20 seeds per readout; 5-class digits, 271 test inputs per seed):

  readout  noiseless  without qang  loss: mean / min / max   pull index   Spearman(pull, loss)
  qubit    0.853      0.679         0.174 / -0.007 / 0.413    0.42-2.80    0.98
  head     0.933      0.718         0.215 /  0.033 / 0.417    0.71-2.36    0.65

  * V1, V2, V4, V5 pass; V3 fails.
  * F7 holds exactly: in all 40 runs the raw decisions equal
    argmax(K L + (1 - K) v) for every test input (agreement 1.000), and in
    the tests on random parameters to 1e-12.
  * The pull index, computed from the trained readout and the training set
    alone, ranks the seeds by their loss without qang: Spearman 0.98 for the
    qubit readout and 0.65 for the head.
  * V3 fails: the top third of the head seeds lose 28.0 points and the bottom
    third 11.1, a factor 2.5, not 3.
  * The spread is large: the head loss runs from 3.3 to 41.7 points and the
    qubit loss from -0.7 to 41.3.
  * V5: with qang every run equals its noiseless accuracy.
  Verdict. The seed-to-seed spread is not noise: without the filter every
  input is pulled toward the class with the largest constant term v_c, set by
  training, and the loss follows how far v is spread relative to the
  decision margins. The filter removes the pull exactly (it drops the |0...0>
  shots that carry it).
"""

import json
import sys

import numpy as np

from qang.qml import MultiClassQNN, kept_fraction

SEEDS = tuple(range(1210, 1230))
GAMMA, EPOCHS = 0.08, 120


def data(seed):
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    d = load_digits()
    k = d.target < 5
    Xtr, Xte, ytr, yte = train_test_split(d.data[k], d.target[k], test_size=0.3, stratify=d.target[k], random_state=seed)
    sc = StandardScaler().fit(Xtr)
    pca = PCA(4, random_state=seed).fit(sc.transform(Xtr))
    Xtr, Xte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    mm = MinMaxScaler((-1, 1)).fit(Xtr)
    return np.clip(mm.transform(Xtr), -1, 1), np.clip(mm.transform(Xte), -1, 1), ytr, yte


def pull_vector(m):
    head = m.params_[m.n_theta:]
    C = m.n_classes
    if m.class_readout == "qubit":
        return head[1:].copy()
    W = head[: m.n * C].reshape(m.n, C)
    return W.sum(axis=0) + head[m.n * C:]


def pull_index(m, X, K):
    L = m.logits(m.params_, X)
    s = np.sort(L, axis=1)
    margin = float(np.median(s[:, -1] - s[:, -2]))
    v = pull_vector(m)
    return (1 - K) / K * float(v.max() - v.min()) / margin


def run(seed, epochs=EPOCHS):
    Xtr, Xte, ytr, yte = data(seed)
    rows = []
    for ro in ("qubit", "head"):
        m = MultiClassQNN(5, 5, readout=ro).fit(Xtr, ytr, epochs, seed=seed)
        K = kept_fraction(GAMMA, 1, m.depth)
        L = m.logits(m.params_, Xte)
        raw_pred = np.argmax(K * L + (1 - K) * pull_vector(m)[None, :], axis=1)
        raw = m.predict(Xte, gamma=GAMMA, qang=False)
        rows.append({
            "seed": seed, "readout": ro, "noiseless": m.score(Xte, yte),
            "with qang": m.score(Xte, yte, gamma=GAMMA, qang=True),
            "without qang": m.score(Xte, yte, gamma=GAMMA, qang=False),
            "F7 agreement": float(np.mean(raw_pred == raw)), "pull": pull_index(m, Xtr, K),
        })
    return rows


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def verdict(rows):
    out = {}
    stats = {}
    for ro in ("qubit", "head"):
        R = sorted([r for r in rows if r["readout"] == ro], key=lambda r: r["pull"])
        loss = np.array([r["noiseless"] - r["without qang"] for r in R])
        pull = np.array([r["pull"] for r in R])
        t = len(R) // 3
        stats[ro] = (spearman(pull, loss), loss[-t:].mean(), loss[:t].mean(), loss.max() - loss.min())
    out["V1"] = stats["head"][0] >= 0.6
    out["V2"] = stats["qubit"][0] >= 0.6
    out["V3"] = stats["head"][1] >= 3 * stats["head"][2]
    out["V4"] = stats["head"][3] >= 0.15
    out["V5"] = all(abs(r["with qang"] - r["noiseless"]) < 1e-12 for r in rows)
    return out, stats


def main(seeds=SEEDS, emit_json=False):
    rows = []
    for sd in seeds:
        r = run(sd)
        rows += r
        if emit_json:
            print(json.dumps(r), flush=True)
    v, stats = verdict(rows)
    for ro, (rho, top, bottom, span) in stats.items():
        print(f"{ro}: Spearman {rho:.2f}, loss top third {top:.3f}, bottom third {bottom:.3f}, span {span:.3f}")
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return rows, v


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(seeds=tuple(args) if args else SEEDS, emit_json=bool(args))
