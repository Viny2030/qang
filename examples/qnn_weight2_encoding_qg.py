"""
Where does the weight-2 QNN lose accuracy? Encoding against circuit (§87)

§79-§81: the weight-2 model W is 3-4 points below the weight-1 model E, and
§81 showed the readout is not the reason. Two candidates are left: the
pair-product encoding (amplitude v_i v_j on the state with qubits i < j
excited) and the circuit (the same 15 RBS angles acting on the 10-state
weight-2 sector). This study separates them, with every reading also given
with qang and without it.

Models (5 qubits, readout over all qg_Z, trained without noise; library
qang.qml.WeightQNN):
  E        weight 1, unary v                       (reference)
  W-pairs  weight 2, v_i v_j, normalized            (§79)
  W-ring   weight 2, v_k on the state with qubits k and k+1 (mod 5) excited:
           the weight-1 data placed on 5 of the 10 weight-2 states
Encoding ceiling: logistic regression (C = 100) on all products psi_a psi_b of
the encoded amplitudes, i.e. the best unconstrained quadratic form in the
amplitudes, which bounds what any readout <Z_i> of any orthogonal sector map
can express (each <Z_i> is such a quadratic form). E and W-ring share the
same amplitudes up to placement, so they share the ceiling.

Protocol: seeds 870-874 x 3 splits x 4 datasets = 60 runs; 95% CIs across
seeds. Under equal T1 (gamma = 0.08) the clean-trained models are also read
with and without qang.

Pre-registered predictions (written and committed before the run; code
debugged on seed 1 with few epochs):
  X1  the pair-product encoding is not the limit: ceiling(W-pairs) >=
      ceiling(E) - 1 point (mean).
  X2  the weight-2 circuit is not the limit either when the data are
      loaded like weight 1: W-ring >= E - 1 point (mean).
  X3  W-ring beats W-pairs: W-ring - W-pairs > 0, CI above 0.
  X4  with qang, W-ring under T1 keeps its noiseless accuracy on all 60
      runs, and the gain of qang for W-ring is >= 10 points (mean).

Uses the installed library (pip install "qang>=0.6.0"), module qang.qml.
Needs scikit-learn and scipy. OMP_NUM_THREADS=1 python
examples/qnn_weight2_encoding_qg.py [seeds]

Findings (python examples/qnn_weight2_encoding_qg.py):

FINDINGS_PLACEHOLDER
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

SEEDS = (870, 871, 872, 873, 874)
SPLITS = 3
EPOCHS = 120
GAMMA = 0.08
MODELS = {"E": dict(weight=1), "W-pairs": dict(weight=2), "W-ring": dict(weight=2, encoding="ring")}


def ci95(values):
    from scipy import stats

    v = np.asarray(values, float)
    mean = float(v.mean())
    if len(v) < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return mean, mean - half, mean + half


def ceiling(model, Xtr, ytr, Xte, yte):
    """Best unconstrained quadratic form in the encoded amplitudes."""
    from sklearn.linear_model import LogisticRegression

    def feats(X):
        psi = model.encode(X)
        nz = np.abs(model.encode(np.vstack([Xtr, Xte]))).sum(axis=0) > 0
        p = psi[:, nz]
        iu = np.triu_indices(p.shape[1])
        return np.einsum("si,sj->sij", p, p)[:, iu[0], iu[1]]

    clf = LogisticRegression(C=100, max_iter=20000).fit(feats(Xtr), ytr)
    return float(clf.score(feats(Xte), yte))


def run_split(Xtr, Xte, ytr, yte, sd, epochs=EPOCHS):
    rec = {}
    for name, kw in MODELS.items():
        m = WeightQNN(5, **kw).fit(Xtr, ytr, epochs, seed=sd)
        rec[f"{name} exact"] = m.score(Xte, yte)
        rec[f"{name} T1 qang"] = m.score(Xte, yte, gamma=GAMMA, qang=True)
        rec[f"{name} T1 raw"] = m.score(Xte, yte, gamma=GAMMA, qang=False)
        if name != "W-ring":
            rec[f"{name} ceiling"] = ceiling(m, Xtr, ytr, Xte, yte)
    return rec


def run_seed(seed, splits=SPLITS, epochs=EPOCHS, datasets=None):
    rows = []
    for name in datasets or Q.DATASETS:
        Xa, ya, use_pca = Q.load(name, np.random.default_rng(seed))
        for s in range(splits):
            Xtr, Xte, ytr, yte = Q.split_prepare(Xa, ya, use_pca, seed * 100 + s)
            r = run_split(Xtr, Xte, ytr, yte, seed * 1000 + 10 * s + 2, epochs)
            r["dataset"] = name
            rows.append(r)
    return rows


def quantities(rows):
    m = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    q = {
        "ceil Wp - ceil E": m("W-pairs ceiling") - m("E ceiling"),
        "Wring - E": m("W-ring exact") - m("E exact"),
        "Wring - Wpairs": m("W-ring exact") - m("W-pairs exact"),
        "Wring exact at T1 with qang": float(np.mean([r["W-ring T1 qang"] == r["W-ring exact"] for r in rows])),
    }
    for name in MODELS:
        q[f"gain {name}"] = m(f"{name} T1 qang") - m(f"{name} T1 raw")
    return q


def verdict(s):
    return {
        "X1": s["ceil Wp - ceil E"][0] >= -0.01,
        "X2": s["Wring - E"][0] >= -0.01,
        "X3": s["Wring - Wpairs"][1] > 0,
        "X4": s["Wring exact at T1 with qang"][0] == 1.0 and s["gain W-ring"][0] >= 0.10,
    }


def main(seeds=SEEDS, emit_json=False):
    per_seed, all_rows = [], {}
    for sd in seeds:
        rows = run_seed(sd)
        all_rows[sd] = rows
        q = quantities(rows)
        per_seed.append(q)
        print(f"seed {sd}: " + ", ".join(f"{k} {v:+.3f}" for k, v in q.items()), flush=True)
    if emit_json:
        print(json.dumps(all_rows))
        return all_rows, None
    summ = {k: ci95([q[k] for q in per_seed]) for k in per_seed[0]}
    for k, (mu, lo, hi) in summ.items():
        print(f"{k}: {mu:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    v = verdict(summ)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return summ, v


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(seeds=tuple(args) if args else SEEDS, emit_json=bool(args))
