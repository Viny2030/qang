"""
Using the whole weight-2 sector without losing information (§90)

§87 found that the weight-2 deficit came from the pair-product encoding
(normalizing v_i v_j distorts the data) and that loading the weight-1 data on
a "ring" of 5 pairs (k, k+1) brings weight 2 within 1 point of weight 1, but
uses only 5 of the 10 weight-2 states. Here the other 5 states, the chords
(k, k+2), carry a second, undistorted view of the data:

  W-dual   v = (x, 1)/|(x, 1)| on the ring and u = (x^2, 1)/|(x^2, 1)| on the
           chords, each half with weight 1/sqrt 2 (all 10 states; library
           qang.qml.WeightQNN(encoding="dual")).
  W-dual-zz  the same with the qg_ZZ readout (rank 10 on the sector, §81).
Compared with E (weight 1) and W-ring (§87). Every model is trained without
noise and also read under equal T1 (gamma = 0.08) with qang and without it.
Encoding ceiling as in §87 (best quadratic form in the encoded amplitudes).
Protocol: seeds 900-904 x 3 splits x 4 datasets = 60 runs; 95% CIs across
seeds.

Pre-registered predictions (written and committed before the run; code
debugged on seed 1 with 2 epochs, ceilings not looked at):
  Y1  the dual encoding carries at least the information of the unary one:
      mean ceiling(W-dual) >= ceiling(E).
  Y2  using the whole sector helps the circuit: mean W-dual >= W-ring.
  Y3  W-dual >= E - 1 point (mean).
  Y4  with qang, W-dual and W-dual-zz under T1 keep their noiseless accuracy
      on all 60 runs, and the gain of qang for W-dual is >= 10 points.
  Y5  the full-rank readout helps once the sector is full: mean W-dual-zz
      >= W-dual.

Uses the installed library (pip install "qang>=0.6.0"), module qang.qml.
Needs scikit-learn and scipy. OMP_NUM_THREADS=1 python
examples/qnn_weight2_full_sector_qg.py [seeds]

Findings (python examples/qnn_weight2_full_sector_qg.py):

Seeds 900-904 x 3 splits x 4 datasets = 60 runs. Mean accuracy (trained
without noise; under T1 read with qang / without):

  model       exact   T1 qang / without   ceiling
  E           0.947   0.947 / 0.918       0.948
  W-ring      0.939   0.939 / 0.735       (= E)
  W-dual      0.937   0.937 / 0.742       0.951
  W-dual-zz   0.947   0.947 / 0.807

  quantity (95% CI across seeds)     value
  ceiling W-dual - ceiling E         +0.4 points [+0.1, +0.6]
  W-dual - W-ring                    -0.2 points [-0.7, +0.3]
  W-dual - E                         -1.0 points [-1.8, -0.2]
  W-dual-zz - W-dual                 +1.0 points [-0.4, +2.4]
  gain of qang (T1): E +2.9, W-ring +20.4, W-dual +19.5, W-dual-zz +14.1

  * Y1 PASS: the dual encoding fills all 10 states and carries slightly
    more usable information than the unary one (ceiling +0.4, CI above 0).
  * Y2 FAILS: with the qg_Z readout the full sector does not help the
    circuit (-0.2 points, CI across 0). Five qg_Z values cannot see what
    the extra states carry (rank 5 of 10, §81).
  * Y3 passes at the threshold (-0.99 points against -1).
  * Y4 PASS: with qang both dual models are exact under T1 on all 60 runs;
    qang adds 19.5 (W-dual) and 14.1 (W-dual-zz) points.
  * Y5 PASS (mean +1.0, CI across 0): with the full sector and the
    full-rank qg_ZZ readout, the weight-2 model reaches the weight-1
    accuracy (0.947 against 0.947).
  Verdict. The weight-2 gap of §79-§81 is closed: it needed both an
  encoding that does not distort the data (§87) and a readout that sees
  the whole sector (W-dual-zz = E). Neither alone was enough (§81: readout
  without encoding; here: encoding without readout). Weight 2 does not beat
  weight 1 on these small datasets, and both remain classically simulable;
  what the larger sector buys here is parity, with the largest gains of
  qang under T1 (14-20 points against 3).
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

SEEDS = (900, 901, 902, 903, 904)
SPLITS = 3
EPOCHS = 120
GAMMA = 0.08
MODELS = {"E": dict(weight=1), "W-ring": dict(weight=2, encoding="ring"),
          "W-dual": dict(weight=2, encoding="dual"), "W-dual-zz": dict(weight=2, encoding="dual", readout="zz")}
CEILING = ("E", "W-dual")


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
        if name in CEILING:
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
        "ceil Wdual - ceil E": m("W-dual ceiling") - m("E ceiling"),
        "Wdual - Wring": m("W-dual exact") - m("W-ring exact"),
        "Wdual - E": m("W-dual exact") - m("E exact"),
        "Wdualzz - Wdual": m("W-dual-zz exact") - m("W-dual exact"),
        "Wdual exact at T1 with qang": float(np.mean([r["W-dual T1 qang"] == r["W-dual exact"]
                                                     and r["W-dual-zz T1 qang"] == r["W-dual-zz exact"] for r in rows])),
    }
    for name in MODELS:
        q[f"gain {name}"] = m(f"{name} T1 qang") - m(f"{name} T1 raw")
    return q


def verdict(s):
    return {
        "Y1": s["ceil Wdual - ceil E"][0] >= 0.0,
        "Y2": s["Wdual - Wring"][0] >= 0.0,
        "Y3": s["Wdual - E"][0] >= -0.01,
        "Y4": s["Wdual exact at T1 with qang"][0] == 1.0 and s["gain W-dual"][0] >= 0.10,
        "Y5": s["Wdualzz - Wdual"][0] >= 0.0,
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
