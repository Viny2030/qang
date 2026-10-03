"""
A weight-2 QNN with two-qubit qg_ZZ readout, with and without qang (§81)

§79-§80: the weight-2 model W is 3.0 points less accurate than the weight-1
model E, although its sector is twice as large (10 states against 5). One
reason, stated before the run: W's readout z = sum_i c_i <Z_i> + b sees only
5 linear functions of the 10 sector probabilities (rank 5). Adding the
correlations qg_ZZ^(ij) = <Z_i Z_j> (i < j) makes the readout a linear
function of the whole sector distribution (rank 10). In the weight-1 sector
the same correlations are linear in the <Z_i> and add nothing (rank 5 = 5).
Both ranks are checked in the tests.

Model WZZ: the §79 model W with readout="zz" (5 + 10 = 15 trainable readout
weights), from the library (qang.qml.WeightQNN). Every result with qang
(filtered readout) and without it (raw). Protocol as §80: seeds 810-814 x 3
splits x 4 datasets = 60 runs; 95% CIs across seeds (seed means, t with 4
degrees of freedom). Equal T1, gamma = 0.08 per qubit per sublayer.

Per run: E (readout z) and W (readout z) trained without noise, for
reference; WZZ trained without noise, evaluated exactly and under T1 with and
without qang; WZZ trained under T1 with qang and without it.

Pre-registered predictions (written and committed before the run; code
debugged on seed 1 with few epochs):
  U1  F4 holds for the correlation readout: under equal T1, WZZ trained
      without noise and read with qang has exactly its noiseless accuracy on
      all 60 runs.
  U2  the correlations help weight 2: WZZ exact - W exact > 0, CI above 0.
  U3  they close the gap to weight 1: mean WZZ exact >= E exact - 1 point.
  U4  qang still matters for training without noise: mean gain of qang for
      WZZ (trained clean, T1) >= 10 points.
  U5  with noise-aware training qang makes no measurable difference:
      |mean (aware qang - aware raw)| < 1 point for WZZ.

Uses the installed library (pip install "qang>=0.6.0"), module qang.qml.
Needs scikit-learn and scipy. OMP_NUM_THREADS=1 python
examples/qnn_zz_readout_qg.py [seeds]

Findings (python examples/qnn_zz_readout_qg.py):

Seeds 810-814 x 3 splits x 4 datasets = 60 runs. Mean accuracy:
E (weight 1) 0.965, W (weight 2, qg_Z readout) 0.931, WZZ (weight 2, qg_Z +
qg_ZZ readout) 0.925. Under T1, WZZ trained clean: with qang 0.925, without
0.776; trained under T1: with qang 0.925, without 0.937.

  quantity (95% CI across seeds)        value
  WZZ - W (exact)                       -0.6 points [-1.5, +0.4]
  WZZ - E (exact)                       -4.0 points [-5.8, -2.2]
  gain of qang, trained clean, T1       +14.9 points [+9.9, +19.9]
  qang - without, trained under T1      -1.2 points [-2.0, -0.4]
  F4 (with qang = exact)                60 of 60 runs

  * U1 PASS: the filter is exact for the correlation readout too.
  * U2 FAILS: the correlations do not help (-0.6 points, CI across 0),
    although they make the readout a linear function of the whole sector
    distribution (rank 10 against 5). The rank argument stated before the
    run was right about the readout and wrong about the cause: the
    weight-2 gap does not come from the readout.
  * U3 FAILS: WZZ stays 4.0 points below E. What limits weight 2 here is
    the pair-product encoding and the circuit, not what is read out.
  * U4 PASS: trained without noise, qang adds 14.9 points.
  * U5 FAILS, against qang: trained under T1, the model without qang is
    1.2 points better, with a CI that excludes 0. With 15 readout weights
    the raw model learns to use the decayed shots (weight 0 and 1), which
    still carry information about where the excitations were before the
    decay; the filter throws that information away. In §80, with 5
    readout weights, the two were equal.
  Verdict. The qg_ZZ readout does not close the weight-2 gap. It shows a
  limit of the filter that had not appeared before: when the model is
  trained under the calibrated noise and has a rich readout, discarding
  the decayed shots costs accuracy (1.2 points). qang remains decisive for
  noise-free training (+14.9 points) and exact under equal T1.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

SEEDS = (810, 811, 812, 813, 814)
SPLITS = 3
EPOCHS = 120
GAMMA = 0.08


def ci95(values):
    from scipy import stats

    v = np.asarray(values, float)
    mean = float(v.mean())
    if len(v) < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return mean, mean - half, mean + half


def run_split(Xtr, Xte, ytr, yte, sd, epochs=EPOCHS):
    E = WeightQNN(5, 1)
    W = WeightQNN(5, 2)
    WZ = WeightQNN(5, 2, readout="zz")
    rec = {}
    rec["E exact"] = E.fit(Xtr, ytr, epochs, seed=sd).score(Xte, yte)
    rec["W exact"] = W.fit(Xtr, ytr, epochs, seed=sd).score(Xte, yte)
    clean = WZ.fit(Xtr, ytr, epochs, seed=sd).params_
    rec["WZZ exact"] = WZ.score(Xte, yte, clean)
    rec["WZZ T1 clean qang"] = WZ.score(Xte, yte, clean, gamma=GAMMA, qang=True)
    rec["WZZ T1 clean raw"] = WZ.score(Xte, yte, clean, gamma=GAMMA, qang=False)
    aq = WZ.fit(Xtr, ytr, epochs, gamma=GAMMA, qang=True, seed=sd).params_
    ar = WZ.fit(Xtr, ytr, epochs, gamma=GAMMA, qang=False, seed=sd).params_
    rec["WZZ T1 aware qang"] = WZ.score(Xte, yte, aq, gamma=GAMMA, qang=True)
    rec["WZZ T1 aware raw"] = WZ.score(Xte, yte, ar, gamma=GAMMA, qang=False)
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
    return {
        "WZZ-W": m("WZZ exact") - m("W exact"),
        "WZZ-E": m("WZZ exact") - m("E exact"),
        "G_ZZ": m("WZZ T1 clean qang") - m("WZZ T1 clean raw"),
        "A_ZZ": m("WZZ T1 aware qang") - m("WZZ T1 aware raw"),
        "F4 ok": float(np.mean([r["WZZ T1 clean qang"] == r["WZZ exact"] for r in rows])),
    }


def verdict(summary):
    s = summary
    return {
        "U1": s["F4 ok"][0] == 1.0,
        "U2": s["WZZ-W"][1] > 0,
        "U3": s["WZZ-E"][0] >= -0.01,
        "U4": s["G_ZZ"][0] >= 0.10,
        "U5": abs(s["A_ZZ"][0]) < 0.01,
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
