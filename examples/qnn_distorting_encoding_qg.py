"""
When do the decayed shots help? Distorting against faithful encoding (§101)

§81 found the one case where qang lost: weight 2, pair-product encoding,
qg_ZZ readout (15 weights), trained under the calibrated T1 noise: the model
without the filter was 1.2 points better. §95 repeated it with the dual
encoding (which loads the data without distortion) and the effect vanished:
the filtered model equalled the noiseless one and the raw one was 0.3 points
behind. The explanation offered in §95 is that the decayed shots only carry
useful information when the encoding distorts the data. This study tests that
explanation directly, with new seeds, in one design.

Models (5 qubits, weight 2, trained and evaluated under equal T1, gamma =
0.08 per qubit per sublayer):
  P-zz  pair-product encoding, qg_ZZ readout   (the §81 case)
  P-z   pair-product encoding, qg_Z readout    (5 weights)
  D-zz  dual encoding, qg_ZZ readout           (the §95 case)
each read with qang (filter) and without qang (raw); P-zz also with both
channels (qang="both"). Same initialization for every readout of a model.
Protocol: seeds 1010-1014 x 3 splits x 4 datasets = 60 runs; 95% CIs across
seeds (t, 4 degrees of freedom).

Pre-registered predictions (written and committed before the run; code
debugged on 1 seed, 1 split, 2 epochs, results not looked at):
  D1  P-zz: raw - filter > 0, CI above 0 (the §81 effect replicates).
  D2  D-zz: raw - filter <= +0.5 points (mean): no raw advantage (as §95).
  D3  interaction: (raw - filter) of P-zz minus that of D-zz > 0, CI above 0.
  D4  P-zz: both channels >= raw - 0.5 points (mean).
  D5  P-z: |raw - filter| < 0.5 points (mean): the effect needs the rich
      readout.

Uses the installed library (pip install "qang>=0.6.4"), module qang.qml.
Needs scikit-learn and scipy. OMP_NUM_THREADS=1 python
examples/qnn_distorting_encoding_qg.py [seeds]

Findings:

Mean test accuracy over 60 runs (5 seeds x 3 splits x 4 datasets), trained
and evaluated under equal T1 (gamma = 0.08), weight 2:

  model                               filter (qang)   raw (no qang)   both
  P-zz  pairs, qg_ZZ (the §81 case)   0.925           0.930           0.927
  P-z   pairs, qg_Z                   0.934           0.928
  D-zz  dual, qg_ZZ (the §95 case)    0.955           0.951

  Paired differences, mean and 95% CI across seeds (points):
  P-zz raw - filter   +0.5 [-0.7, +1.6]     P-z  raw - filter   -0.6 [-1.0, -0.1]
  D-zz raw - filter   -0.4 [-1.2, +0.4]     interaction         +0.9 [-0.6, +2.4]
  P-zz both - raw     -0.3 [-1.0, +0.4]

  * D2 and D4 pass; D1, D3 and D5 fail.
  * D1 FAILS: with new seeds the §81 effect is smaller than reported (+0.5
    points against +1.2) and its CI includes 0. The decayed shots did not
    help significantly even with the distorting encoding.
  * D3 FAILS: the direction agrees with the explanation of §95 (raw - filter
    is +0.5 with the pair-product encoding and -0.4 with the dual one), but
    the interaction (+0.9 points) is not significant.
  * D5 FAILS the other way: with the 5-weight qg_Z readout the filtered model
    is better, by 0.6 [0.1, 1.0] points.
  * D2, D4: with the dual encoding raw is not ahead, and the two channels are
    within 0.5 points of raw.
  Verdict. The one case where qang lost (§81) does not hold up as a
  significant effect: over 60 new runs no readout trained under T1 beats the
  filter significantly, in any encoding, and with the simple readout the
  filter is significantly better. The faithful encoding (dual) remains the
  best weight-2 model (0.955), with the filter.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

SEEDS = (1010, 1011, 1012, 1013, 1014)
SPLITS = 3
EPOCHS = 120
GAMMA = 0.08
MODELS = {
    "P-zz": (dict(weight=2, encoding="pairs", readout="zz"), {"filter": True, "raw": False, "both": "both"}),
    "P-z": (dict(weight=2, encoding="pairs", readout="z"), {"filter": True, "raw": False}),
    "D-zz": (dict(weight=2, encoding="dual", readout="zz"), {"filter": True, "raw": False}),
}


def ci95(values):
    from scipy import stats

    v = np.asarray(values, float)
    mean = float(v.mean())
    if len(v) < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return mean, mean - half, mean + half


def run_split(Xtr, Xte, ytr, yte, sd, epochs=EPOCHS):
    rec = {}
    for name, (kw, readouts) in MODELS.items():
        m = WeightQNN(5, **kw)
        for label, q in readouts.items():
            p = m.fit(Xtr, ytr, epochs, gamma=GAMMA, qang=q, seed=sd).params_
            rec[f"{name} {label}"] = m.score(Xte, yte, p, gamma=GAMMA, qang=q)
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
    q = {name: m(f"{name} raw") - m(f"{name} filter") for name in MODELS}
    q = {f"{k} raw-filter": v for k, v in q.items()}
    q["interaction"] = q["P-zz raw-filter"] - q["D-zz raw-filter"]
    q["P-zz both-raw"] = m("P-zz both") - m("P-zz raw")
    return q


def verdict(s):
    return {
        "D1": s["P-zz raw-filter"][1] > 0,
        "D2": s["D-zz raw-filter"][0] <= 0.005,
        "D3": s["interaction"][1] > 0,
        "D4": s["P-zz both-raw"][0] >= -0.005,
        "D5": abs(s["P-z raw-filter"][0]) < 0.005,
    }


def summarize(per_seed):
    return {k: ci95([q[k] for q in per_seed]) for k in per_seed[0]}


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
    s = summarize(per_seed)
    for k, (mu, lo, hi) in s.items():
        print(f"{k}: {mu:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    v = verdict(s)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return s, v


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(seeds=tuple(args) if args else SEEDS, emit_json=bool(args))
