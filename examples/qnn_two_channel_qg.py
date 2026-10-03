"""
A two-channel qg readout: keep the filter, and use the decayed shots too (§95)

§81 found the one case where qang lost: trained under the calibrated T1
noise with a rich readout (15 weights), the model without the filter was
1.2 points better, because the shots that decayed out of the sector still
carry information about where the excitations were. The two-channel readout
(qang.qml: qang="both") gives the classifier both the filtered features and
the raw ones (2 x 15 weights). Its features contain the raw readout exactly,
and the filtered one, so a model trained under the noise can use either.

Models (5 qubits, trained under equal T1 gamma = 0.08 per qubit per sublayer,
evaluated under the same noise), each with three readouts, with qang
(filter), without qang (raw), and both channels:
  W  weight 2, dual encoding, qg_ZZ readout (the best weight-2 model, §90)
  E  weight 1, qg_Z readout
Protocol: seeds 950-954 x 3 splits x 4 datasets = 60 runs; 95% CIs across
seeds; same initialization for the three readouts of a model.

Pre-registered predictions (written and committed before the run; code
debugged on seed 1 with 2 epochs, results not looked at):
  B1  for W, the two channels are at least as good as either: mean both >=
      raw and mean both >= filter.
  B2  for W, both - raw > 0 with the CI above 0.
  B3  the §81 effect replicates with the dual encoding: for W, mean raw >=
      filter.
  B4  for E (5 features, little to gain), both >= max(raw, filter) - 0.5
      points (mean).

Uses the installed library (pip install "qang>=0.6.4"), module qang.qml.
Needs scikit-learn and scipy. OMP_NUM_THREADS=1 python
examples/qnn_two_channel_qg.py [seeds]

Findings (python examples/qnn_two_channel_qg.py):

Mean test accuracy over 60 runs (5 seeds x 3 splits x 4 datasets), trained
and evaluated under equal T1 (gamma = 0.08):

  model                         noiseless   filter (qang)   raw (no qang)   both channels
  W  weight 2, dual, qg_ZZ      0.953       0.953           0.950           0.950
  E  weight 1, qg_Z             0.954       0.954           0.952           0.955

  Paired differences, mean and 95% CI across seeds (points):
  W  both - raw      -0.0 [-0.6, +0.6]     W  both - filter   -0.3 [-0.8, +0.2]
  W  raw - filter    -0.3 [-0.8, +0.2]     E  both - best     +0.0 [-0.4, +0.4]

  * B4 passes; B1, B2, B3 fail.
  * B3 FAILS, and it is the main finding: the §81 effect does not replicate
    with the dual encoding. Trained under the noise, the filtered model is
    as good as the raw one or better (0.953 against 0.950; it is level with
    the noiseless model, as F3 requires), so the decayed shots carried no
    extra information here. In §81 (pair-product encoding) they did (+1.2
    points for raw); the §81 gap came with an encoding that distorted the
    data.
  * B1, B2 FAIL: with nothing to recover from the decayed shots, the two
    channels add nothing (both = raw to 0.01 points, 0.3 points below the
    filter, CIs across zero). The extra 15 readout weights did not hurt
    either.
  * B4: for weight 1 the two channels match the best single readout
    (+0.0 points).
  Verdict. The two-channel readout is safe but not useful on these
  datasets: it never lost more than half a point and never gained. With a
  faithful encoding, filter-only readout under equal T1 is as good as any
  readout trained under the noise, and it is the only one that equals the
  noiseless model exactly. qang="both" stays in the library as an option
  for encodings or devices where the decayed shots do carry information.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

SEEDS = (950, 951, 952, 953, 954)
SPLITS = 3
EPOCHS = 120
GAMMA = 0.08
MODELS = {"W": dict(weight=2, encoding="dual", readout="zz"), "E": dict(weight=1)}
READOUTS = {"filter": True, "raw": False, "both": "both"}


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
    for name, kw in MODELS.items():
        m = WeightQNN(5, **kw)
        rec[f"{name} exact"] = m.fit(Xtr, ytr, epochs, seed=sd).score(Xte, yte)
        for label, q in READOUTS.items():
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
    q = {}
    for name in MODELS:
        q[f"{name} both-raw"] = m(f"{name} both") - m(f"{name} raw")
        q[f"{name} both-filter"] = m(f"{name} both") - m(f"{name} filter")
        q[f"{name} raw-filter"] = m(f"{name} raw") - m(f"{name} filter")
        q[f"{name} both-best"] = m(f"{name} both") - max(m(f"{name} raw"), m(f"{name} filter"))
    return q


def verdict(s):
    return {
        "B1": s["W both-raw"][0] >= 0 and s["W both-filter"][0] >= 0,
        "B2": s["W both-raw"][1] > 0,
        "B3": s["W raw-filter"][0] >= 0,
        "B4": s["E both-best"][0] >= -0.005,
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
