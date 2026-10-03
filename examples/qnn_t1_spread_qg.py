"""
How much T1 spread across qubits can the qg filter take? (§82)

The filter is exact for a weight-conserving QNN when every qubit has the same
damping (§78 F4). Real devices have a spread of T1 across qubits. §78 and §80
measured one spread (+-50% around gamma = 0.08) and found a small loss
(0.5-0.7 points over 60 runs). This study maps the loss as a function of the
spread, to give a practical rule: up to what spread can a network trained on
a simulator be run with the filter alone, without a noise model?

Damping per qubit after every sublayer: gamma_q = 0.08 (1 + s u_q), with u_q
evenly spaced in [-1, 1] and randomly assigned to the qubits in each run
(seeded); s = 0, 0.1, 0.2, 0.3, 0.5, 0.75, 1 (at s = 1 gamma_q runs from 0 to
0.16). Since gamma ~ t / T1 for short sublayers, s is the relative spread of
the decay rates 1/T1. Models E (weight 1) and W (weight 2), readout over all
qg_Z, trained without noise (the simulator-to-hardware case) and evaluated
exactly and under each spread, with qang (filtered) and without it (raw).
Protocol: seeds 820-824 x 3 splits x 4 datasets = 60 runs; 95% CIs across
seeds.

Quantities, per spread s and model:
  loss(s) = exact - with qang        (what the spread costs the filter)
  gain(s) = with qang - without qang (what qang still adds)

Pre-registered predictions (written and committed before the run; code
debugged on seed 1 with few epochs):
  V1  at s = 0 the loss is exactly 0 on every run (F4), for E and W.
  V2  at s = 0.5 (the §78/§80 setting) the mean loss is below 1 point for E
      and W.
  V3  even at s = 1 (gamma from 0 to 0.16) the mean loss stays below 2
      points for E and W.
  V4  qang still helps at every spread: gain(s) > 0 with the CI above 0 for
      every s and both models.
  V5  weight 2 is more sensitive: mean loss of W >= mean loss of E at every
      s >= 0.3.

Uses the installed library (pip install "qang>=0.6.0"), module qang.qml.
Needs scikit-learn and scipy. OMP_NUM_THREADS=1 python
examples/qnn_t1_spread_qg.py [seeds]

Findings (python examples/qnn_t1_spread_qg.py):

Seeds 820-824 x 3 splits x 4 datasets = 60 runs per model. Loss of the
filter (exact - with qang) and gain of qang (with - without), in points,
mean and 95% CI across seeds:

  spread s   E loss              E gain               W loss              W gain
  0          0                   +2.4 [+0.5, +4.2]    0                   +15.5 [+10.3, +20.8]
  0.1        -0.0 [-0.2, +0.2]   +2.5 [+0.4, +4.5]    +0.0 [-0.3, +0.4]   +15.3 [+9.7, +20.8]
  0.2        +0.1 [-0.2, +0.3]   +2.5 [+0.1, +4.9]    -0.1 [-0.3, +0.2]   +15.1 [+9.9, +20.2]
  0.3        +0.1 [-0.5, +0.7]   +2.5 [-0.1, +5.1]    -0.2 [-0.8, +0.5]   +15.3 [+10.2, +20.5]
  0.5        +0.3 [-0.8, +1.5]   +3.1 [+0.3, +6.0]    +0.2 [-0.2, +0.6]   +15.2 [+10.5, +19.9]
  0.75       +0.9 [-0.7, +2.4]   +3.4 [+0.1, +6.6]    +0.7 [-0.1, +1.5]   +15.1 [+9.7, +20.6]
  1          +1.4 [-0.7, +3.5]   +3.5 [+0.4, +6.6]    +1.3 [+0.6, +1.9]   +15.1 [+9.9, +20.4]

  * V1, V2, V3 pass; V4 and V5 fail.
  * V1: at s = 0 the filter is exact on all 60 runs (both models).
  * V2: at s = 0.5 the loss is 0.3 (E) and 0.2 (W) points.
  * V3: even at s = 1 (gamma from 0 to 0.16) it stays at 1.4 and 1.3.
  * V4 FAILS, narrowly: the gain of qang is positive at every spread, but
    for E at s = 0.3 its CI reaches -0.1 points. For W every CI is far
    above 0.
  * V5 FAILS: weight 2 is not more sensitive to the spread than weight 1
    (its loss is smaller at every s >= 0.2), although its states decay
    twice as fast. A possible reason, not tested here: the distortion
    depends on differences between the decay rates of the excited qubits,
    and with two excitations part of those differences averages out.
  Practical rule: a network trained on a simulator can be run with the
  filter alone, losing less than 1 point, as long as the decay rates 1/T1
  of the qubits differ by up to about +-80% around their mean (the loss
  reaches 1 point between s = 0.75 and s = 1); with a larger spread, train
  under the calibrated noise. With or without spread, qang adds 2.4-3.5
  points at weight 1 and about 15 at weight 2 over the raw readout.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

SEEDS = (820, 821, 822, 823, 824)
SPLITS = 3
EPOCHS = 120
GAMMA = 0.08
SPREADS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


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
    for name, weight in (("E", 1), ("W", 2)):
        m = WeightQNN(5, weight).fit(Xtr, ytr, epochs, seed=sd)
        rec[f"{name} exact"] = m.score(Xte, yte)
        rng = np.random.default_rng(sd)
        perm_u = rng.permutation(np.linspace(-1, 1, 5))  # one assignment per run, shared by all s
        for s in SPREADS:
            g = GAMMA * (1 + s * perm_u)
            rec[f"{name} {s} qang"] = m.score(Xte, yte, gamma=g, qang=True)
            rec[f"{name} {s} raw"] = m.score(Xte, yte, gamma=g, qang=False)
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
    for name in ("E", "W"):
        for s in SPREADS:
            q[f"{name} loss {s}"] = m(f"{name} exact") - m(f"{name} {s} qang")
            q[f"{name} gain {s}"] = m(f"{name} {s} qang") - m(f"{name} {s} raw")
        q[f"{name} exact at s=0"] = float(np.mean([r[f"{name} 0.0 qang"] == r[f"{name} exact"] for r in rows]))
    return q


def verdict(summary):
    s = summary
    return {
        "V1": s["E exact at s=0"][0] == 1.0 and s["W exact at s=0"][0] == 1.0,
        "V2": s["E loss 0.5"][0] < 0.01 and s["W loss 0.5"][0] < 0.01,
        "V3": s["E loss 1.0"][0] < 0.02 and s["W loss 1.0"][0] < 0.02,
        "V4": all(s[f"{n} gain {sp}"][1] > 0 for n in ("E", "W") for sp in SPREADS),
        "V5": all(s[f"W loss {sp}"][0] >= s[f"E loss {sp}"][0] for sp in SPREADS if sp >= 0.3),
    }


def main(seeds=SEEDS, emit_json=False):
    per_seed, all_rows = [], {}
    for sd in seeds:
        rows = run_seed(sd)
        all_rows[sd] = rows
        per_seed.append(quantities(rows))
        print(f"seed {sd} done", flush=True)
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
