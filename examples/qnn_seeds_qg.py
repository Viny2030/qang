"""
Replicating the QNN results over seeds, with and without qang (§80)

§75-§79 ran each study with one seed. One test sample is 2-3 points, and
§78-§79 already disagreed on one effect (the unequal-T1 loss of the filter:
1.5 points in §78, 0.2-0.6 in §79). Before writing up, the key comparisons
are repeated over 5 seeds (80-84) x 3 splits x 4 datasets = 60 runs per
model, with 95% confidence intervals across seeds (seed means over
datasets and splits, t distribution with 4 degrees of freedom).

Models: E (weight 1, §76) and W (weight 2, §79), same architecture, data,
epochs and simulator as §79. Each run trains: clean (no noise), aware under
equal T1 with qang and without qang, aware under T1 + dephasing with qang.
"qang" = the readout keeps only the shots in the input's weight sector
(qang.sectors.filter_distribution); "without" = the raw readout.

Paired quantities (per seed: mean over datasets and splits):
  G_m   gain of qang, trained clean, under equal T1:  clean qang - clean raw
  A_m   aware training under T1:                      aware qang - aware raw
  H_m   loss under unequal T1 (§78 S2):               exact - clean qang (H)
  Dp_m  dephasing, aware vs clean (with qang):        aware qang (D) - clean qang (D)
  dW    weight 2 vs weight 1 (§79 P3):                W exact - E exact

Pre-registered predictions (written and committed before the run; code
debugged on seed 1 with few epochs):
  T1  qang helps when training is noise-free, more at weight 2: G_E > 0 and
      G_W > 0 with the 95% CI above 0, and G_W - G_E with the CI above 0.
  T2  with noise-aware training qang makes no measurable difference:
      |mean A_E| < 1 point and |mean A_W| < 1 point.
  T3  the unequal-T1 loss of the filter is small: mean H_E < 1 point and
      mean H_W < 1 point.
  T4  dephasing needs noise-aware training: Dp_W > 0 with the CI above 0.
  T5  the §79 accuracy cost of weight 2 replicates: dW < 0 with the CI
      below 0.

Uses the installed library (pip install qang) through the §79 module.
Needs scikit-learn and scipy. OMP_NUM_THREADS=1 python
examples/qnn_seeds_qg.py [seeds]  (default 80 81 82 83 84; with arguments,
prints JSON for those seeds only).

Findings (python examples/qnn_seeds_qg.py):

5 seeds x 3 splits x 4 datasets = 60 runs per model. Mean accuracy over the
60 runs, then the paired quantities as mean and 95% CI across the 5 seeds:

                        E (weight 1)          W (weight 2)
  exact (no noise)      0.952                 0.922
  T1, trained clean     qang 0.952 / without 0.928    qang 0.922 / without 0.757
  T1, trained aware     qang 0.952 / without 0.951    qang 0.922 / without 0.924
  unequal T1, clean     qang 0.947 / without 0.921    qang 0.916 / without 0.732
  dephasing, clean      qang 0.918 / without 0.792    qang 0.882 / without 0.656
  dephasing, aware      qang 0.950                    qang 0.926

  G_E  +0.025 [+0.008, +0.041]     G_W  +0.165 [+0.118, +0.212]
  G_W - G_E                        +0.140 [+0.083, +0.197]
  A_E  +0.001 [-0.006, +0.009]     A_W  -0.001 [-0.008, +0.005]
  H_E  +0.005 [-0.002, +0.012]     H_W  +0.007 [-0.000, +0.013]
  Dp_E +0.032 [+0.011, +0.053]     Dp_W +0.045 [+0.034, +0.056]
  dW   -0.030 [-0.040, -0.020]
  (not predicted) gain of qang under dephasing, trained clean:
  D_E  +0.126 [+0.091, +0.162]     D_W  +0.226 [+0.191, +0.262]

  * All five predictions pass.
  * T1: the effect of qang when training is noise-free is robust: +2.5
    points at weight 1 and +16.5 at weight 2, both CIs well above 0, and
    the difference between weights (+14.0) too. qang is ahead or tied on 50
    of 60 runs at weight 1 and 56 of 60 at weight 2.
  * T2: with noise-aware training, with and without qang are the same
    (differences of 0.1 points, CIs within +-0.9 points).
  * T3: the unequal-T1 loss of the filter is small: 0.5 and 0.7 points.
    The 1.5 points of §78 were on the high side of the seed spread.
  * T4: dephasing needs noise-aware training: +3.2 (E) and +4.5 (W).
  * T5: weight 2 costs 3.0 points of accuracy (§79 P3 replicated).
  * Not predicted: under dephasing, a model trained clean still gains 12.6
    (E) and 22.6 (W) points from qang. The filter does not correct
    dephasing, but it still removes the T1 part of the combined noise.
  Verdict. With 60 runs per model the picture of §77-§79 holds. qang makes
  noise-free training valid under T1, exactly, with a gain that grows with
  the weight; with noise-aware training it adds nothing measurable; it does
  not replace a noise model for dephasing; and the weight-2 model is less
  accurate than the weight-1 one. No quantum advantage is claimed.
  Limitations: simulated noise, one architecture per weight, small
  datasets.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
import qnn_noise_aware_qg as N77  # noqa: E402
import qnn_weight2_qg as W  # noqa: E402

SEEDS = (80, 81, 82, 83, 84)
SPLITS = 3
EPOCHS = 120


def run_split(name, seed, s, epochs=EPOCHS, data=None):
    Xa, ya, use_pca = data if data is not None else Q.load(name, np.random.default_rng(seed))
    Xtr, Xte, ytr, yte = Q.split_prepare(Xa, ya, use_pca, seed * 100 + s)
    sd = seed * 1000 + 10 * s
    rec = {}
    for m in W.MODELS:
        psi = W.encode(m, Xtr)

        def train(ro):
            return N77.adam_train(W.E.NP, W.N, ro, ytr, np.random.default_rng(sd + 2), epochs)

        clean = train(W.readout(m, psi))
        T1, H, D = W.CONDITIONS["T1"], W.CONDITIONS["H"], W.CONDITIONS["D"]
        aq = train(W.readout(m, psi, T1, qang=True))
        ar = train(W.readout(m, psi, T1, qang=False))
        dq = train(W.readout(m, psi, D, qang=True))
        ev = lambda p, cond=None, q=False: W.evaluate(m, p, Xte, yte, cond, q)[0]  # noqa: E731
        rec[f"{m} exact"] = ev(clean)
        rec[f"{m} T1 clean qang"] = ev(clean, T1, True)
        rec[f"{m} T1 clean raw"] = ev(clean, T1)
        rec[f"{m} T1 aware qang"] = ev(aq, T1, True)
        rec[f"{m} T1 aware raw"] = ev(ar, T1)
        rec[f"{m} H clean qang"] = ev(clean, H, True)
        rec[f"{m} H clean raw"] = ev(clean, H)
        rec[f"{m} D clean qang"] = ev(clean, D, True)
        rec[f"{m} D clean raw"] = ev(clean, D)
        rec[f"{m} D aware qang"] = ev(dq, D, True)
    return rec


def run_seed(seed, splits=SPLITS, epochs=EPOCHS, datasets=None):
    rows = []
    for name in datasets or Q.DATASETS:
        data = Q.load(name, np.random.default_rng(seed))
        for s in range(splits):
            r = run_split(name, seed, s, epochs, data)
            r["dataset"] = name
            rows.append(r)
    return rows


def quantities(rows):
    """Per-seed paired quantities from the rows of one seed."""
    m = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    q = {}
    for mod in W.MODELS:
        q[f"G_{mod}"] = m(f"{mod} T1 clean qang") - m(f"{mod} T1 clean raw")
        q[f"A_{mod}"] = m(f"{mod} T1 aware qang") - m(f"{mod} T1 aware raw")
        q[f"H_{mod}"] = m(f"{mod} exact") - m(f"{mod} H clean qang")
        q[f"Dp_{mod}"] = m(f"{mod} D aware qang") - m(f"{mod} D clean qang")
        q[f"D_{mod}"] = m(f"{mod} D clean qang") - m(f"{mod} D clean raw")
    q["G_W-G_E"] = q["G_W"] - q["G_E"]
    q["dW"] = m("W exact") - m("E exact")
    return q


def ci95(values):
    from scipy import stats

    v = np.asarray(values, float)
    mean = float(v.mean())
    if len(v) < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return mean, mean - half, mean + half


def summarize(per_seed):
    keys = per_seed[0].keys()
    return {k: ci95([q[k] for q in per_seed]) for k in keys}


def verdict(summary):
    s = summary
    return {
        "T1": s["G_E"][1] > 0 and s["G_W"][1] > 0 and s["G_W-G_E"][1] > 0,
        "T2": abs(s["A_E"][0]) < 0.01 and abs(s["A_W"][0]) < 0.01,
        "T3": s["H_E"][0] < 0.01 and s["H_W"][0] < 0.01,
        "T4": s["Dp_W"][1] > 0,
        "T5": s["dW"][2] < 0,
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
    summ = summarize(per_seed)
    for k, (mu, lo, hi) in summ.items():
        print(f"{k}: {mu:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    v = verdict(summ)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return summ, v


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(seeds=tuple(args) if args else SEEDS, emit_json=bool(args))
