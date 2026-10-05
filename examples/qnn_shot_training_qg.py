"""
Training the QNN from shots, with and without qang (§109)

Every QNN training so far (§75-§108) used exact outcome probabilities. On a
device the features come from a finite number of shots, and the qg filter
keeps only part of them: under equal T1 it removes the decay bias but raises
the shot noise of every feature, including the ones the optimizer uses to
estimate gradients. This study trains from shots.

Model: qang.qml.WeightQNN, 5 qubits, weight 1, qg_Z readout (model E), equal
T1 gamma = 0.08 per qubit per sublayer (kept fraction (1 - 0.08)^9 = 0.47).
Training: Adam (lr 0.05) for 200 epochs on the full training set; the 15
circuit angles get SPSA gradients (one random +-1 perturbation of size 0.15
per epoch, two extra evaluations of the training set), the classical linear
head gets its exact gradient from the sampled features. Every evaluation of
an input uses S fresh shots (S = 100 or 1000), read with qang (filter) or
without it (all shots). Same initialization for every condition of a run.
Test: 1000 shots per test input under the same T1, with the readout the
model was trained with. Reference: the same model trained on exact noiseless
probabilities (WeightQNN.fit) and run with qang (§80: noiseless training is
exactly valid under equal T1 with the filter).

Protocol: iris, breast cancer, wine, digits x seeds 1090-1094 (split and
initialization per seed) = 20 runs per condition; 95% CIs across seeds.
SPSA settings were fixed on a noiseless check (no T1, iris, one seed) before
these predictions.

Pre-registered predictions (committed before the noisy runs):
  S1  S = 1000: |mean accuracy trained with qang - trained without qang| is
      below 1 point.
  S2  S = 100: trained with qang is worse than trained without qang by more
      than 1 point (mean): with half the shots discarded, the filter's shot
      cost shows in training.
  S3  the noiselessly trained model run with qang is at least as accurate as
      the best shot-trained model at S = 1000, minus 1 point (mean): training
      on a simulator stays the better route.
  S4  S = 1000, trained with qang: within 3 points of the noiselessly trained
      model (mean).

OMP_NUM_THREADS=1 python examples/qnn_shot_training_qg.py [seeds]

Findings:

Mean test accuracy over 20 runs (4 datasets x 5 seeds), equal T1 gamma =
0.08, test with 1000 shots and the training readout:

  training                              with qang        without qang
  exact noiseless probabilities         0.945            -
  from shots, S = 1000 per evaluation   0.939            0.929
  from shots, S = 100 per evaluation    0.930            0.915

  Paired differences (points, 95% CI across seeds):
  S = 1000, qang - raw   +1.0 [-1.6, +3.7]    S = 100, qang - raw   +1.5 [-1.6, +4.6]
  exact - best S=1000    +0.5 [-1.0, +1.9]    exact - qang S=1000   +0.6 [-0.7, +2.0]

  * S3, S4 pass; S1, S2 fail.
  * S2 FAILS, and that is the finding: the filter's shot cost did not show in
    training. With 100 shots per evaluation, of which the filter keeps 47%,
    the model trained with qang was 1.5 points ahead of the one trained on all
    shots, not behind. The noisier but unbiased features trained as well as
    or better than the biased ones.
  * S1 fails narrowly: at 1000 shots qang was 1.0 points ahead (the
    prediction was |difference| < 1); the CI includes 0 in both cases.
  * S3, S4: training on exact noiseless probabilities and running with the
    filter (0.945) stays the best route, within 0.6 points of training from
    shots with qang, at no shot cost for training.
  Verdict. Training from shots under T1 does not change the picture of
  §80: with qang the shot-trained model is as good as or slightly better than
  without it at both shot levels, and the cheapest route, noiseless training
  plus the filter at run time, is as good as any.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

SEEDS = (1090, 1091, 1092, 1093, 1094)
GAMMA = 0.08
EPOCHS, LR, C_SPSA = 200, 0.05, 0.15
SHOTS_TRAIN = (100, 1000)
SHOTS_TEST = 1000


def sampled_features(m, th, psi, gamma, shots, qang, rng):
    pr = m.probs(th, psi, gamma)
    pr = np.array([rng.multinomial(shots, p / p.sum()) / shots for p in pr])
    return m.qg_z(pr, qang)


def train_shots(m, X, y, shots, qang, init, rng, gamma=GAMMA, epochs=EPOCHS):
    psi = m.encode(X)
    y = np.asarray(y, float)
    nt = m.n_theta
    p = init.copy()
    mo, v = np.zeros_like(p), np.zeros_like(p)
    for t in range(1, epochs + 1):
        th, c, b = p[:nt], p[nt:-1], p[-1]
        R = sampled_features(m, th, psi, gamma, shots, qang, rng)
        z = R @ c + b
        dz = (1 / (1 + np.exp(-z)) - y) / len(y)
        g = np.zeros_like(p)
        g[nt:-1] = dz @ R
        g[-1] = dz.sum()
        delta = rng.choice([-1.0, 1.0], nt)

        def loss(th2):
            R2 = sampled_features(m, th2, psi, gamma, shots, qang, rng)
            z2 = R2 @ c + b
            return float(np.mean(np.logaddexp(0, z2) - y * z2))

        g[:nt] = (loss(th + C_SPSA * delta) - loss(th - C_SPSA * delta)) / (2 * C_SPSA) * delta
        mo = 0.9 * mo + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        p = p - LR * (mo / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return p


def test_accuracy(m, params, X, y, qang, rng, gamma=GAMMA, shots=SHOTS_TEST):
    R = sampled_features(m, params[: m.n_theta], m.encode(X), gamma, shots, qang, rng)
    return float(np.mean(((R @ params[m.n_theta:-1] + params[-1]) > 0).astype(int) == np.asarray(y)))


def run(seed, datasets=None, epochs=EPOCHS, gamma=GAMMA):
    rows = []
    for name in datasets or Q.DATASETS:
        Xa, ya, use_pca = Q.load(name, np.random.default_rng(seed))
        Xtr, Xte, ytr, yte = Q.split_prepare(Xa, ya, use_pca, seed * 100)
        m = WeightQNN(5, 1)
        r0 = np.random.default_rng(seed * 10 + 1)
        init = np.concatenate([r0.uniform(-np.pi, np.pi, m.n_theta), r0.normal(0, 0.5, m.n_head), [0.0]])
        rec = {"dataset": name}
        exact = m.fit(Xtr, ytr, 120, seed=seed * 10 + 1).params_
        rec["exact, run with qang"] = test_accuracy(m, exact, Xte, yte, True, np.random.default_rng(seed + 7), gamma)
        for S in SHOTS_TRAIN:
            for label, q in (("qang", True), ("raw", False)):
                p = train_shots(m, Xtr, ytr, S, q, init, np.random.default_rng(seed * 1000 + S), gamma, epochs)
                rec[f"S={S} {label}"] = test_accuracy(m, p, Xte, yte, q, np.random.default_rng(seed + 7), gamma)
        rows.append(rec)
    return rows


def quantities(rows):
    a = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    q = {k: a(k) for k in rows[0] if k != "dataset"}
    q["S=1000 qang-raw"] = q["S=1000 qang"] - q["S=1000 raw"]
    q["S=100 qang-raw"] = q["S=100 qang"] - q["S=100 raw"]
    q["exact-best1000"] = q["exact, run with qang"] - max(q["S=1000 qang"], q["S=1000 raw"])
    q["exact-qang1000"] = q["exact, run with qang"] - q["S=1000 qang"]
    return q


def ci95(values):
    from scipy import stats

    v = np.asarray(values, float)
    h = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return float(v.mean()), float(v.mean()) - h, float(v.mean()) + h


def verdict(s):
    return {
        "S1": abs(s["S=1000 qang-raw"][0]) < 0.01,
        "S2": s["S=100 qang-raw"][0] < -0.01,
        "S3": s["exact-best1000"][0] >= -0.01,
        "S4": s["exact-qang1000"][0] <= 0.03,
    }


def main(seeds=SEEDS, emit_json=False):
    per_seed, all_rows = [], {}
    for sd in seeds:
        rows = run(sd)
        all_rows[sd] = rows
        per_seed.append(quantities(rows))
        print(f"seed {sd}: " + ", ".join(f"{k} {v:+.3f}" for k, v in per_seed[-1].items()), flush=True)
    if emit_json:
        print(json.dumps(all_rows))
        return all_rows, None
    s = {k: ci95([q[k] for q in per_seed]) for k in per_seed[0]}
    for k, (mu, lo, hi) in s.items():
        print(f"{k}: {mu:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    v = verdict(s)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return s, v


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(seeds=tuple(args) if args else SEEDS, emit_json=bool(args))
