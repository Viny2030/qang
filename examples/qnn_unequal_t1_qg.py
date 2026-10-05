"""
Correcting unequal T1 with the filter: train under the calibrated decay
rates and read with qang (§117)

Under equal T1 the filtered readout of a weight-conserving QNN is exactly
noiseless (F4). Under unequal T1 it is not, and the filter alone lost 1.3-1.4
points at a +-100% spread of 1/T1 (§82). Theory: conditioned on no decay
(the shots the filter keeps), the state is A psi / |A psi| with
A = D U_d ... D U_1 and D the diagonal no-jump factor, prod over excited
qubits of sqrt(1 - gamma_q). A common factor in the (1 - gamma_q) cancels in
the normalization, so a model trained with the filter under the device's
decay rates depends only on their *ratios*; the raw readout depends on their
absolute values. If that holds, training with qang under calibrated T1 should
recover the noiseless accuracy, tolerate calibration errors, and be less
hurt than a raw noise-aware model when all T1 drift together.

Setting: WeightQNN, 5 qubits, weight 1 (pairs encoding, qg_Z readout) and
weight 2 (dual encoding, qg_ZZ readout); iris, cancer, wine, digits (3 vs 8);
seeds 1170-1174 (split, initialization and calibration error per seed); 120
epochs. True decay per sublayer gamma_q = 0.08 (1 + u_q), u_q evenly over
[-1, 1] (spread s = 1: 0 to 0.16). Calibration given to training:
gamma_hat_q = gamma_q (1 + e_q), e_q ~ N(0, 0.10) per qubit and seed.
Drift at run time: every T1 shortened by 1.5, (1 - gamma'_q) = (1 - gamma_q)^1.5.

Models (same initialization):
  A  trained noiselessly, read with qang          (the filter alone)
  B  trained noiselessly, read without qang
  C  trained under gamma_hat with qang, read with qang
  C* trained under the exact gamma with qang, read with qang
  D  trained under gamma_hat without qang, read without qang

Pre-registered predictions (committed before the run; code checked on seed
0 with 3 epochs; the invariance below is checked in the tests):
  U1  at the true rates, C is within 0.5 points of the noiseless accuracy on
      average, both weights.
  U2  C is more accurate than A on average, both weights.
  U3  |C - D| <= 1 point on average at the true rates, both weights.
  U4  under the drift, D loses more than C (each against its own accuracy at
      the true rates), both weights.
  U5  C* - C <= 0.5 points on average (a 10% calibration error costs little),
      both weights.

python examples/qnn_unequal_t1_qg.py            # all seeds
python examples/qnn_unequal_t1_qg.py 1170       # one seed, JSON rows

Findings:

FINDINGS_PLACEHOLDER
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qnn_scaling_qg as S  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

N = 5
SEEDS = (1170, 1171, 1172, 1173, 1174)
DATASETS = ("iris", "cancer", "wine", "digits")
MODELS = {1: dict(weight=1), 2: dict(weight=2, encoding="dual", readout="zz")}
GAMMA, SPREAD, CAL_ERROR, DRIFT = 0.08, 1.0, 0.10, 1.5
EPOCHS = 120


def rates(seed):
    true = GAMMA * (1 + SPREAD * np.linspace(-1, 1, N))
    cal = np.clip(true * (1 + CAL_ERROR * np.random.default_rng(seed + 7).normal(size=N)), 0.0, 0.5)
    drift = 1 - (1 - true) ** DRIFT
    return true, cal, drift


def run(seed, datasets=DATASETS, epochs=EPOCHS):
    true, cal, drift = rates(seed)
    rows = []
    for w, kw in MODELS.items():
        for name in datasets:
            Xtr, Xte, ytr, yte = S.prepare(name, N, seed)
            m = WeightQNN(N, **kw)
            p0 = m.fit(Xtr, ytr, epochs, seed=seed).params_
            pc = m.fit(Xtr, ytr, epochs, gamma=cal, qang=True, seed=seed).params_
            ps = m.fit(Xtr, ytr, epochs, gamma=true, qang=True, seed=seed).params_
            pd = m.fit(Xtr, ytr, epochs, gamma=cal, qang=False, seed=seed).params_
            sc = lambda p, g, q: m.score(Xte, yte, p, gamma=g, qang=q)  # noqa: E731
            rows.append({
                "weight": w, "dataset": name, "seed": seed, "noiseless": m.score(Xte, yte, p0),
                "A": sc(p0, true, True), "B": sc(p0, true, False), "C": sc(pc, true, True),
                "C*": sc(ps, true, True), "D": sc(pd, true, False),
                "A drift": sc(p0, drift, True), "B drift": sc(p0, drift, False),
                "C drift": sc(pc, drift, True), "D drift": sc(pd, drift, False),
            })
    return rows


KEYS = ("noiseless", "A", "B", "C", "C*", "D", "A drift", "B drift", "C drift", "D drift")


def summary(rows):
    return {w: {k: float(np.mean([r[k] for r in rows if r["weight"] == w])) for k in KEYS}
            for w in sorted({r["weight"] for r in rows})}


def verdict(rows):
    s = summary(rows)
    return {
        "U1": all(v["noiseless"] - v["C"] <= 0.005 for v in s.values()),
        "U2": all(v["C"] > v["A"] for v in s.values()),
        "U3": all(abs(v["C"] - v["D"]) <= 0.01 for v in s.values()),
        "U4": all(v["D"] - v["D drift"] > v["C"] - v["C drift"] for v in s.values()),
        "U5": all(v["C*"] - v["C"] <= 0.005 for v in s.values()),
    }


def report(rows):
    print("weight " + "".join(f"{k:>10}" for k in KEYS))
    for w, v in summary(rows).items():
        print(f"{w:>6} " + "".join(f"{v[k]:>10.3f}" for k in KEYS))
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
