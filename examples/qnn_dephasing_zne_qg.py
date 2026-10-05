"""
Dephasing: the filter plus zero-noise extrapolation (§118)

Dephasing conserves the Hamming weight, so the qg filter cannot see it; it was
the one noise left uncorrected in the QML note. Theory (F6, tested): with
equal T1 and dephasing, the filtered readout equals the readout of the same
circuit with dephasing alone, exactly (the no-jump factor is a scalar on the
sector and commutes with the diagonal dephasing channel). The filter thus
removes T1 and leaves a single noise parameter, which is what zero-noise
extrapolation (ZNE) needs.

ZNE here: run at noise scale lambda = 1, 2, 3 (as gate folding would:
1 - gamma -> (1 - gamma)^lambda, 1 - 2p -> (1 - 2p)^lambda), compute the
readout features (qg_Z, and qg_ZZ for weight 2) at each scale, extrapolate
every feature to lambda = 0 (linear from 1, 2: 2 f1 - f2; Richardson from
1, 2, 3: 3 f1 - 3 f2 + f3), clip to [-1, 1], and apply the trained linear
head. Readouts: raw, filter, raw + ZNE, filter + ZNE (linear and
Richardson). Reference: a model trained with the filter under the same noise
(noise-aware).

Setting: WeightQNN, 5 qubits, weight 1 (pairs encoding, qg_Z) and weight 2
(dual encoding, qg_ZZ); iris, cancer, wine, digits (3 vs 8); seeds
1180-1184; 120 epochs, trained noiselessly. Noise per sublayer: equal T1
gamma = 0.08 and dephasing p = 0.03 or 0.06. Readings from exact
probabilities, and from 1000 shots per circuit and scale (ZNE uses 2 or 3
times the shots of a single reading). Feature error: mean |f - f_noiseless|
over test inputs and features.

Pre-registered predictions (committed before the run; code checked on seed 0
with 3 epochs, noiseless only; F6 checked in the tests):
  Z1  exact probabilities, p = 0.03: the feature error of filter + ZNE
      (linear) is at most half that of the filter alone, both weights.
  Z2  exact probabilities, p = 0.03: filter + ZNE (linear) has a lower
      feature error than raw + ZNE (linear), both weights.
  Z3  exact probabilities, p = 0.03: filter + ZNE (linear) is within 1 point
      of the noiseless accuracy on average, both weights.
  Z4  1000 shots, p = 0.03: filter + ZNE (linear) is at least as accurate as
      the filter alone on average, both weights.
  Z5  exact probabilities, p = 0.06, weight 2: filter + ZNE (linear) is at
      least 2 points more accurate than the filter alone.

python examples/qnn_dephasing_zne_qg.py            # all seeds
python examples/qnn_dephasing_zne_qg.py 1180       # one seed, JSON rows

Findings (4 datasets x 5 seeds = 20 runs per weight; accuracy, feature
error in brackets; noiseless 0.952 / 0.956 for weight 1 / 2):

  weight 1, p = 0.03   raw          filter       raw + ZNE    filter + ZNE  filter + ZNE (Richardson)
    exact              0.815 (.257) 0.933 (.168) 0.915 (.187) 0.949 (.084)  0.953 (.045)
    1000 shots         0.817 (.258) 0.932 (.171) 0.909 (.190) 0.945 (.107)  0.904 (.145)
  weight 1, p = 0.06
    exact              0.709 (.291) 0.837 (.257) 0.814 (.250) 0.918 (.190)  0.938 (.145)
    1000 shots         0.710 (.291) 0.819 (.257) 0.808 (.252) 0.881 (.201)  0.851 (.209)
  weight 2, p = 0.03
    exact              0.686 (.520) 0.919 (.192) 0.844 (.319) 0.943 (.111)  0.946 (.068)
    1000 shots         0.679 (.520) 0.893 (.197) 0.841 (.321) 0.864 (.183)  0.710 (.388)
  weight 2, p = 0.06
    exact              0.610 (.531) 0.775 (.279) 0.703 (.355) 0.881 (.222)  0.902 (.182)
    1000 shots         0.599 (.532) 0.749 (.284) 0.700 (.358) 0.778 (.264)  0.684 (.434)
  trained with the filter under the same noise: 0.953 / 0.951 (weight 1,
  p = 0.03 / 0.06) and 0.956 / 0.952 (weight 2).

  * Z2, Z5 pass; Z1, Z3, Z4 fail.
  * F6 holds exactly (tests): with equal T1, the filtered readout equals the
    dephasing-only readout.
  * Z1 fails narrowly: linear filter + ZNE leaves 50.3% (weight 1) and 57.5%
    (weight 2) of the filter's feature error, not at most 50%; Richardson
    leaves 27% and 35%.
  * Z2: filter + ZNE is 3.4 / 9.9 points more accurate than raw + ZNE and has
    a lower feature error at both weights; ZNE on the raw readout has to
    extrapolate T1 and dephasing together.
  * Z3 fails at weight 2: 1.3 points below noiseless (weight 1: 0.3).
  * Z4 fails at weight 2: with 1000 shots per scale, linear filter + ZNE is
    2.9 points below the filter alone (0.864 against 0.893); at weight 1 it
    is 1.2 points ahead. Richardson amplifies shot noise further and loses at
    both weights with shots (down to 0.710 at weight 2).
  * Z5: at p = 0.06, weight 2, filter + ZNE gains 10.6 points over the filter
    alone with exact probabilities (0.881 against 0.775).
  * Not predicted: training with the filter under the calibrated noise
    recovers the noiseless accuracy at both dephasing levels (0.951-0.956),
    better than any extrapolation and with no extra circuits.
  Verdict. Removing T1 with the filter leaves dephasing as the only noise,
  and zero-noise extrapolation of the filtered features then cuts the
  feature error by half (linear) to two thirds (Richardson) and recovers most
  of the lost accuracy with exact probabilities. With 1000 shots per scale
  the extrapolation's variance, larger at weight 2 where fewer shots are
  kept, cancels the gain. For classification, noise-aware training with the
  filter is the better correction for dephasing; ZNE of the filtered
  readout is the tool for estimating observables, given enough shots.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qnn_scaling_qg as S  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

N = 5
SEEDS = (1180, 1181, 1182, 1183, 1184)
DATASETS = ("iris", "cancer", "wine", "digits")
MODELS = {1: dict(weight=1), 2: dict(weight=2, encoding="dual", readout="zz")}
GAMMA, DEPHASING, SHOTS, EPOCHS = 0.08, (0.03, 0.06), 1000, 120
READOUTS = ("raw", "filter", "raw + ZNE", "filter + ZNE", "filter + ZNE (Richardson)")


def scaled(gamma, p, lam):
    """Noise at scale lam, as gate folding would give it."""
    return 1 - (1 - gamma) ** lam, (1 - (1 - 2 * p) ** lam) / 2


def extrapolate(f, order=1):
    """Zero-noise value from features at lambda = 1, 2 (order 1) or 1, 2, 3 (order 2)."""
    out = 2 * f[0] - f[1] if order == 1 else 3 * f[0] - 3 * f[1] + f[2]
    return np.clip(out, -1.0, 1.0)


def features(m, th, psi, gamma, p, qang, shots=None, rng=None):
    pr = m.probs(th, psi, gamma, p)
    if shots:
        pr = np.array([rng.multinomial(shots, q / q.sum()) / shots for q in pr])
    return m.qg_z(pr, qang)


def readout_features(m, th, psi, p, shots=None, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for qang, tag in ((False, "raw"), (True, "filter")):
        f = [features(m, th, psi, *scaled(GAMMA, p, lam), qang, shots, rng) for lam in (1, 2, 3)]
        out[tag] = f[0]
        out[f"{tag} + ZNE"] = extrapolate(f, 1)
        if qang:
            out["filter + ZNE (Richardson)"] = extrapolate(f, 2)
    return out


def accuracy(params, m, F, y):
    return float(np.mean(((F @ params[m.n_theta:-1] + params[-1]) > 0).astype(int) == np.asarray(y)))


def run(seed, datasets=DATASETS, epochs=EPOCHS):
    rows = []
    for w, kw in MODELS.items():
        for name in datasets:
            Xtr, Xte, ytr, yte = S.prepare(name, N, seed)
            m = WeightQNN(N, **kw)
            p0 = m.fit(Xtr, ytr, epochs, seed=seed).params_
            th, psi = p0[: m.n_theta], m.encode(Xte)
            f_exact = m.qg_z(m.probs(th, psi), True)
            row = {"weight": w, "dataset": name, "seed": seed, "noiseless": accuracy(p0, m, f_exact, yte)}
            for p in DEPHASING:
                pa = m.fit(Xtr, ytr, epochs, gamma=GAMMA, dephasing=p, qang=True, seed=seed).params_
                row[f"p={p} noise-aware filter"] = m.score(Xte, yte, pa, gamma=GAMMA, dephasing=p, qang=True)
                for label, shots in (("exact", None), ("shots", SHOTS)):
                    F = readout_features(m, th, psi, p, shots, seed)
                    for k in READOUTS:
                        row[f"p={p} {label} {k} acc"] = accuracy(p0, m, F[k], yte)
                        row[f"p={p} {label} {k} err"] = float(np.mean(np.abs(F[k] - f_exact)))
            rows.append(row)
    return rows


def summary(rows):
    keys = [k for k in rows[0] if k not in ("weight", "dataset", "seed")]
    return {w: {k: float(np.mean([r[k] for r in rows if r["weight"] == w])) for k in keys}
            for w in sorted({r["weight"] for r in rows})}


def verdict(rows):
    s = summary(rows)
    g = lambda v, p, lab, k, x: v[f"p={p} {lab} {k} {x}"]  # noqa: E731
    return {
        "Z1": all(g(v, 0.03, "exact", "filter + ZNE", "err") <= 0.5 * g(v, 0.03, "exact", "filter", "err") for v in s.values()),
        "Z2": all(g(v, 0.03, "exact", "filter + ZNE", "err") < g(v, 0.03, "exact", "raw + ZNE", "err") for v in s.values()),
        "Z3": all(v["noiseless"] - g(v, 0.03, "exact", "filter + ZNE", "acc") <= 0.01 for v in s.values()),
        "Z4": all(g(v, 0.03, "shots", "filter + ZNE", "acc") >= g(v, 0.03, "shots", "filter", "acc") for v in s.values()),
        "Z5": g(s[2], 0.06, "exact", "filter + ZNE", "acc") - g(s[2], 0.06, "exact", "filter", "acc") >= 0.02,
    }


def report(rows):
    for w, v in summary(rows).items():
        print(f"weight {w}: noiseless {v['noiseless']:.3f}")
        for p in DEPHASING:
            print(f"  p={p}: noise-aware filter {v[f'p={p} noise-aware filter']:.3f}")
            for lab in ("exact", "shots"):
                print(f"    {lab:<6}" + "".join(f"  {k}: {v[f'p={p} {lab} {k} acc']:.3f} ({v[f'p={p} {lab} {k} err']:.3f})" for k in READOUTS))
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
