"""
Dephasing with the filter plus ZNE at 8 qubits (§122)

Section 118 at a larger register: WeightQNN with 8 qubits, weight 1 (pairs
encoding, qg_Z) and weight 2 (dual encoding, qg_ZZ, 28 pair correlations);
cancer, wine, digits (3 vs 8), PCA to 7; seeds 1220-1222; 120 epochs,
trained noiselessly; equal T1 gamma = 0.08 plus dephasing p = 0.03 per
sublayer (depth 9). Same readouts as section 118: raw, filter, raw + ZNE,
filter + ZNE (linear from noise scales 1, 2; Richardson from 1, 2, 3), from
exact probabilities and from 1000 shots per circuit and scale; reference:
trained with the filter under the same noise. F6 (the filter turns equal T1
plus dephasing into dephasing alone) is checked at 8 qubits in the tests.

Pre-registered predictions (committed before the run):
  Y1  exact: linear filter + ZNE leaves at most 60% of the filter's feature
      error, both weights.
  Y2  exact: filter + ZNE (linear) is at least as accurate as the filter
      alone, both weights.
  Y3  1000 shots: at weight 2, filter + ZNE (linear) is less accurate than the
      filter alone (shot noise of the extrapolation, as at 5 qubits).
  Y4  training with the filter under the noise is within 1 point of the
      noiseless accuracy, both weights.
  Y5  exact: filter + ZNE (linear) is more accurate than raw + ZNE, both
      weights.

python examples/qnn_dephasing_zne_8q_qg.py            # all seeds
python examples/qnn_dephasing_zne_8q_qg.py 1220       # one seed, JSON rows

Findings (3 datasets x 3 seeds = 9 runs per weight; accuracy, feature error
in brackets; noiseless 0.959 / 0.964 for weight 1 / 2):

  weight 1     raw          filter       raw + ZNE    filter + ZNE  Richardson
    exact      0.788 (.158) 0.925 (.100) 0.914 (.113) 0.955 (.050)  0.955 (.026)
    1000 shots 0.773 (.159) 0.913 (.102) 0.898 (.117) 0.942 (.070)  0.930 (.107)
  weight 2
    exact      0.650 (.387) 0.950 (.105) 0.851 (.210) 0.962 (.058)  0.966 (.035)
    1000 shots 0.652 (.387) 0.915 (.116) 0.833 (.214) 0.879 (.155)  0.720 (.381)
  trained with the filter under the noise: 0.963 (weight 1), 0.961 (weight 2)

  * Y1-Y5 all pass.
  * F6 holds at 8 qubits (tests).
  * Y1: linear filter + ZNE leaves 50% / 56% of the filter's feature error
    (Richardson 26% / 34%), as at 5 qubits.
  * Y2, Y5: with exact probabilities filter + ZNE lands within 0.4 points of
    noiseless and beats raw + ZNE by 4.1 / 11.1 points.
  * Y3: with 1000 shots per scale at weight 2 it falls 3.6 points below the
    filter alone, and Richardson to 0.720: the shot cost grows with the
    number of features to extrapolate (28 correlations at weight 2).
  * Y4: training with the filter under the noise stays within 0.3 points of
    noiseless.
  Verdict. The picture of section 118 holds at 8 qubits: the filter turns
  T1 plus dephasing into dephasing alone, ZNE of the filtered readout removes
  most of the bias with exact probabilities but not at 1000 shots for
  weight 2, and noise-aware training with the filter is the practical
  correction for classifiers.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qnn_dephasing_zne_qg as Z  # noqa: E402
import qnn_scaling_qg as S  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

N = 8
SEEDS = (1220, 1221, 1222)
DATASETS = ("cancer", "wine", "digits")
P_DEPH = 0.03
READOUTS = Z.READOUTS


def run(seed, datasets=DATASETS, epochs=Z.EPOCHS):
    rows = []
    for w, kw in Z.MODELS.items():
        for name in datasets:
            Xtr, Xte, ytr, yte = S.prepare(name, N, seed)
            m = WeightQNN(N, **kw)
            p0 = m.fit(Xtr, ytr, epochs, seed=seed).params_
            th, psi = p0[: m.n_theta], m.encode(Xte)
            f_exact = m.qg_z(m.probs(th, psi), True)
            row = {"weight": w, "dataset": name, "seed": seed, "noiseless": Z.accuracy(p0, m, f_exact, yte)}
            pa = m.fit(Xtr, ytr, epochs, gamma=Z.GAMMA, dephasing=P_DEPH, qang=True, seed=seed).params_
            row["noise-aware filter"] = m.score(Xte, yte, pa, gamma=Z.GAMMA, dephasing=P_DEPH, qang=True)
            for label, shots in (("exact", None), ("shots", Z.SHOTS)):
                F = Z.readout_features(m, th, psi, P_DEPH, shots, seed)
                for k in READOUTS:
                    row[f"{label} {k} acc"] = Z.accuracy(p0, m, F[k], yte)
                    row[f"{label} {k} err"] = float(np.mean(np.abs(F[k] - f_exact)))
            rows.append(row)
    return rows


def summary(rows):
    keys = [k for k in rows[0] if k not in ("weight", "dataset", "seed")]
    return {w: {k: float(np.mean([r[k] for r in rows if r["weight"] == w])) for k in keys}
            for w in sorted({r["weight"] for r in rows})}


def verdict(rows):
    s = summary(rows)
    return {
        "Y1": all(v["exact filter + ZNE err"] <= 0.6 * v["exact filter err"] for v in s.values()),
        "Y2": all(v["exact filter + ZNE acc"] >= v["exact filter acc"] for v in s.values()),
        "Y3": s[2]["shots filter + ZNE acc"] < s[2]["shots filter acc"],
        "Y4": all(v["noiseless"] - v["noise-aware filter"] <= 0.01 for v in s.values()),
        "Y5": all(v["exact filter + ZNE acc"] > v["exact raw + ZNE acc"] for v in s.values()),
    }


def main(seeds=SEEDS, emit_json=False):
    rows = []
    for sd in seeds:
        r = run(sd)
        rows += r
        print(f"seed {sd} done", flush=True)
        if emit_json:
            print(json.dumps(r), flush=True)
    for w, v in summary(rows).items():
        print(f"weight {w}: noiseless {v['noiseless']:.3f}, noise-aware filter {v['noise-aware filter']:.3f}")
        for lab in ("exact", "shots"):
            print(f"  {lab:<6}" + "".join(f"  {k}: {v[f'{lab} {k} acc']:.3f} ({v[f'{lab} {k} err']:.3f})" for k in READOUTS))
    v = verdict(rows)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return rows, v


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(seeds=tuple(args) if args else SEEDS, emit_json=bool(args))
