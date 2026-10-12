"""
Fitting a weighted-QSP polynomial from shots: the qg (arcsine) scale against
the linear scale (§134)

Weighted QSP (Sarkar et al., arXiv:2609.21567) learns its phases from a
Hadamard test: at each signal x the ancilla gives p0 = (1 + Re P(x)) / 2, so
Re P(x) is the ancilla's qg_Z. With N shots the estimate 2 k0/N - 1 has
variance (1 - Re P^2)/N, which changes with x: small where Re P is near +-1,
largest where it is near 0. Fitting it by plain least squares weighs every x
alike. In the qang angle theta = 2 arccos(sqrt(p0)) (the arcsine transform,
qang.proportions) the variance is ~1/N at every x, so plain least squares in
that scale weighs the points by their precision.

Set-up (fixed before any fit):
  target   WQSP with weights (1, 2, 1, 2) and phases (0.3, -0.7, 1.1, 0.4,
           -0.2); Re P runs from -0.995 to 0.62 over x in [-1, 1].
  data     41 equally spaced x in [-1, 1]; Hadamard-test counts k0 ~
           Binomial(N, p0(x)); 200 independent data sets (seeds 0-199).
  fits     the same WQSP structure (5 phases), scipy least_squares from 8
           seeded random starts, identical for both scales; scale "linear"
           fits Re P, scale "arcsine" fits theta.
  error    RMS of Re P_fit - Re P_true over 401 x in [-1, 1].

Pre-registered predictions (committed before any fit is run):
  Q1  at N = 100 shots per x, the mean error of the arcsine fit is lower
      than that of the linear fit.
  Q2  at N = 100 the arcsine fit's mean error is at most 0.9 times the
      linear fit's.
  Q3  at N = 1000 the arcsine fit's mean error is still lower than the
      linear fit's.
  Q4  (variance stabilization, no fit) at N = 100, at every x where
      N p0 (1 - p0) >= 5, the standard deviation of theta over the 200 data
      sets is within 20% of 1/sqrt(N), while that of 2 k0/N - 1 varies by
      more than a factor of 2 across those x.

python examples/qsp_arcsine_fit_qg.py

Findings:

Mean RMS error of Re P over 200 data sets:

  shots per x   linear fit   arcsine fit   arcsine better in
  100           0.0228       0.0220        125 / 200
  1000          0.00742      0.00708       147 / 200

  Q1 PASS: the arcsine fit is better at N = 100 (by 3.6%).
  Q2 FAIL: it is 0.96 times the linear error, not 0.9 or less.
  Q3 PASS: it stays better at N = 1000 (by 4.6%, in 74% of the data sets).
  Q4 FAIL: the theta scale is stabilized (sd 0.90 to 1.18 times 1/sqrt N
     at the 36 eligible x, within 20%), but over those x the linear sd
     varies by 1.93, short of the predicted factor 2: the x where it
     varies most (Re P near -1) are the ones excluded by N p0 (1 - p0) >= 5.
  * Fitting in the qg (arcsine) scale is consistently, but only slightly,
    better: the gain comes from the few x near the pole, where the linear
    scale is most precise and plain least squares underuses it.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import qsp  # noqa: E402

WEIGHTS = (1, 2, 1, 2)
PHASES = (0.3, -0.7, 1.1, 0.4, -0.2)
X = np.linspace(-1, 1, 41)
XF = np.linspace(-1, 1, 401)
REPS = 200
STARTS = 8
OUT = os.path.join(HERE, "data", "qsp_arcsine_fit.json")


def errors(shots, reps=REPS, starts=STARTS):
    true = qsp.wqsp_poly(XF, PHASES, WEIGHTS).real
    out = {"linear": [], "arcsine": []}
    for r in range(reps):
        k0 = qsp.sample_hadamard_test(X, PHASES, WEIGHTS, shots=shots, rng=r)
        for scale in out:
            ph, _ = qsp.fit_phases(X, k0, shots, len(PHASES), WEIGHTS, scale=scale, starts=starts, seed=1000 + r)
            out[scale].append(float(np.sqrt(np.mean((qsp.wqsp_poly(XF, ph, WEIGHTS).real - true) ** 2))))
    return out


def stabilization(shots=100, reps=REPS):
    p0 = qsp.hadamard_test_p0(X, PHASES, WEIGHTS)
    k = np.array([qsp.sample_hadamard_test(X, PHASES, WEIGHTS, shots=shots, rng=r) for r in range(reps)])
    th = 2 * np.arccos(np.sqrt(k / shots))
    lin = 2 * k / shots - 1
    ok = shots * p0 * (1 - p0) >= 5
    sd_th, sd_lin = th.std(axis=0)[ok], lin.std(axis=0)[ok]
    return {"points": int(ok.sum()), "theta sd / (1/sqrt N) min, max": [float(sd_th.min() * np.sqrt(shots)),
                                                                        float(sd_th.max() * np.sqrt(shots))],
            "linear sd max / min": float(sd_lin.max() / sd_lin.min())}


def main():
    res = {}
    for n in (100, 1000):
        e = errors(n)
        res[n] = {"linear": float(np.mean(e["linear"])), "arcsine": float(np.mean(e["arcsine"])),
                  "arcsine better in": int(np.sum(np.array(e["arcsine"]) < np.array(e["linear"])))}
        print(json.dumps({"shots": n, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in res[n].items()}}),
              flush=True)
    st = stabilization()
    print(json.dumps(st))
    lo, hi = st["theta sd / (1/sqrt N) min, max"]
    v = {"Q1": res[100]["arcsine"] < res[100]["linear"],
         "Q2": res[100]["arcsine"] <= 0.9 * res[100]["linear"],
         "Q3": res[1000]["arcsine"] < res[1000]["linear"],
         "Q4": lo >= 0.8 and hi <= 1.2 and st["linear sd max / min"] > 2}
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"errors": res, "stabilization": st, "verdict": v}, fh, indent=1)
    return res, st, v


if __name__ == "__main__":
    main()
