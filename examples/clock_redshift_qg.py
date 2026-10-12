"""
Two clocks and gravity in qg units: the differential phase from the ellipse (§126)

Two atomic clocks at different heights tick at different rates (gravitational
redshift, fractional shift g h / c^2 = 1.09e-16 per metre). Interrogated by the
same laser, after a Ramsey time t they give qg_Z1 = cos(phi + delta) and
qg_Z2 = cos(phi - delta), with 2 delta = 2 pi nu (g h / c^2) t. When the laser
cannot hold the common phase phi fixed, the pairs fall on an ellipse whose
opening gives delta (correlation spectroscopy; ellipse fitting as in Foster et
al. 2002 and Chwalla et al. 2007). qang.clock_comparison fits it with the
finite-shot bias of the squares removed (qang.statistics' unbiased square).

Simulation: common phase uniform on the circle, ideal contrast, each clock read
with N projective shots per run (binomial projection noise), M runs per
estimate, 400 independent estimates per condition (seed 126).

Pre-registered predictions (committed before the run):
  G1  N = 1000, M = 200: the corrected ellipse estimate of delta is unbiased
      within 2% for delta in {0.15, 0.3, 0.6, 1.0} rad.
  G2  at delta = 0.15 (same data) its standard deviation is at most one third
      of the moment estimator's.
  G3  N = 100, delta = 0.15: without the shot-noise correction the ellipse
      estimate is biased by more than 10%; with it, within 3%.
  G4  strontium clocks (nu = 429.228 THz), t = 1 s, 1 m height difference
      (delta = 0.147 rad), N = 1000, M = 200: mean / standard deviation of the
      estimate is at least 5 (the metre is resolved at 5 sigma).

The model leaves out laser decoherence within the Ramsey time, contrast below
1 and the offsets of real clocks; it measures what the estimator does, not
what a laboratory achieves.

python examples/clock_redshift_qg.py

Findings:

FINDINGS_PLACEHOLDER
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qang import clock_comparison as CC  # noqa: E402

REPS = 400
SEED = 126
NU_SR = 429.228004229873e12


def estimates(delta, n_runs, n_shots, reps=REPS, seed=SEED):
    rng = np.random.default_rng(seed)
    out = {"ellipse": [], "ellipse_raw": [], "moment": []}
    for _ in range(reps):
        x, y = CC.simulate_pairs(delta, n_runs, n_shots, rng)
        out["ellipse"].append(CC.ellipse_estimate(x, y, n_shots)["delta"])
        out["ellipse_raw"].append(CC.ellipse_estimate(x, y, None)["delta"])
        out["moment"].append(CC.moment_estimate(x, y, n_shots)["delta"])
    return {k: np.array(v) for k, v in out.items()}


def run():
    res = {"G1": {}, "G2": None, "G3": None, "G4": None}
    for d in (0.15, 0.3, 0.6, 1.0):
        e = estimates(d, 200, 1000)
        res["G1"][d] = float(e["ellipse"].mean() / d - 1)
        if d == 0.15:
            res["G2"] = (float(e["ellipse"].std()), float(e["moment"].std()))
    e = estimates(0.15, 200, 100)
    res["G3"] = (float(e["ellipse_raw"].mean() / 0.15 - 1), float(e["ellipse"].mean() / 0.15 - 1))
    d_sr = CC.redshift_phase(NU_SR, 1.0, 1.0) / 2
    e = estimates(d_sr, 200, 1000)
    res["G4"] = (d_sr, float(e["ellipse"].mean()), float(e["ellipse"].std()))
    return res


def verdict(res):
    return {
        "G1": all(abs(b) <= 0.02 for b in res["G1"].values()),
        "G2": res["G2"][0] <= res["G2"][1] / 3,
        "G3": abs(res["G3"][0]) > 0.10 and abs(res["G3"][1]) <= 0.03,
        "G4": res["G4"][1] / res["G4"][2] >= 5,
    }


def main():
    res = run()
    print("G1 relative bias (N=1000, M=200):", {d: round(b, 4) for d, b in res["G1"].items()})
    print("G2 std ellipse / moment at delta=0.15:", tuple(round(v, 4) for v in res["G2"]))
    print("G3 bias at N=100 (raw, corrected):", tuple(round(v, 4) for v in res["G3"]))
    d, m, s = res["G4"]
    print(f"G4 Sr, 1 m, t = 1 s: delta = {d:.4f} rad, estimate {m:.4f} +- {s:.4f} ({m / s:.1f} sigma)")
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in verdict(res).items()))
    return res


if __name__ == "__main__":
    main()
