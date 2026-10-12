"""
The qang angle as the angular (arcsine) transformation of a proportion (§125)

qang's angle theta = 2*arccos(sqrt(p0)) is, up to theta -> pi - theta, the
classical arcsine transformation of a binomial proportion (Bartlett 1947,
Anscombe 1948, Freeman & Tukey 1950). qang.statistics already shows that its
first-order variance is 1/N for every theta; this section checks how far the
same angle works as a tool for ANY proportion, not only a measured qubit,
using qang.proportions (Anscombe-corrected angle, interval on the angle mapped
back, Cohen's h for two samples).

Everything is computed EXACTLY by summing over the binomial distribution (no
Monte Carlo), for the grid p in {0.1, ..., 0.9} unless stated otherwise.

Pre-registered predictions (committed before the run):
  S1  n = 100: (n + 1/2) * Var(theta_A) lies in [0.9, 1.1] for every p of the
      grid (variance stabilization with Anscombe's correction).
  S2  the 95% angular interval has exact coverage in [0.93, 0.97] for
      n in {30, 100, 1000} and every p of the grid.
  S3  n = 50, p in {0.02, 0.05}: the angular interval covers at least 5 points
      more often than the Wald (delta-method) interval.
  S4  n in {100, 1000}: the expected width of the angular interval is within
      10% of the Wilson interval's (qang.statistics.wilson_qg_estimate) for
      every p of the grid.

Illustration (not a prediction; the shares were seen before this section):
the share of direct contracts in COMPR.AR 2020 for the 25 agencies with most
procedures (examples/data/comprar_2020_directas.csv, from datos.gob.ar), with
angular intervals and Cohen's h between agencies.

python examples/proportions_angular_qg.py

Findings:

  S1 PASS: (n + 1/2) Var(theta_A) = 1.000-1.001 at n = 100 for every p of the
     grid: Anscombe's corrected angle stabilizes the variance almost exactly.
  S2 PASS: exact coverage 0.930-0.962 over n in {30, 100, 1000} and the grid
     (the lowest values at n = 30).
  S3 FAIL: n = 50, p = 0.02: angular 0.982 against Wald 0.635 (as predicted),
     but p = 0.05: angular 0.911 against Wald 0.920. With np = 2.5 the
     angular interval undercovers too; its advantage is at the extremes
     (np ~ 1), not in the whole small-count region.
  S4 PASS: angular / Wilson expected width 0.99-1.01 at n in {100, 1000}.
  Verdict: for moderate and large n the qang angle gives intervals as good
  as Wilson's, with a variance that does not depend on p; for very small
  counts (np below ~3) prefer Wilson or the Bayes interval of
  qang.statistics.
  Illustration (COMPR.AR 2020): Gendarmería 83.4% [80.5, 86.1] and CNEA 81.1%
  [77.6, 84.3] are indistinguishable (h = 0.06, 95% CI -0.05 to 0.17,
  p = 0.30); Gendarmería against the Army (7.7%) is a very large effect
  (h = 1.74).
"""

import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import proportions as PR  # noqa: E402
from qang.statistics import delta_qg_estimate, wilson_qg_estimate  # noqa: E402

GRID = [round(0.1 * i, 1) for i in range(1, 10)]
Z95 = PR.z_value(0.95)


def binom_pmf(n, p):
    from math import lgamma, log, exp
    k = np.arange(n + 1)
    if p in (0.0, 1.0):
        out = np.zeros(n + 1)
        out[0 if p == 0.0 else n] = 1.0
        return out
    lc = np.array([lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) for i in k])
    return np.exp(lc + k * log(p) + (n - k) * log(1 - p))


def angular_var(n, p):
    w = binom_pmf(n, p)
    th = np.array([PR.anscombe_theta(k, n) for k in range(n + 1)])
    m = float(w @ th)
    return float(w @ (th - m) ** 2)


def coverage_and_width(n, p, method):
    w = binom_pmf(n, p)
    cov = width = 0.0
    for k in range(n + 1):
        if method == "angular":
            _, lo, hi = PR.proportion_interval(k, n, 0.95)
        else:
            e = (wilson_qg_estimate if method == "wilson" else delta_qg_estimate)(k, n, Z95)
            lo, hi = (e.low + 1) / 2, (e.high + 1) / 2
        cov += w[k] * (lo <= p <= hi)
        width += w[k] * (hi - lo)
    return float(cov), float(width)


def run():
    res = {"S1": {}, "S2": {}, "S3": {}, "S4": {}}
    for p in GRID:
        res["S1"][p] = (100 + 0.5) * angular_var(100, p)
    for n in (30, 100, 1000):
        for p in GRID:
            res["S2"][(n, p)] = coverage_and_width(n, p, "angular")[0]
    for p in (0.02, 0.05):
        res["S3"][p] = (coverage_and_width(50, p, "angular")[0], coverage_and_width(50, p, "wald")[0])
    for n in (100, 1000):
        for p in GRID:
            res["S4"][(n, p)] = coverage_and_width(n, p, "angular")[1] / coverage_and_width(n, p, "wilson")[1]
    return res


def verdict(res):
    return {
        "S1": all(0.9 <= v <= 1.1 for v in res["S1"].values()),
        "S2": all(0.93 <= v <= 0.97 for v in res["S2"].values()),
        "S3": all(a - w >= 0.05 for a, w in res["S3"].values()),
        "S4": all(abs(r - 1) <= 0.10 for r in res["S4"].values()),
    }


def comprar_illustration(path=os.path.join(HERE, "data", "comprar_2020_directas.csv")):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = []
    for r in rows:
        k, n = int(r["directas"]), int(r["procedimientos"])
        p, lo, hi = PR.proportion_interval(k, n)
        out.append((r["organismo"], n, k, p, lo, hi))
    return out


def main():
    res = run()
    print("S1 (n+1/2)Var(theta_A), n=100:", {p: round(v, 3) for p, v in res["S1"].items()})
    print("S2 coverage min/max:", round(min(res["S2"].values()), 4), round(max(res["S2"].values()), 4))
    for (n, p), v in sorted(res["S2"].items()):
        if not 0.93 <= v <= 0.97:
            print("   outside [0.93, 0.97]:", n, p, round(v, 4))
    print("S3 n=50 (angular, wald):", {p: (round(a, 3), round(w, 3)) for p, (a, w) in res["S3"].items()})
    print("S4 width angular/wilson min/max:", round(min(res["S4"].values()), 3), round(max(res["S4"].values()), 3))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in verdict(res).items()))
    print("\nCOMPR.AR 2020, share of direct contracts (95% angular interval):")
    ill = comprar_illustration()
    for org, n, k, p, lo, hi in ill:
        print(f"  {org[:55]:55s} {k:5d}/{n:<5d} {p:6.1%}  [{lo:6.1%}, {hi:6.1%}]")
    by = {o: (k, n) for o, n, k, *_ in ill}
    for a, b in [("Gendarmeria Nacional", "Comisión Nacional de Energía Atómica"),
                 ("Gendarmeria Nacional", "Estado Mayor General del Ejercito")]:
        c = PR.compare_proportions(*by[a], *by[b])
        print(f"  h({a}, {b}) = {c['h']:.3f} [{c['h_low']:.3f}, {c['h_high']:.3f}], p = {c['p_value']:.3g}")
    return res


if __name__ == "__main__":
    main()
