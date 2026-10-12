"""
Wave-particle duality as the Bloch radius, and entropic uncertainty with qg_S (§127)

Duality: path predictability P = |qg_Z| and fringe visibility
V = sqrt(qg_X^2 + qg_Y^2) satisfy P^2 + V^2 = r^2 <= 1 (Greenberger & Yasin
1988; Jaeger, Shimony & Vaidman 1995; Englert 1996). Entropic uncertainty:
qg_S(Z) + qg_S(X) >= 1 + S(rho) (Maassen & Uffink 1988; mixed-state term from
Berta et al. 2010). qang.complementarity computes both from qg values and from
counts; this section checks the identities and, mainly, how they behave with
finite shots.

Random states: uniform in the Bloch ball (seed 127).

Pre-registered predictions (committed before the run):
  U1  on 2000 random states P^2 + V^2 equals r^2 within 1e-12 and never
      exceeds 1.
  U2  on the same states qg_S(Z) + qg_S(X) - (1 + S(rho)) >= -1e-12, and its
      minimum is below 1e-3 (the bound is tight).
  U3  N = 1000 shots per basis, 20 random states, 2000 repetitions each: the
      unbiased estimate of P^2 + V^2 has |mean bias| < 5e-4 on every state,
      and the plug-in estimate is biased upwards by 0.0015-0.0035 on average.
  U4  |0>, N = 100 shots per basis, 2000 repetitions: the plug-in entropic
      sum falls below 1 (an apparent violation) in more than 80% of runs; with
      the Miller-Madow correction, in fewer than 60%.

python examples/complementarity_qg.py

Findings:

  U1 PASS: |P^2 + V^2 - r^2| <= 2.2e-16 and max P^2 + V^2 = 0.99992 on 2000
     states (the identity is exact).
  U2 PASS: the minimum slack of qg_S(Z) + qg_S(X) >= 1 + S(rho) over the
     random states is 1e-5 and it is 0 for |0> (the bound is tight).
  U3 FAIL: the plug-in estimate of P^2 + V^2 is biased upwards by 0.0027 on
     average (0.0012-0.0037, as predicted), but the largest |mean bias| of
     the unbiased estimate is 0.0014, above the 5e-4 threshold. The
     threshold was set below the Monte Carlo resolution: the standard error
     of a mean over 2000 repetitions is about 0.0012 here, so 0.0014 on the
     worst of 20 states is noise; the estimator is exactly unbiased by
     construction (qang.statistics.qg2_unbiased). The prediction was badly
     designed, and it is reported as failed.
  U4 PASS: for |0> with 100 shots per basis the plug-in entropic sum is
     below 1 in 93.1% of runs (an apparent violation of the bound); with
     the Miller-Madow correction, in 37.2%.
  Verdict: duality is the Bloch radius and entropic uncertainty is a bound
  on two entropic qangs. An experimental test of either needs bias-corrected
  estimators: plug-in entropies make a pure state seem to violate the
  uncertainty relation most of the time.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qang import complementarity as CM  # noqa: E402

SEED = 127


def random_states(n, rng):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    r = rng.uniform(0, 1, n) ** (1 / 3)
    return v * r[:, None]


def run():
    rng = np.random.default_rng(SEED)
    S = random_states(2000, rng)
    d = [abs(CM.duality_sum(x, y, z) - (x * x + y * y + z * z)) for x, y, z in S]
    mx = max(CM.duality_sum(x, y, z) for x, y, z in S)
    slack = [CM.entropic_sum(z, x) - CM.entropic_bound(x, y, z) for x, y, z in S]
    res = {"U1": (max(d), mx), "U2": (min(slack),)}
    # U2 tightness: the bound is reached by pure states on the Z (or X) axis
    res["U2"] += (CM.entropic_sum(1.0, 0.0) - CM.entropic_bound(0.0, 0.0, 1.0),)

    n, reps = 1000, 2000
    unb, plug = [], []
    for x, y, z in random_states(20, rng):
        r2 = x * x + y * y + z * z
        ks = [rng.binomial(n, (1 + q) / 2, reps) for q in (x, y, z)]
        e_u = np.array([CM.duality_from_counts(*[(int(k[i]), n) for k in ks])[0] for i in range(reps)])
        e_p = np.array([CM.duality_from_counts(*[(int(k[i]), n) for k in ks], unbiased=False)[0] for i in range(reps)])
        unb.append(float(e_u.mean() - r2))
        plug.append(float(e_p.mean() - r2))
    res["U3"] = (max(abs(b) for b in unb), float(np.mean(plug)), min(plug), max(plug))

    n, reps = 100, 2000
    kz = np.full(reps, n)                         # |0>: every Z shot gives 0
    kx = rng.binomial(n, 0.5, reps)
    plug_s = [CM.entropic_sum_from_counts((int(a), n), (int(b), n), "none") for a, b in zip(kz, kx)]
    mm_s = [CM.entropic_sum_from_counts((int(a), n), (int(b), n), "miller_madow") for a, b in zip(kz, kx)]
    res["U4"] = (float(np.mean(np.array(plug_s) < 1)), float(np.mean(np.array(mm_s) < 1)))
    return res


def verdict(res):
    return {
        "U1": res["U1"][0] < 1e-12 and res["U1"][1] <= 1 + 1e-12,
        "U2": res["U2"][0] >= -1e-12 and min(res["U2"]) < 1e-3,
        "U3": res["U3"][0] < 5e-4 and 0.0015 <= res["U3"][1] <= 0.0035,
        "U4": res["U4"][0] > 0.80 and res["U4"][1] < 0.60,
    }


def main():
    res = run()
    print("U1 max |P^2+V^2-r^2|, max P^2+V^2:", res["U1"])
    print("U2 min slack over random states, slack at |0>:", tuple(round(v, 6) for v in res["U2"]))
    print("U3 max |bias| unbiased, mean/min/max plug-in bias:", tuple(round(v, 5) for v in res["U3"]))
    print("U4 fraction below 1 (plug-in, Miller-Madow):", tuple(round(v, 3) for v in res["U4"]))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in verdict(res).items()))
    return res


if __name__ == "__main__":
    main()
