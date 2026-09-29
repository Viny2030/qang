"""
Filter or zero-noise extrapolation? An adaptive choice from the shot budget.

Across §24, §39 and §60 the same pattern appeared: the qg filter has a
residual bias and a small variance, zero-noise extrapolation (ZNE, linear
Richardson from 1x and 3x CNOT folding) removes more bias but amplifies the
variance ((1.5^2 + 0.5^2) = 2.5x per shot, with the shots split between two
circuits), and filter + ZNE combines both. Which one wins depends on the
number of shots. Here the question is turned into a rule that runs on data.

Test bed: the §60 XXZ chain (n = 6, 3 CNOTs per bond, p2 = 0.01,
gamma = 0.005, e = 0.01) at 2, 4 and 6 Trotter steps, observable the
imbalance. Exact outcome distributions of the 1x and 3x circuits; the shots
are sampled. For a total budget S:

  fixed strategies
    filter        all S shots on the 1x circuit, post-selected on N = n/2
    ZNE           S/2 on each circuit, raw estimates, 1.5 f1 - 0.5 f3
    filter + ZNE  S/2 on each circuit, filtered estimates extrapolated
  adaptive (label-free)
    a pilot of 10 % of S on each circuit; the bias of the filter is estimated
    as |filtered - (filter + ZNE)| on the pilot, and each strategy's variance
    from the pilot counts, scaled to the full budget; the strategy with the
    smallest estimated mean squared error (squared bias estimate minus its
    own noise, floored at 0, plus variance) gets the remaining 80 % of the
    budget, and the pilot shots are reused.

Prediction, written before the run: at every budget S in {500, 2000, 10^4,
5x10^4} and every depth, the adaptive rule's RMSE is within 1.25x of the
best fixed strategy, while every fixed strategy is at least 1.5x worse than
the best at some (S, depth).

Findings (python examples/shot_budget_adaptive_zne_qg.py):

  RMSE of the imbalance (400 repetitions per cell):
    depth  S       filter  ZNE     f+ZNE   adaptive  best
    2      500     0.030   0.048   0.061   0.031     filter
    2      2000    0.020   0.024   0.028   0.021     filter
    2      10^4    0.016   0.014   0.013   0.017     filter+ZNE
    2      5x10^4  0.016   0.009   0.007   0.012     filter+ZNE
    4      500     0.046   0.061   0.079   0.049     filter
    4      2000    0.034   0.041   0.042   0.035     filter
    4      10^4    0.031   0.031   0.018   0.031     filter+ZNE
    4      5x10^4  0.031   0.029   0.010   0.023     filter+ZNE
    6      500     0.150   0.173   0.102   0.150     filter+ZNE
    6      2000    0.148   0.170   0.073   0.135     filter+ZNE
    6      10^4    0.146   0.168   0.061   0.068     filter+ZNE
    6      5x10^4  0.146   0.168   0.059   0.059     filter+ZNE
  Worst ratio to the best fixed strategy over the 12 cells: filter 3.07,
  ZNE 2.93, filter + ZNE 2.02, adaptive (pre-registered) 2.29, post-hoc
  variant 1.49.

  * The trade-off is real and has a clear shape. For shallow circuits
    (2-4 steps) the filter alone wins up to ~2000 shots (1.7-2.0x lower RMSE
    than filter + ZNE at 500 shots); from ~10^4 shots filter + ZNE wins,
    by up to 3.1x at 5x10^4. For deep circuits (6 steps, 90 CNOTs) the
    filter's bias dominates at every budget and filter + ZNE wins everywhere.
    ZNE alone is never the best.
  * The pre-registered prediction FAILS: the adaptive rule is 2.29x worse
    than the best fixed strategy in the worst cell, worse than simply always
    using filter + ZNE (2.02x). A 10 % pilot cannot see a bias of ~0.02
    under its own noise, so it keeps choosing the filter at 10^4-5x10^4
    shots, where filter + ZNE is better.
  * A variant chosen after the failure (20 % pilot, no noise subtraction,
    filter vs filter + ZNE only) reaches 1.49x in the worst cell. It is
    exploratory, not a result: it was tuned on these data.
  * Practical rule from these numbers (a heuristic, not a derived bound):
    use the filter alone only for shallow circuits and at most a few
    thousand shots; otherwise use filter + ZNE; never ZNE alone when a
    symmetry filter is available.

  Honest scope: one model (the §60 chain, 3-CNOT bonds, one noise mix),
  linear Richardson with 1x and 3x folding only, and budget split equally
  between the two ZNE circuits. Bias-variance trade-offs in error
  mitigation are known; the contribution is the measured crossover for the
  qg filter and the negative result on a simple pilot-based switch.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

N_Q = 6
BUDGETS = (500, 2000, 10000, 50000)
DEPTHS = (2, 4, 6)
NOISE = dict(p2=0.01, gamma=0.005, e=0.01)


def _tables():
    import xxz_trotter_filter_qg as X

    imb, N = X._tables(N_Q)
    return imb, N == N_Q // 2


IMB, KEEP = _tables()


def distributions(steps, dt=0.25):
    import xxz_trotter_filter_qg as X

    p1 = X.probabilities(N_Q, steps, dt, fold=1, **NOISE)
    p3 = X.probabilities(N_Q, steps, dt, fold=3, **NOISE)
    ideal = X.imbalance(X.probabilities(N_Q, steps, dt), N_Q)[0]
    return p1 / p1.sum(), p3 / p3.sum(), ideal


def _stats(counts):
    """(raw mean, raw var per shot, filtered mean, filtered var per shot of the
    filtered estimator, kept fraction)."""
    S = counts.sum()
    raw = counts @ IMB / S
    raw_v = counts @ (IMB - raw) ** 2 / S
    k = counts[KEEP].sum()
    if k == 0:
        return raw, raw_v, raw, raw_v, 0.0
    f = counts[KEEP] @ IMB[KEEP] / k
    fv = counts[KEEP] @ (IMB[KEEP] - f) ** 2 / k
    return raw, raw_v, f, fv / max(k / S, 1e-9), k / S


def fixed(p1, p3, S, rng):
    c1 = rng.multinomial(S, p1)
    h1 = rng.multinomial(S // 2, p1)
    h3 = rng.multinomial(S - S // 2, p3)
    f_full = _stats(c1)[2]
    r1, _, f1, _, _ = _stats(h1)
    r3, _, f3, _, _ = _stats(h3)
    return {"filter": f_full, "ZNE": 1.5 * r1 - 0.5 * r3, "filter+ZNE": 1.5 * f1 - 0.5 * f3}


def adaptive(p1, p3, S, rng, pilot=0.1):
    n_p = max(int(pilot * S), 20)
    a1 = rng.multinomial(n_p, p1)
    a3 = rng.multinomial(n_p, p3)
    r1, rv1, f1, fv1, _ = _stats(a1)
    r3, rv3, f3, fv3, _ = _stats(a3)
    fz = 1.5 * f1 - 0.5 * f3
    rest = S - 2 * n_p
    # per-shot variances -> variance at the full budget for each strategy
    var_f = fv1 / (n_p + rest)
    var_fz = (2.25 * fv1 + 0.25 * fv3) / (S / 2)
    var_z = (2.25 * rv1 + 0.25 * rv3) / (S / 2)
    # bias of the filter estimated against filter+ZNE (assumed unbiased), minus its own noise
    bias_f2 = max((f1 - fz) ** 2 - (fv1 / n_p + (2.25 * fv1 + 0.25 * fv3) / n_p), 0.0)
    bias_z2 = max((1.5 * r1 - 0.5 * r3 - fz) ** 2 - ((2.25 * rv1 + 0.25 * rv3) / n_p + (2.25 * fv1 + 0.25 * fv3) / n_p), 0.0)
    mse = {"filter": bias_f2 + var_f, "filter+ZNE": var_fz, "ZNE": bias_z2 + var_z}
    choice = min(mse, key=mse.get)
    if choice == "filter":
        c = a1 + rng.multinomial(rest, p1)
        return _stats(c)[2], choice
    b1 = a1 + rng.multinomial(rest // 2, p1)
    b3 = a3 + rng.multinomial(rest - rest // 2, p3)
    s1, s3 = _stats(b1), _stats(b3)
    if choice == "ZNE":
        return 1.5 * s1[0] - 0.5 * s3[0], choice
    return 1.5 * s1[2] - 0.5 * s3[2], choice


def adaptive_posthoc(p1, p3, S, rng, pilot=0.2):
    """EXPLORATORY variant chosen after seeing the pre-registered rule fail:
    20 % pilot, no noise subtraction, only filter vs filter+ZNE."""
    n_p = max(int(pilot * S), 20)
    a1 = rng.multinomial(n_p, p1)
    a3 = rng.multinomial(n_p, p3)
    r1, rv1, f1, fv1, _ = _stats(a1)
    r3, rv3, f3, fv3, _ = _stats(a3)
    fz = 1.5 * f1 - 0.5 * f3
    rest = S - 2 * n_p
    if (f1 - fz) ** 2 + fv1 / (n_p + rest) < (2.25 * fv1 + 0.25 * fv3) / (S / 2):
        return _stats(a1 + rng.multinomial(rest, p1))[2], "filter"
    b1 = a1 + rng.multinomial(rest // 2, p1)
    b3 = a3 + rng.multinomial(rest - rest // 2, p3)
    return 1.5 * _stats(b1)[2] - 0.5 * _stats(b3)[2], "filter+ZNE"


def study(reps=400, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for d in DEPTHS:
        p1, p3, ideal = distributions(d)
        for S in BUDGETS:
            errs = {"filter": [], "ZNE": [], "filter+ZNE": [], "adaptive": [], "posthoc": []}
            picks = {}
            for _ in range(reps):
                fx = fixed(p1, p3, S, rng)
                for k, v in fx.items():
                    errs[k].append(v - ideal)
                a, ch = adaptive(p1, p3, S, rng)
                errs["adaptive"].append(a - ideal)
                picks[ch] = picks.get(ch, 0) + 1
                errs["posthoc"].append(adaptive_posthoc(p1, p3, S, rng)[0] - ideal)
            rmse = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in errs.items()}
            rows.append({"depth": d, "S": S, "rmse": rmse, "picks": {k: v / reps for k, v in picks.items()}})
    return rows


def main():
    rows = study()
    print(f"{'depth':>5} {'S':>6} | {'filter':>7} {'ZNE':>7} {'f+ZNE':>7} {'adapt':>7} | best fixed  adapt/best  picks")
    worst_fixed = {"filter": 1.0, "ZNE": 1.0, "filter+ZNE": 1.0}
    worst_adapt = 1.0
    worst_post = 1.0
    for r in rows:
        m = r["rmse"]
        best = min(m["filter"], m["ZNE"], m["filter+ZNE"])
        bname = min(("filter", "ZNE", "filter+ZNE"), key=lambda k: m[k])
        for k in worst_fixed:
            worst_fixed[k] = max(worst_fixed[k], m[k] / best)
        worst_adapt = max(worst_adapt, m["adaptive"] / best)
        worst_post = max(worst_post, m["posthoc"] / best)
        pk = " ".join(f"{k}:{v:.2f}" for k, v in sorted(r["picks"].items()))
        print(f"{r['depth']:5d} {r['S']:6d} | {m['filter']:.4f} {m['ZNE']:.4f} {m['filter+ZNE']:.4f} {m['adaptive']:.4f}"
              f" | {bname:10s} {m['adaptive'] / best:5.2f}      {pk}")
    print(f"\nworst ratio to the best fixed strategy: " + ", ".join(f"{k} {v:.2f}" for k, v in worst_fixed.items())
          + f", adaptive (pre-registered) {worst_adapt:.2f}, post-hoc variant {worst_post:.2f}")
    return rows


if __name__ == "__main__":
    main()
