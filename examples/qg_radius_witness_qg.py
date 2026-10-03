"""
The local radius as an entanglement measure under T1, with and without qang (§98)

§97 adds the local radius to the qg formulation: r_q^2 = qg_X^2 + qg_Y^2 +
qg_Z^2 for each qubit (the sphere area 4 pi r^2; the deficit 1 - r^2). For a
pure global state the deficit is the one-tangle of qubit q with the rest
(Coffman-Kundu-Wootters; the mean over qubits is the Meyer-Wallach Q). For a
noisy state it mixes entanglement with noise, and that is the problem this
study tests.

Weight-conserving circuits give a way out. In a state of definite Hamming
weight every local qg_X and qg_Y is zero, so r_q = |qg_Z| and the deficit is
1 - qg_Z^2, read from Z-basis shots alone. Under equal T1 the qg filter
restores the noiseless state exactly (F1/F4), so the filtered deficit should
be the noiseless tangle; the raw deficit is not.

Setup: qang.qml.WeightQNN RBS circuits, n = 4, 6 qubits, weight k = 1 .. n/2,
L = 4, 8, 16 layers (depth d = 12, 24, 48 sublayers), random real inputs in
the weight-k sector, gamma = 0.02 per qubit per sublayer; noise: equal T1,
unequal T1 (gamma_q = gamma (1 + u_q), u_q spread over [-1, 1]), equal T1
plus dephasing 0.01. Truth: the noiseless deficit 1 - qg_Z^2 per qubit.
Estimators from S = 1000 and 10000 Z-basis shots, per qubit,
1 - qg2_unbiased(k0, N) (qang.statistics), with qang on the kept shots and
without qang on all shots. 8 circuits x 20 repetitions per configuration.
Product family: basis-state inputs and idle circuits (all angles 0), truth
deficit 0. Seed 98.

Pre-registered predictions (written and committed before the run; code
debugged on a 1-circuit, 2-repetition grid, results not looked at):
  E1  under equal T1 the exact filtered deficit equals the noiseless tangle
      to 1e-10 for every qubit in every configuration.
  E2  under equal T1 the exact raw deficit has a mean absolute error above
      0.05 in every configuration with k < n/2, and a smaller mean absolute
      error at half filling (k = n/2) than at k = 1 for the same n and d
      (first-order cancellation near p = 1/2).
  E3  product family: with qang the estimated deficit is exactly 0 for every
      qubit under all three noise models (no false entanglement); without qang,
      under equal T1, every initially excited qubit shows a deficit above 0.5.
  E4  random circuits, equal T1: the deficit MSE with qang is lower than
      without in at least 90% of the configurations with K S >= 20.
  E5  with qang under equal T1, the mean error of the unbiased estimator is
      within 3 standard errors of 0 in at least 90% of the configurations with
      K S >= 20.
  E6  under unequal T1 and dephasing the filtered deficit is no longer exact,
      but its exact mean absolute error is lower than the raw one in at least
      80% of those configurations.

Uses the installed library (pip install "qang>=0.6.4"): qang.qml,
qang.statistics.qg2_unbiased, qang.formulation (radius functions).
python examples/qg_radius_witness_qg.py [out.json]

Findings:

FINDINGS_PLACEHOLDER
"""

import json
import sys

import numpy as np

from qang.qml import WeightQNN
from qang.sectors import filter_distribution

GAMMA = 0.02
DEPHASING = 0.01
SHOTS = (1000, 10000)
LAYERS = (4, 8, 16)
N_QUBITS = (4, 6)
NOISES = ("equal", "unequal", "dephasing")
CIRCUITS = 8
REPS = 20


def noise_args(kind, n, rng):
    if kind == "equal":
        return GAMMA, 0.0
    if kind == "unequal":
        return GAMMA * (1 + rng.permutation(np.linspace(-1, 1, n))), 0.0
    return GAMMA, DEPHASING


def deficit_from_dist(dist, zsign):
    return 1.0 - (dist @ zsign) ** 2


def deficit_from_counts(c, zsign):
    """Per-qubit 1 - unbiased qg^2 from a count vector (0 shots -> nan)."""
    N = c.sum()
    if N < 2:
        return np.full(zsign.shape[1], np.nan)
    q = (c @ zsign) / N
    return 1.0 - (N * q * q - 1.0) / (N - 1.0)


def run_config(n, k, L, kind, rng, circuits=CIRCUITS, reps=REPS):
    m = WeightQNN(n, k, layers=L)
    idx = m.idx[k]
    keep = m.wt == k
    rows = {S: {"mse qang": [], "mse raw": [], "bias": [], "se": []} for S in SHOTS}
    exact_f, exact_r, Ks = [], [], []
    for _ in range(circuits):
        th = rng.uniform(-np.pi, np.pi, m.n_theta)
        psi = np.zeros((1, m.dim))
        psi[0, idx] = rng.normal(size=len(idx))
        psi /= np.linalg.norm(psi)
        g, ph = noise_args(kind, n, rng)
        p0 = m.probs(th, psi)[0]
        p1 = np.clip(m.probs(th, psi, g, ph)[0], 0, None)
        p1 /= p1.sum()
        truth = deficit_from_dist(p0, m.zsign)
        K = p1[keep].sum()
        Ks.append(K)
        f = filter_distribution(p1, n, k)[0]
        exact_f.append(np.abs(deficit_from_dist(f, m.zsign) - truth))
        exact_r.append(np.abs(deficit_from_dist(p1, m.zsign) - truth))
        for S in SHOTS:
            eq, er, err = [], [], []
            for _ in range(reps):
                c = rng.multinomial(S, p1)
                dr = deficit_from_counts(c, m.zsign)
                dq = deficit_from_counts(np.where(keep, c, 0), m.zsign)
                dq = np.where(np.isnan(dq), 0.5, dq)  # no kept shots: no information
                er.append(np.mean((dr - truth) ** 2))
                eq.append(np.mean((dq - truth) ** 2))
                err.append(np.mean(dq - truth))
            rows[S]["mse qang"].append(np.mean(eq))
            rows[S]["mse raw"].append(np.mean(er))
            rows[S]["bias"].append(np.mean(err))
            rows[S]["se"].append(np.std(err, ddof=1) / np.sqrt(len(err)))
    K = float(np.mean(Ks))
    out = []
    for S in SHOTS:
        b = np.array(rows[S]["bias"])
        se = np.array(rows[S]["se"])
        out.append({"n": n, "k": k, "depth": m.depth, "noise": kind, "shots": S, "K": K, "KS": K * S,
                    "mse qang": float(np.mean(rows[S]["mse qang"])), "mse raw": float(np.mean(rows[S]["mse raw"])),
                    "bias qang": float(np.mean(b)), "bias se": float(np.sqrt(np.sum(se**2)) / len(se)),
                    "exact err qang": float(np.mean(exact_f)), "exact max err qang": float(np.max(exact_f)),
                    "exact err raw": float(np.mean(exact_r))})
    return out


def product_family(n, k, L, kind, rng, S=1000):
    """Basis-state input, idle circuit (all angles 0): truth deficit 0."""
    m = WeightQNN(n, k, layers=L)
    s = rng.choice(m.idx[k])
    psi = np.zeros((1, m.dim))
    psi[0, s] = 1.0
    g, ph = noise_args(kind, n, rng)
    p1 = np.clip(m.probs(np.zeros(m.n_theta), psi, g, ph)[0], 0, None)
    p1 /= p1.sum()
    c = rng.multinomial(S, p1)
    dq = deficit_from_counts(np.where(m.wt == k, c, 0), m.zsign)
    dr = deficit_from_counts(c, m.zsign)
    excited = m.zsign[s] < 0
    return {"n": n, "k": k, "depth": m.depth, "noise": kind, "max deficit qang": float(np.nanmax(np.abs(dq))),
            "min raw deficit excited": float(np.min(dr[excited]))}


def verdict(rows, prod):
    eq = [r for r in rows if r["noise"] == "equal"]
    e1 = max(r["exact max err qang"] for r in eq) < 1e-10
    seen = {(r["n"], r["k"], r["depth"]): r["exact err raw"] for r in eq}
    e2a = all(v > 0.05 for (n, k, d), v in seen.items() if k < n / 2)
    e2b = all(seen[(n, n // 2, d)] < seen[(n, 1, d)] for (n, k, d) in seen)
    e3 = all(p["max deficit qang"] == 0.0 for p in prod) and all(
        p["min raw deficit excited"] > 0.5 for p in prod if p["noise"] == "equal")
    big = [r for r in eq if r["KS"] >= 20]
    e4 = np.mean([r["mse qang"] < r["mse raw"] for r in big]) >= 0.90
    e5 = np.mean([abs(r["bias qang"]) <= 3 * r["bias se"] for r in big]) >= 0.90
    other = [r for r in rows if r["noise"] != "equal" and r["shots"] == SHOTS[0]]
    e6 = np.mean([r["exact err qang"] < r["exact err raw"] for r in other]) >= 0.80
    return {"E1": bool(e1), "E2": bool(e2a and e2b), "E3": bool(e3), "E4": bool(e4), "E5": bool(e5), "E6": bool(e6)}


def main(out=None, circuits=CIRCUITS, reps=REPS):
    rng = np.random.default_rng(98)
    rows, prod = [], []
    for n in N_QUBITS:
        for k in range(1, n // 2 + 1):
            for L in LAYERS:
                for kind in NOISES:
                    rows += run_config(n, k, L, kind, rng, circuits, reps)
                    prod.append(product_family(n, k, L, kind, rng))
    for r in rows:
        w = "qang" if r["mse qang"] < r["mse raw"] else "raw"
        print(f"n={r['n']} k={r['k']} d={r['depth']:2d} {r['noise']:<9} S={r['shots']:5d} K={r['K']:.2f} | "
              f"exact err qang {r['exact err qang']:.1e} raw {r['exact err raw']:.3f} | "
              f"MSE qang {r['mse qang']:.1e} raw {r['mse raw']:.1e} -> {w} | bias {r['bias qang']:+.1e} "
              f"(se {r['bias se']:.1e})")
    for p in prod:
        print(f"product n={p['n']} k={p['k']} d={p['depth']:2d} {p['noise']:<9} | max deficit qang "
              f"{p['max deficit qang']:.2e} | min raw deficit (excited) {p['min raw deficit excited']:.3f}")
    v = verdict(rows, prod)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    if out:
        json.dump({"rows": rows, "product": prod, "verdict": v}, open(out, "w"), indent=1)
    return rows, prod, v


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
