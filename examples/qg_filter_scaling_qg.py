"""
How the qg filter scales: qubits, weight, depth and shots (§94)

The filter keeps a known fraction of the shots, K = (1 - gamma)^(k d) for
weight k and depth d under equal T1 (§77, §78 F4). It removes the bias that
T1 puts on the readout, but the discarded shots raise the variance. Which
effect wins depends on n, k, d and the number of shots S. This study maps it
for the qg_Z readout of weight-conserving circuits, with and without qang.

For each configuration: random RBS circuits (qang.qml.WeightQNN, any weight),
random real input states in the weight-k sector, equal T1 gamma = 0.02 per
qubit per sublayer. The readout is the vector of qg_Z values. With qang it is
estimated from the kept shots; without qang it is estimated from all shots.
The figure of merit is the mean squared error against the noiseless qg_Z.

Prediction rule, stated before the run (per qubit i, ideal z_i, raw noisy
expectation r_i):
    MSE with qang    ~ mean_i (1 - z_i^2) / (K S)
    MSE without qang = mean_i [(r_i - z_i)^2 + (1 - r_i^2) / S]
The filter should win when the squared bias of the raw readout exceeds the
extra variance from discarding shots.

Grid: n = 4, 6, 8 qubits; k = 1 .. n/2; layers L = 1, 2, 4, 8 (depth
d = 3L sublayers); S = 100, 1000, 10000 shots; 10 random circuits per
configuration, 20 repetitions of the shots each. Seed 94.

Pre-registered predictions (committed before the run):
  G1  the kept fraction equals (1 - gamma)^(k d) in every configuration
      (to 1e-12).
  G2  the measured MSE with qang is within 25% of the predicted
      (1 - z^2)/(K S) in every configuration with K S >= 50.
  G3  the prediction rule picks the empirical winner (qang or raw) in at
      least 90% of the configurations.
  G4  at S = 1000 the filter has the lower MSE in every configuration with
      K >= 0.3.

Uses the installed library (pip install "qang>=0.6.3"), module qang.qml.
python examples/qg_filter_scaling_qg.py

Findings:

FINDINGS_PLACEHOLDER
"""

import itertools
import json
import os
import sys

import numpy as np

from qang.qml import WeightQNN, kept_fraction
from qang.sectors import filter_distribution

GAMMA = 0.08 / 4
SHOTS = (100, 1000, 10000)
LAYERS = (1, 2, 4, 8)
N_QUBITS = (4, 6, 8)
CIRCUITS = 10
REPS = 20


def configs():
    for n in N_QUBITS:
        for k in range(1, n // 2 + 1):
            for L in LAYERS:
                yield n, k, L


def run_config(n, k, L, rng):
    m = WeightQNN(n, k, layers=L)
    idx = m.idx[k]
    out = {s: {"mse qang": [], "mse raw": [], "pred qang": [], "pred raw": []} for s in SHOTS}
    kept_err = 0.0
    for _ in range(CIRCUITS):
        th = rng.uniform(-np.pi, np.pi, m.n_theta)
        psi = np.zeros((1, m.dim))
        psi[0, idx] = rng.normal(size=len(idx))
        psi /= np.linalg.norm(psi)
        p_exact = m.probs(th, psi)[0]
        p_noisy = m.probs(th, psi, GAMMA)[0]
        z = p_exact @ m.zsign
        r = p_noisy @ m.zsign
        K = p_noisy[idx].sum()
        kept_err = max(kept_err, abs(K - kept_fraction(GAMMA, k, m.depth)))
        for S in SHOTS:
            eq, er = [], []
            for _ in range(REPS):
                c = rng.multinomial(S, p_noisy / p_noisy.sum())
                er.append(np.mean((c / S @ m.zsign - z) ** 2))
                f, kept = filter_distribution(c, n, k)
                if kept > 0:
                    eq.append(np.mean((f @ m.zsign - z) ** 2))
                else:
                    eq.append(np.mean((0.0 - z) ** 2))  # no shot kept: no information
            out[S]["mse qang"].append(np.mean(eq))
            out[S]["mse raw"].append(np.mean(er))
            out[S]["pred qang"].append(np.mean((1 - z**2) / (K * S)))
            out[S]["pred raw"].append(np.mean((r - z) ** 2 + (1 - r**2) / S))
    K = kept_fraction(GAMMA, k, m.depth)
    rows = []
    for S in SHOTS:
        d = {key: float(np.mean(v)) for key, v in out[S].items()}
        d.update({"n": n, "k": k, "layers": L, "depth": m.depth, "shots": S, "K": K, "KS": K * S})
        rows.append(d)
    return rows, kept_err


def verdict(rows, kept_err):
    g1 = kept_err < 1e-12
    g2 = all(abs(r["mse qang"] / r["pred qang"] - 1) < 0.25 for r in rows if r["KS"] >= 50)
    agree = [(r["pred qang"] < r["pred raw"]) == (r["mse qang"] < r["mse raw"]) for r in rows]
    g3 = np.mean(agree) >= 0.90
    g4 = all(r["mse qang"] < r["mse raw"] for r in rows if r["shots"] == 1000 and r["K"] >= 0.3)
    return {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4)}, float(np.mean(agree))


def main(out=None):
    rng = np.random.default_rng(94)
    rows, kept_err = [], 0.0
    for n, k, L in configs():
        r, e = run_config(n, k, L, rng)
        rows += r
        kept_err = max(kept_err, e)
        for x in r:
            win = "qang" if x["mse qang"] < x["mse raw"] else "raw"
            print(f"n={n} k={k} d={x['depth']:2d} S={x['shots']:5d} K={x['K']:.3f} | MSE qang {x['mse qang']:.2e} "
                  f"(pred {x['pred qang']:.2e}) raw {x['mse raw']:.2e} (pred {x['pred raw']:.2e}) -> {win}", flush=True)
    v, agree = verdict(rows, kept_err)
    print(f"kept-fraction max error {kept_err:.1e}; rule agrees with the winner in {agree:.0%} of configurations")
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    if out:
        json.dump({"rows": rows, "verdict": v, "agreement": agree}, open(out, "w"), indent=1)
    return rows, v


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
