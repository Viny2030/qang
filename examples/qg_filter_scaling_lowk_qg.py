"""
The qg filter at low kept fraction and under realistic noise (§96)

§94 mapped the filter for kept fractions K >= 0.14 under equal T1: it won in
106 of 108 configurations. This study goes where the rule says it can lose
(very small K, few kept shots) and adds the noise the filter does not
correct (unequal T1 across qubits, dephasing), with every result again with
and without qang.

Grid: n = 4, 6 qubits; k = 1 .. n/2; depth d = 12, 24, 48 sublayers
(L = 4, 8, 16 layers); gamma = 0.02, 0.05, 0.10 (mean damping per qubit per
sublayer); noise: equal T1, unequal T1 (gamma_q = gamma (1 + u_q), u_q
evenly spread over [-1, 1], random assignment), equal T1 plus dephasing
0.01; S = 100, 1000, 10000 shots; 8 random circuits x 12 repetitions.
Seed 96. K ranges from about 0.6 down to below 1e-6.

Generalized rule, stated before the run. With f the exact filtered
expectation (biased when T1 is unequal or there is dephasing), r the raw
expectation, z the noiseless one and K the kept fraction:
    MSE with qang    ~ mean_i [(f_i - z_i)^2 + (1 - f_i^2) / (K S)]
    MSE without qang = mean_i [(r_i - z_i)^2 + (1 - r_i^2) / S]
When K S is so small that often no shot is kept, the filtered estimate
falls back to 0 (no information) and the rule does not apply.

Pre-registered predictions (committed before the run):
  H1  the generalized rule picks the empirical winner in at least 90% of
      the configurations with K S >= 5.
  H2  the filter loses (raw has the lower MSE) in every configuration with
      K S < 1.
  H3  under equal T1 the filter wins in every configuration with K S >= 20.
  H4  with dephasing 0.01 the filter still wins in at least 80% of the
      configurations with K S >= 20.

Uses the installed library (pip install "qang>=0.6.3"), module qang.qml.
python examples/qg_filter_scaling_lowk_qg.py

Findings:

405 configurations; K from 0.81 down to 2.6e-7.

  configurations              filter has the lower MSE   MSE without / with qang
  K S >= 20, equal T1         94 of 94                   2.7 - 1871
  K S >= 20, unequal T1       94 of 94                   2.1 - 94
  K S >= 20, dephasing 0.01   94 of 94                   1.8 - 58
  1 <= K S < 20               66 of 66
  K S < 1                     51 of 57
  all                         399 of 405

  * H1, H3, H4 pass; H2 fails.
  * H1: the generalized rule picks the winner in 342 of 342 configurations
    with K S >= 5 (100%), including unequal T1 and dephasing, where the
    filtered readout is biased.
  * H3, H4: with at least 20 kept shots the filter wins everywhere, under
    equal T1, unequal T1 (up to a factor 2 between qubits) and dephasing.
    Its margin shrinks with the noise it cannot correct: up to 1871x lower
    MSE under equal T1, 94x with unequal T1, 58x with dephasing.
  * H2 FAILS, and the reason matters. With less than one kept shot on
    average, the filtered estimate is usually "no information" (0, the
    fallback), and that still has a lower MSE than the raw readout in 51 of
    57 configurations: the raw readout has collapsed towards |0...0>
    (qg_Z -> +1), a worse guess than 0. Both readouts are useless there
    (MSE 0.4-0.7 with the filter, 0.5-1.4 without); the
    prediction assumed the raw readout would keep some information, and it
    does not once the state has decayed this far.
  Verdict. Combined with §94 (513 configurations in all): the filter has
  the lower readout error whenever at least a few shots are kept, also
  under unequal T1 and dephasing, and the generalized rule, which needs only
  the exact filtered and raw expectations and K, predicts the winner every
  time it can be applied (K S >= 5). The regime where the filter's shot
  cost decides against it was not found: when too few shots survive the
  filter, the raw readout has already lost the information too.
"""

import json
import sys

import numpy as np

from qang.qml import WeightQNN
from qang.sectors import filter_distribution

GAMMAS = (0.02, 0.05, 0.10)
SHOTS = (100, 1000, 10000)
LAYERS = (4, 8, 16)
N_QUBITS = (4, 6)
NOISES = ("equal", "unequal", "dephasing")
DEPHASING = 0.01
CIRCUITS = 8
REPS = 12


def noise_args(kind, gamma, n, rng):
    if kind == "equal":
        return gamma, 0.0
    if kind == "unequal":
        return gamma * (1 + rng.permutation(np.linspace(-1, 1, n))), 0.0
    return gamma, DEPHASING


def run_config(n, k, L, gamma, kind, rng):
    m = WeightQNN(n, k, layers=L)
    idx = m.idx[k]
    acc = {S: {"mse qang": [], "mse raw": [], "pred qang": [], "pred raw": []} for S in SHOTS}
    Ks = []
    for _ in range(CIRCUITS):
        th = rng.uniform(-np.pi, np.pi, m.n_theta)
        psi = np.zeros((1, m.dim))
        psi[0, idx] = rng.normal(size=len(idx))
        psi /= np.linalg.norm(psi)
        g, ph = noise_args(kind, gamma, n, rng)
        p0 = m.probs(th, psi)[0]
        p1 = m.probs(th, psi, g, ph)[0]
        p1 = np.clip(p1, 0, None)
        p1 /= p1.sum()
        z = p0 @ m.zsign
        r = p1 @ m.zsign
        K = p1[idx].sum()
        Ks.append(K)
        f = (filter_distribution(p1, n, k)[0] @ m.zsign) if K > 0 else np.zeros(n)
        for S in SHOTS:
            eq, er = [], []
            for _ in range(REPS):
                c = rng.multinomial(S, p1)
                er.append(np.mean((c / S @ m.zsign - z) ** 2))
                fc, kept = filter_distribution(c, n, k)
                est = fc @ m.zsign if kept > 0 else np.zeros(n)
                eq.append(np.mean((est - z) ** 2))
            acc[S]["mse qang"].append(np.mean(eq))
            acc[S]["mse raw"].append(np.mean(er))
            acc[S]["pred qang"].append(np.mean((f - z) ** 2 + (1 - f**2) / max(K * S, 1e-300)))
            acc[S]["pred raw"].append(np.mean((r - z) ** 2 + (1 - r**2) / S))
    K = float(np.mean(Ks))
    rows = []
    for S in SHOTS:
        d = {key: float(np.mean(v)) for key, v in acc[S].items()}
        d.update({"n": n, "k": k, "depth": m.depth, "gamma": gamma, "noise": kind, "shots": S, "K": K, "KS": K * S})
        rows.append(d)
    return rows


def verdict(rows):
    win = lambda r: r["mse qang"] < r["mse raw"]  # noqa: E731
    rule = [((r["pred qang"] < r["pred raw"]) == win(r)) for r in rows if r["KS"] >= 5]
    h1 = np.mean(rule) >= 0.90
    h2 = all(not win(r) for r in rows if r["KS"] < 1)
    h3 = all(win(r) for r in rows if r["noise"] == "equal" and r["KS"] >= 20)
    deph = [win(r) for r in rows if r["noise"] == "dephasing" and r["KS"] >= 20]
    h4 = np.mean(deph) >= 0.80
    return {"H1": bool(h1), "H2": bool(h2), "H3": bool(h3), "H4": bool(h4)}, float(np.mean(rule)), float(np.mean(deph))


def main(out=None):
    rng = np.random.default_rng(96)
    rows = []
    for n in N_QUBITS:
        for k in range(1, n // 2 + 1):
            for L in LAYERS:
                for gamma in GAMMAS:
                    for kind in NOISES:
                        rows += run_config(n, k, L, gamma, kind, rng)
    for r in rows:
        w = "qang" if r["mse qang"] < r["mse raw"] else "raw"
        print(f"n={r['n']} k={r['k']} d={r['depth']:2d} g={r['gamma']:.2f} {r['noise']:<9} S={r['shots']:5d} "
              f"K={r['K']:.1e} KS={r['KS']:.1e} | qang {r['mse qang']:.2e} raw {r['mse raw']:.2e} -> {w}")
    v, rule, deph = verdict(rows)
    print(f"{len(rows)} configurations; rule agreement (KS >= 5) {rule:.0%}; filter wins with dephasing (KS >= 20) {deph:.0%}")
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    if out:
        json.dump({"rows": rows, "verdict": v, "rule agreement": rule, "dephasing wins": deph}, open(out, "w"), indent=1)
    return rows, v


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
