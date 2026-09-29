"""
The (1 - h) rule for qg on erasure qubits, tested with its own
pre-registration (§74).

§73 found that combining the qg T1 reweighting with heralded erasures
naively is harmful when heralding is good (0.46x at h = 0.99), and an
exploratory fix, found after the run on one seed: scale the prior of the
UNHERALDED decays by (1 - h), in the erasure decoder and in qg + erasure.
Here the fix is tested on new seeds, a second noise regime, a finer grid
of h, and with h misestimated by the decoder.

Model: the §73 memory (the §64 circuit-level Z-memory; each data decay
1 -> 0 heralded with probability h at its exact location).
Regimes: T1-dominated (p2 = 0.0005, gamma = 0.003, q = 0.002, witness 1.68
at d = 3) and mixed (p2 = 0.002, gamma = 0.002, q = 0.002, witness 1.27);
qg is switched on by the §64 witness in both. d = 3 (3 rounds) and d = 5
(5 rounds); h in {0.25, 0.5, 0.75, 0.9, 0.99}.

Decoders (the decoder assumes heralding efficiency h_dec; h_dec = h unless
stated):
  erasure               heralded edges at p = 1/2, decay priors unchanged (§73)
  erasure, calibrated   ... and unheralded decay priors scaled by (1 - h_dec)
  qg + erasure, cal.    ... and the §64 T1 reweighting

Pre-registered predictions (written and committed before the seed-740 run;
60 000 shots per point):
  F1  qg + erasure, calibrated is never worse than erasure, calibrated
      (paired z > -3) at any h, d and regime.
  F2  at h = 0.5, T1-dominated regime, it beats erasure, calibrated by at
      least 1.3x with z >= 3 at d = 3 and d = 5.
  F3  erasure, calibrated is never worse than erasure (z > -3) anywhere.
  F4  misestimated h (d = 3, T1-dominated): with true h = 0.5 and h_dec in
      {0.4, 0.6}, and true h = 0.9 and h_dec in {0.8, 0.97}, qg + erasure,
      calibrated is still never worse than erasure, calibrated (z > -3).

Needs pymatching.

Findings (python examples/erasure_calibrated_qg.py):

FINDINGS_PLACEHOLDER
"""

import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import erasure_qubits_qg as X  # noqa: E402
import qutrit_leakage_qg as Q  # noqa: E402
import surface_code_circuit_t1_qg as SC  # noqa: E402

REGIMES = {
    "T1-dominated": dict(p2=0.0005, gamma=0.003, q=0.002),
    "mixed": dict(p2=0.002, gamma=0.002, q=0.002),
}
HS = (0.25, 0.5, 0.75, 0.9, 0.99)
MISEST = ((0.5, 0.4), (0.5, 0.6), (0.9, 0.8), (0.9, 0.97))
DECODERS = ("erasure", "erasure, calibrated", "qg + erasure, calibrated")


def run(h, d=3, h_dec=None, shots=60000, seed=740, p2=0.0005, gamma=0.003, q=0.002):
    h_dec = h if h_dec is None else h_dec
    rounds = d
    rng = np.random.default_rng(seed)
    code = SC.Code(d)
    E = SC.prepare_edges(SC.fault_list(code, rounds, p2, gamma, q))
    tab = X.edge_table(code, rounds)
    wit = SC.witness_ratio(d, rounds, p2, gamma, q)
    use_t1 = wit > 1
    base = np.full(code.nd, 1 - h_dec)
    cache = {}

    def get(key, mult, er):
        if key not in cache:
            if len(cache) > 20000:
                cache.clear()
            cache[key] = Q.build(E, q, mult, [er] if er else ())
        return cache[key]

    fails = {k: [] for k in DECODERS}
    for logical in (0, 1):
        meas, final, her = X.simulate(code, rounds, shots // 2, rng, p2, gamma, q, h, logical)
        det, lg = SC.detectors(code, meas, final)
        f = {k: np.zeros(len(det), bool) for k in DECODERS}
        groups = defaultdict(list)
        for i in range(len(det)):
            er = frozenset(tab[kx] for kx in her[i] if kx in tab)
            groups[(er, tuple(final[i].astype(np.uint8)) if use_t1 else None)].append(i)
        for (er, bits), idx in groups.items():
            idx = np.array(idx)
            dd, ll = det[idx], lg[idx] ^ bool(logical)
            f["erasure"][idx] = get(("e", er), None, er).decode_batch(dd)[:, 0].astype(bool) ^ ll
            f["erasure, calibrated"][idx] = get(("ec", er), base, er).decode_batch(dd)[:, 0].astype(bool) ^ ll
            if use_t1:
                m = np.where(np.array(bits) == 0, 2.0, 2.0 * q) * base
                f["qg + erasure, calibrated"][idx] = get(("gc", er, bits), m, er).decode_batch(dd)[:, 0].astype(bool) ^ ll
            else:
                f["qg + erasure, calibrated"][idx] = f["erasure, calibrated"][idx]
        for k in DECODERS:
            fails[k].append(f[k])
    fails = {k: np.concatenate(v) for k, v in fails.items()}
    out = {k: float(v.mean()) for k, v in fails.items()}

    def z(a, b):  # positive when b fails less than a
        n1 = int(np.sum(fails[a] & ~fails[b]))
        n2 = int(np.sum(fails[b] & ~fails[a]))
        return (n1 - n2) / math.sqrt(n1 + n2) if n1 + n2 else 0.0

    out["gain"] = out["erasure, calibrated"] / max(out["qg + erasure, calibrated"], 1e-12)
    out["z_gain"] = z("erasure, calibrated", "qg + erasure, calibrated")
    out["z_cal"] = z("erasure", "erasure, calibrated")
    out["witness"] = wit
    return out


def verdict(main_res, mis_res):
    f1 = all(r["z_gain"] > -3 for r in main_res.values())
    f2 = all(main_res[("T1-dominated", d, 0.5)]["gain"] >= 1.3 and main_res[("T1-dominated", d, 0.5)]["z_gain"] >= 3
             for d in (3, 5))
    f3 = all(r["z_cal"] > -3 for r in main_res.values())
    f4 = all(r["z_gain"] > -3 for r in mis_res.values())
    return {"F1": f1, "F2": f2, "F3": f3, "F4": f4}


def main(shots=60000, seed=740):
    res, mis = {}, {}
    for g, kw in REGIMES.items():
        for d in (3, 5):
            for h in HS:
                r = run(h, d=d, shots=shots, seed=seed, **kw)
                res[(g, d, h)] = r
                print(f"{g:13s} d={d} h={h:<4} (witness {r['witness']:.2f}): "
                      + " | ".join(f"{k} {r[k]:.5f}" for k in DECODERS)
                      + f" || gain {r['gain']:.2f}x (z {r['z_gain']:+.1f}); calibration vs naive z {r['z_cal']:+.1f}", flush=True)
    for h, hd in MISEST:
        r = run(h, d=3, h_dec=hd, shots=shots, seed=seed + 1, **REGIMES["T1-dominated"])
        mis[(h, hd)] = r
        print(f"misestimated: h={h}, decoder assumes {hd}: " + " | ".join(f"{k} {r[k]:.5f}" for k in DECODERS)
              + f" || gain {r['gain']:.2f}x (z {r['z_gain']:+.1f})", flush=True)
    v = verdict(res, mis)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return res, mis, v


if __name__ == "__main__":
    main()
