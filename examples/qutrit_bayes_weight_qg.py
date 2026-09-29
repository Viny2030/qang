"""
A rule instead of a recipe: the Bayesian weight of a |2> flag, and the
qutrit gain at distance 5 (§71).

§70 and §70b gave opposite answers for the same |2> flag: useless (even
harmful) when a leaked qubit keeps its value, useful when it does not. Both
used erasure (edge probability 1/2 for a flagged qubit), a rule that ignores
where the leak came from. The principled weight:

  a data qubit leaks from |1> with probability ell and from |0> with
  probability a * ell (0 <= a <= 1, measurable by calibration). Seen in |2>
  at the final readout, its pre-leak value was 1 with posterior
      P(b = 1 | |2>) = 1 / (1 + a)          (uniform prior on b).
  So: report the flagged bit as 1 and give its final-readout edges the
  flip probability
      p_flag = max(a / (1 + a), q).
  a = 0: the flag says "the bit is 1" with certainty (no erasure);
  a = 1: p_flag = 1/2, plain erasure. A flagged qubit does not decay, so its
  §64 T1 multiplier is set to 1 (neutral).

The rule contains §70 as a special case: there a = 0 and |2> is read as 1,
so the Bayesian decoder reduces to the standard one, and should never be
worse than it, unlike erasure.

Models (the §64 circuit-level Z-memory plus data-qubit leakage after
CNOTs, ancilla kicked at random by a leaked control, return probability r
per layer, three-level final readout):
  M0      the §70 model: a = 0, return to |1>, |2> read as 1
  M(a)    a in {0, 0.25, 0.5, 1}: return to a random bit, |2> read as a
          random bit (M(0.5) is the §70b model)
Regimes: T1-dominated (p2 = 0.0005, gamma = 0.003, q = 0.002) and
depolarizing (p2 = 0.004, gamma = 0.0005, q = 0.002); ell = 0.002,
r = 0.05. d = 3 (3 rounds), and d = 5 (5 rounds) in the T1-dominated
regime for M0, M(0), M(0.5), M(1).

Decoders (MWPM on the §64 detector graph; flags from the final readout):
  standard         leakage ignored
  erasure          flagged final-readout edges at p = 1/2 (§70b "final flags")
  bayes            the rule above
  qg               the §64 T1 reweighting, switched by the §64 witness
  qg + erasure, qg + bayes

Pre-registered predictions (written and committed before the seed-711 run;
the code was debugged on seed 1 with small shot counts):
  P1  bayes is never worse than standard (paired z > -3) in any model,
      regime and distance, including M0, where erasure is expected to be
      worse (z <= -3).
  P2  bayes beats erasure (z >= 3) for a <= 0.25 (M0, M(0), M(0.25)) at
      d = 3 in both regimes, and |z| < 3 at a = 1.
  P3  qg + bayes is never worse than qg + erasure (z > -3) anywhere.
  P4  distance: in the T1-dominated regime, M(0.5), qg + bayes beats the
      better of erasure and bayes by at least 1.3x with z >= 3 at d = 5 as
      well as at d = 3 (the qutrit + qg gain does not vanish with distance).
All four must hold for the rule to count as confirmed.

Needs pymatching.

Findings (python examples/qutrit_bayes_weight_qg.py):

FINDINGS_PLACEHOLDER
"""

import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import surface_code_circuit_t1_qg as SC  # noqa: E402


def simulate(code, rounds, shots, rng, p2, gamma, q, ell, r_seep, a, ret, read2, logical=0):
    """ret, read2 in {"one", "random"}. Returns (meas, final bits, final |2> flags)."""
    nd, na = code.nd, code.na
    st = np.zeros((shots, code.nq), dtype=bool)
    st[:, :nd] = code.random_codewords(shots, rng, logical)
    leak = np.zeros((shots, nd), dtype=bool)
    meas = np.zeros((shots, rounds, na), dtype=bool)
    g_idle, g_m = gamma, 2 * gamma

    def decay(cols, g):
        if g > 0:
            cols = np.asarray(cols)
            hit = rng.random((shots, len(cols))) < g
            dm = cols < nd  # a leaked data qubit does not decay
            if dm.any():
                hit[:, dm] &= ~leak[:, cols[dm]]
            st[:, cols] = st[:, cols] & ~hit

    def seep():
        if r_seep > 0:
            back = leak & (rng.random((shots, nd)) < r_seep)
            val = np.ones((shots, nd), bool) if ret == "one" else rng.random((shots, nd)) < 0.5
            st[:, :nd] = np.where(back, val, st[:, :nd])
            leak[back] = False

    for op in SC.ops_list(code, rounds):
        if op[0] == "reset":
            st[:, nd:] = False
            if q > 0:
                st[:, nd:] = rng.random((shots, na)) < q
            decay(list(range(nd)), g_idle)
            seep()
        elif op[0] == "cx":
            busy = set()
            for c, tq in op[3]:
                rand = rng.random(shots) < 0.5
                st[:, tq] ^= np.where(leak[:, c], rand, st[:, c])
                busy |= {c, tq}
                if p2 > 0:
                    pick = rng.integers(1, 16, size=shots)
                    err = rng.random(shots) < p2
                    pa, pb = pick // 4, pick % 4
                    st[:, c] ^= err & ((pa == 1) | (pa == 2)) & ~leak[:, c]
                    st[:, tq] ^= err & ((pb == 1) | (pb == 2))
                if gamma > 0:
                    hit = rng.random(shots) < gamma
                    st[:, c] &= ~(hit & ~leak[:, c])
                    st[:, tq] &= ~(rng.random(shots) < gamma)
                if ell > 0:
                    pl = np.where(st[:, c], ell, a * ell)
                    leak[:, c] |= ~leak[:, c] & (rng.random(shots) < pl)
            idle = [x for x in range(code.nq) if x not in busy]
            if idle:
                decay(idle, g_idle)
            seep()
        elif op[0] == "meas":
            decay(list(range(nd, code.nq)), g_m)
            decay(list(range(nd)), g_m)
            out = st[:, nd:].copy()
            if q > 0:
                out ^= rng.random((shots, na)) < q
            meas[:, op[1]] = out
        elif op[0] == "final":
            decay(list(range(nd)), g_m)
            v2 = np.ones((shots, nd), bool) if read2 == "one" else rng.random((shots, nd)) < 0.5
            final = np.where(leak, v2, st[:, :nd])
            if q > 0:
                final = final ^ (rng.random((shots, nd)) < q)
    return meas, final, leak.copy()


def p_flag(a, q):
    """Flip probability of a flagged qubit's reported bit (reported as 1)."""
    return max(a / (1 + a), q)


def build(E, q, mult=None, flagged=(), pf=0.5):
    """§64 matching with optional T1 multipliers, and flagged qubits whose
    final-readout faults get probability pf (pf = 1/2: erasure)."""
    import pymatching

    p = E["p"].copy()
    tq = E["q"]
    if mult is not None:
        has = tq >= 0
        m = np.ones_like(p)
        m[has] = mult[tq[has]]
        dec = np.where(E["final"], p - q, p)
        p = np.where(E["final"], q + dec * m, p * m)
    if len(flagged):
        sel = E["final"] & np.isin(tq, list(flagged))
        p[sel] = pf
    p = np.clip(p, 1e-12, 0.4999)
    logs = np.bincount(E["edge"], weights=np.log(1 - 2 * p), minlength=len(E["keys"]))
    pe = (1 - np.exp(logs)) / 2
    best = {}
    for k, (u, v, lg) in enumerate(E["keys"]):
        if (u, v) not in best or pe[k] > pe[best[(u, v)]]:
            best[(u, v)] = k
    mt = pymatching.Matching()
    for (u, v), k in best.items():
        pk = float(min(max(pe[k], 1e-12), 0.4999))
        w = math.log((1 - pk) / pk)
        fid = {0} if E["keys"][k][2] else set()
        if v < 0:
            mt.add_boundary_edge(u, weight=w, fault_ids=fid, error_probability=pk)
        else:
            mt.add_edge(u, v, weight=w, fault_ids=fid, error_probability=pk)
    return mt


DECODERS = ("standard", "erasure", "bayes", "qg", "qg + erasure", "qg + bayes")


def run(model, p2, gamma, q, ell=0.002, r_seep=0.05, d=3, rounds=None, shots=60000, seed=711):
    a, ret, read2 = model
    rounds = rounds or d
    rng = np.random.default_rng(seed)
    code = SC.Code(d)
    E = SC.prepare_edges(SC.fault_list(code, rounds, p2, gamma, q))
    wit = SC.witness_ratio(d, rounds, p2, gamma, q)
    use_t1 = wit > 1
    pf = p_flag(a, q)
    std = build(E, q)
    cache = {}

    def get(key, **kw):
        if key not in cache:
            cache[key] = build(E, q, **kw)
        return cache[key]

    fails = {k: [] for k in DECODERS}
    leaked = 0.0
    for logical in (0, 1):
        meas, final, flag = simulate(code, rounds, shots // 2, rng, p2, gamma, q, ell, r_seep, a, ret, read2, logical)
        fb = final | flag  # bayes: a flagged qubit is reported as 1
        det, lg = SC.detectors(code, meas, final)
        det_b, lg_b = SC.detectors(code, meas, fb)
        leaked += flag.any(axis=1).mean() / 2
        n = len(det)
        f = {k: np.zeros(n, bool) for k in DECODERS}
        f["standard"] = std.decode_batch(det)[:, 0].astype(bool) ^ lg ^ bool(logical)
        groups = defaultdict(list)
        for i in range(n):
            fl = tuple(np.nonzero(flag[i])[0].tolist())
            groups[(fl, tuple(final[i].astype(np.uint8)) if use_t1 else None)].append(i)
        for (fl, bits), idx in groups.items():
            idx = np.array(idx)
            dd, ll = det[idx], lg[idx] ^ bool(logical)
            db, lb = det_b[idx], lg_b[idx] ^ bool(logical)
            if fl:
                f["erasure"][idx] = get(("e", fl), flagged=fl, pf=0.5).decode_batch(dd)[:, 0].astype(bool) ^ ll
                f["bayes"][idx] = get(("b", fl), flagged=fl, pf=pf).decode_batch(db)[:, 0].astype(bool) ^ lb
            else:
                f["erasure"][idx] = f["standard"][idx]
                f["bayes"][idx] = f["standard"][idx]
            if use_t1:
                mult = np.where(np.array(bits) == 0, 2.0, 2.0 * q)
                f["qg"][idx] = get(("g", bits), mult=mult).decode_batch(dd)[:, 0].astype(bool) ^ ll
                if fl:
                    m2 = mult.copy()
                    m2[list(fl)] = 1.0
                    f["qg + erasure"][idx] = get(("ge", fl, bits), mult=m2, flagged=fl, pf=0.5).decode_batch(dd)[:, 0].astype(bool) ^ ll
                    f["qg + bayes"][idx] = get(("gb", fl, bits), mult=m2, flagged=fl, pf=pf).decode_batch(db)[:, 0].astype(bool) ^ lb
                else:
                    f["qg + erasure"][idx] = f["qg"][idx]
                    f["qg + bayes"][idx] = f["qg"][idx]
            else:
                f["qg"][idx] = f["standard"][idx]
                f["qg + erasure"][idx] = f["erasure"][idx]
                f["qg + bayes"][idx] = f["bayes"][idx]
            if len(cache) > 20000:  # d = 5: almost every shot has its own key
                cache.clear()
        for k in DECODERS:
            fails[k].append(f[k])
    fails = {k: np.concatenate(v) for k, v in fails.items()}
    out = {k: float(v.mean()) for k, v in fails.items()}

    def z(x, y):  # positive when y fails less than x
        n1 = int(np.sum(fails[x] & ~fails[y]))
        n2 = int(np.sum(fails[y] & ~fails[x]))
        return (n1 - n2) / math.sqrt(n1 + n2) if n1 + n2 else 0.0

    out["z_bayes_vs_std"] = z("standard", "bayes")
    out["z_erasure_vs_std"] = z("standard", "erasure")
    out["z_bayes_vs_erasure"] = z("erasure", "bayes")
    out["z_qgb_vs_qge"] = z("qg + erasure", "qg + bayes")
    best = "bayes" if out["bayes"] <= out["erasure"] else "erasure"
    out["best_flag"] = best
    out["ratio_P4"] = out[best] / max(out["qg + bayes"], 1e-12)
    out["z_P4"] = z(best, "qg + bayes")
    out["witness"] = wit
    out["leaked_at_end"] = float(leaked)
    out["p_flag"] = pf
    out["shots"] = len(fails["standard"])
    return out


MODELS = {
    "M0": (0.0, "one", "one"),
    "M(0)": (0.0, "random", "random"),
    "M(0.25)": (0.25, "random", "random"),
    "M(0.5)": (0.5, "random", "random"),
    "M(1)": (1.0, "random", "random"),
}
REGIMES = {
    "T1-dominated": dict(p2=0.0005, gamma=0.003, q=0.002),
    "depolarizing": dict(p2=0.004, gamma=0.0005, q=0.002),
}
D5_MODELS = ("M0", "M(0)", "M(0.5)", "M(1)")


def plan():
    runs = [(m, r, 3) for r in REGIMES for m in MODELS]
    runs += [(m, "T1-dominated", 5) for m in D5_MODELS]
    return runs


def verdict(res):
    """res: {(model, regime, d): run output}. Returns the four predictions."""
    p1 = all(r["z_bayes_vs_std"] > -3 for r in res.values())
    p2 = all(res[(m, g, 3)]["z_bayes_vs_erasure"] >= 3 for m in ("M0", "M(0)", "M(0.25)") for g in REGIMES) and all(
        abs(res[("M(1)", g, 3)]["z_bayes_vs_erasure"]) < 3 for g in REGIMES)
    p3 = all(r["z_qgb_vs_qge"] > -3 for r in res.values())
    p4 = all(res[("M(0.5)", "T1-dominated", d)]["ratio_P4"] >= 1.3 and res[("M(0.5)", "T1-dominated", d)]["z_P4"] >= 3 for d in (3, 5))
    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4}


def main(shots3=60000, shots5=60000, seed=711):
    res = {}
    for m, g, d in plan():
        r = run(MODELS[m], d=d, shots=shots3 if d == 3 else shots5, seed=seed, **REGIMES[g])
        res[(m, g, d)] = r
        print(f"d={d} {g:13s} {m:8s} (p_flag {r['p_flag']:.3f}, witness {r['witness']:.2f}, leaked at end {r['leaked_at_end']:.3f})")
        print("   " + " | ".join(f"{k} {r[k]:.4f}" for k in DECODERS))
        print(f"   z: bayes vs std {r['z_bayes_vs_std']:+.1f}, erasure vs std {r['z_erasure_vs_std']:+.1f},"
              f" bayes vs erasure {r['z_bayes_vs_erasure']:+.1f}, qg+bayes vs qg+erasure {r['z_qgb_vs_qge']:+.1f};"
              f" qg+bayes over best flag decoder ({r['best_flag']}) {r['ratio_P4']:.2f}x (z {r['z_P4']:+.1f})", flush=True)
    v = verdict(res)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return res, v


if __name__ == "__main__":
    main()
