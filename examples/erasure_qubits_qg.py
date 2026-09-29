"""
Does erasure conversion make the qg T1 decoder obsolete? (§73)

The qg decoder of §64 exploits the bias T1 leaves in the final data readout
(a qubit read as 1 has not decayed). Erasure qubits (dual-rail transmons,
metastable-atom qubits with erasure conversion) instead HERALD decay
events as they happen, so the decoder knows where and when they occurred.
If heralding were perfect, the final-readout bias would carry no extra
information. The question: how much of the qg gain survives as the
heralding efficiency h grows?

Model: the §64 circuit-level Z-memory (T1-dominated regime: p2 = 0.0005,
gamma = 0.003, q = 0.002), d = 3 (3 rounds) and d = 5 (5 rounds). Every
decay that takes a DATA qubit from 1 to 0 is heralded with probability h,
with its exact location (qubit, circuit step). Ancilla decays are not
heralded. Heralding efficiency h in {0, 0.5, 0.9, 0.99}.

Decoders (minimum-weight matching on the §64 detector graph):
  standard      heralds ignored
  erasure       the graph edge of each heralded decay set to p = 1/2
  qg            the §64 T1 reweighting (switched by the §64 witness)
  qg + erasure  both

Pre-registered predictions (written and committed before the seed-73 run,
60 000 shots per point; code debugged on seed 1 with small shot counts):
  E1  at h = 0 heralds carry nothing: qg + erasure beats erasure by at
      least 1.5x (z >= 3) at d = 3 (the §64 gain).
  E2  the gain of qg + erasure over erasure decreases monotonically with h
      (point estimates) at each distance.
  E3  at h = 0.99 qg is redundant: qg + erasure over erasure below 1.1x at
      both distances.
  E4  qg + erasure is never worse than erasure (paired z > -3) anywhere.

Needs pymatching.

Findings (python examples/erasure_qubits_qg.py):

FINDINGS_PLACEHOLDER
"""

import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qutrit_leakage_qg as Q  # noqa: E402
import surface_code_circuit_t1_qg as SC  # noqa: E402

REGIME = dict(p2=0.0005, gamma=0.003, q=0.002)
HS = (0.0, 0.5, 0.9, 0.99)


def edge_table(code, rounds):
    """(op index, data qubit) -> detector-graph edge (u, v) of a flip there."""
    tab = {}
    for k, op in enumerate(SC.ops_list(code, rounds)):
        if op[0] in ("reset", "cx", "meas", "final"):
            for x in range(code.nd):
                if op[0] == "final":
                    dets, _ = SC._propagate(code, rounds, k, [], final_flip=x)
                else:
                    dets, _ = SC._propagate(code, rounds, k, [x])
                if 0 < len(dets) <= 2:
                    u = dets[0]
                    v = dets[1] if len(dets) == 2 else -1
                    tab[(k, x)] = (min(u, v), max(u, v)) if v >= 0 else (u, -1)
    return tab


def simulate(code, rounds, shots, rng, p2, gamma, q, h, logical=0):
    """§64 memory; returns (meas, final, heralds) with heralds[i] = list of
    (op index, data qubit) of heralded 1 -> 0 decays."""
    nd, na = code.nd, code.na
    st = np.zeros((shots, code.nq), dtype=bool)
    st[:, :nd] = code.random_codewords(shots, rng, logical)
    meas = np.zeros((shots, rounds, na), dtype=bool)
    heralds = [[] for _ in range(shots)]
    g_idle, g_m = gamma, 2 * gamma

    def decay(cols, g, k, flip_as_final=False):
        if g <= 0:
            return
        cols = np.asarray(cols)
        hit = rng.random((shots, len(cols))) < g
        eff = hit & st[:, cols]
        st[:, cols] = st[:, cols] & ~hit
        dm = cols < nd
        if h > 0 and dm.any():
            her = eff[:, dm] & (rng.random((shots, int(dm.sum()))) < h)
            for i, j in zip(*np.nonzero(her)):
                heralds[i].append((k, int(cols[dm][j])))

    ops = SC.ops_list(code, rounds)
    for k, op in enumerate(ops):
        if op[0] == "reset":
            st[:, nd:] = False
            if q > 0:
                st[:, nd:] = rng.random((shots, na)) < q
            decay(list(range(nd)), g_idle, k)
        elif op[0] == "cx":
            busy = set()
            for c, tq in op[3]:
                st[:, tq] ^= st[:, c]
                busy |= {c, tq}
                if p2 > 0:
                    pick = rng.integers(1, 16, size=shots)
                    err = rng.random(shots) < p2
                    pa, pb = pick // 4, pick % 4
                    st[:, c] ^= err & ((pa == 1) | (pa == 2))
                    st[:, tq] ^= err & ((pb == 1) | (pb == 2))
                decay([c, tq], gamma, k)
            idle = [x for x in range(code.nq) if x not in busy]
            if idle:
                decay(idle, g_idle, k)
        elif op[0] == "meas":
            decay(list(range(nd, code.nq)), g_m, k)
            decay(list(range(nd)), g_m, k)
            out = st[:, nd:].copy()
            if q > 0:
                out ^= rng.random((shots, na)) < q
            meas[:, op[1]] = out
        elif op[0] == "final":
            decay(list(range(nd)), g_m, k)
            final = st[:, :nd].copy()
            if q > 0:
                final ^= rng.random((shots, nd)) < q
    return meas, final, heralds


DECODERS = ("standard", "erasure", "qg", "qg + erasure")


def run(h, d=3, rounds=None, shots=60000, seed=73, p2=0.0005, gamma=0.003, q=0.002):
    rounds = rounds or d
    rng = np.random.default_rng(seed)
    code = SC.Code(d)
    E = SC.prepare_edges(SC.fault_list(code, rounds, p2, gamma, q))
    tab = edge_table(code, rounds)
    wit = SC.witness_ratio(d, rounds, p2, gamma, q)
    use_t1 = wit > 1
    std = Q.build(E, q)
    cache = {}

    def get(key, mult, erased):
        if key not in cache:
            cache[key] = Q.build(E, q, mult, [erased] if erased else ())
            if len(cache) > 20000:
                cache.clear()
                cache[key] = Q.build(E, q, mult, [erased] if erased else ())
        return cache[key]

    fails = {k: [] for k in DECODERS}
    n_her = 0.0
    for logical in (0, 1):
        meas, final, her = simulate(code, rounds, shots // 2, rng, p2, gamma, q, h, logical)
        det, lg = SC.detectors(code, meas, final)
        n_her += sum(len(x) for x in her) / shots
        n = len(det)
        f = {k: np.zeros(n, bool) for k in DECODERS}
        f["standard"] = std.decode_batch(det)[:, 0].astype(bool) ^ lg ^ bool(logical)
        groups = defaultdict(list)
        for i in range(n):
            er = frozenset(tab[kx] for kx in her[i] if kx in tab)
            groups[(er, tuple(final[i].astype(np.uint8)) if use_t1 else None)].append(i)
        for (er, bits), idx in groups.items():
            idx = np.array(idx)
            dd, ll = det[idx], lg[idx] ^ bool(logical)
            f["erasure"][idx] = (get(("e", er), None, er).decode_batch(dd)[:, 0].astype(bool) ^ ll) if er else f["standard"][idx]
            if use_t1:
                mult = np.where(np.array(bits) == 0, 2.0, 2.0 * q)
                f["qg"][idx] = get(("g", bits), mult, frozenset()).decode_batch(dd)[:, 0].astype(bool) ^ ll
                f["qg + erasure"][idx] = (get(("ge", er, bits), mult, er).decode_batch(dd)[:, 0].astype(bool) ^ ll) if er else f["qg"][idx]
            else:
                f["qg"][idx] = f["standard"][idx]
                f["qg + erasure"][idx] = f["erasure"][idx]
        for k in DECODERS:
            fails[k].append(f[k])
    fails = {k: np.concatenate(v) for k, v in fails.items()}
    out = {k: float(v.mean()) for k, v in fails.items()}

    def z(a, b):  # positive when b fails less than a
        n1 = int(np.sum(fails[a] & ~fails[b]))
        n2 = int(np.sum(fails[b] & ~fails[a]))
        return (n1 - n2) / math.sqrt(n1 + n2) if n1 + n2 else 0.0

    out["gain"] = out["erasure"] / max(out["qg + erasure"], 1e-12)
    out["z_gain"] = z("erasure", "qg + erasure")
    out["z_erasure_vs_std"] = z("standard", "erasure")
    out["witness"] = wit
    out["heralds_per_shot"] = n_her
    return out


def verdict(res):
    e1 = res[(0.0, 3)]["gain"] >= 1.5 and res[(0.0, 3)]["z_gain"] >= 3
    e2 = all(all(res[(HS[i], d)]["gain"] > res[(HS[i + 1], d)]["gain"] for i in range(len(HS) - 1)) for d in (3, 5))
    e3 = all(res[(0.99, d)]["gain"] < 1.1 for d in (3, 5))
    e4 = all(r["z_gain"] > -3 for r in res.values())
    return {"E1": e1, "E2": e2, "E3": e3, "E4": e4}


def main(shots=60000, seed=73):
    res = {}
    for d in (3, 5):
        for h in HS:
            r = run(h, d=d, shots=shots, seed=seed, **REGIME)
            res[(h, d)] = r
            print(f"d={d} h={h:<4} (witness {r['witness']:.2f}, heralds/shot {r['heralds_per_shot']:.3f}): "
                  + " | ".join(f"{k} {r[k]:.4f}" for k in DECODERS)
                  + f" || qg+erasure over erasure {r['gain']:.2f}x (z {r['z_gain']:+.1f}); erasure vs std z {r['z_erasure_vs_std']:+.1f}",
                  flush=True)
    v = verdict(res)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    return res, v


if __name__ == "__main__":
    main()
