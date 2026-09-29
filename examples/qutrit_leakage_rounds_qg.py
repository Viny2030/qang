"""
The qutrit test, second round (§70b): leakage from both levels, per-round
leakage flags, and a pre-registered criterion against the BEST known
leakage-aware decoder.

§70 found no qutrit advantage, but in a model where a leaked qubit carries
its value (leaks only from |1>, returns to |1>), so the |2> flag carried no
information. An exploratory rerun (leak from |0> or |1>, random return)
showed the flag helping. This test fixes the model and the criterion
before the run.

Model (the §64 circuit-level Z-memory, d = 3, 3 rounds, plus leakage):
  * after each CNOT a non-leaked data qubit leaks: from |1> with
    probability ell, from |0> with probability ell/2 (asymmetric, as in
    transmons where |1> -> |2> dominates);
  * a leaked control kicks its ancilla at random (probability 1/2);
  * a leaked qubit returns with probability r per layer, to a RANDOM bit;
  * per-round leakage detection: at each syndrome round a leaked data qubit
    is flagged with probability h (herald efficiency, e.g. a leakage-
    detection unit or ancilla-pattern classifier);
  * the final three-level readout flags every leaked data qubit and reads
    |2> as a random bit.
Other noise as §64: two-qubit depolarizing p2, decay gamma, readout q.

Decoders (MWPM on the §64 detector graph; erasure = p set to 1/2):
  standard          leakage ignored
  final flags       erase the final-readout layer of qubits flagged at the end
  round flags       also use the per-round flags: a qubit flagged in rounds
                    R has its edges erased in the detector layers
                    [min R - 1, max R + 1] (plus the final layer if flagged
                    at the end)
  qg, no flags      the §64 T1 reweighting alone (switched by the witness)
  qg + round flags  round flags + the §64 T1 reweighting (the qang decoder)

Pre-registered criterion (written before the run; the code was debugged on
seed 1 with 4000 shots, the reported run uses seed 70 and 60 000 shots):
  (A) qang gains from the qutrit setting only if "qg + round flags" beats
      the better of the two non-qang leakage-aware decoders (final flags,
      round flags) by at least 1.3x in logical error with paired z >= 3 in
      at least one regime, and is never worse with z <= -3;
  (B) and the qutrit information adds to qang: "qg + round flags" beats
      "qg, no flags" with z >= 3 in every regime where leakage is at least
      as strong as in the T1-dominated one.
  Both must hold. Even then the gain is attributed separately: the flag
  part is known erasure decoding, the reweighting part is §64; only if the
  combination beats both parts is it reported as a qang + qutrit result.

Needs pymatching.

Findings (python examples/qutrit_leakage_rounds_qg.py):

  Logical error, d = 3, 3 rounds, seed 70, 60 000 shots per regime (same
  shots for every decoder), herald efficiency h = 0.8:

    regime (witness)              standard  final   round   qg, no   qg + round
                                            flags   flags   flags    flags
    leakage + T1-dominated (1.68)  0.0189   0.0178  0.0177  0.0130   0.0106
    leakage + mixed (1.27)         0.0127   0.0114  0.0114  0.0100   0.0089
    leakage + depolarizing (0.34)  0.0052   0.0043  0.0044  0.0052   0.0044
    strong leakage, depol. (0.46)  0.0075   0.0047  0.0046  0.0075   0.0046
  Shots ending with a leaked data qubit: 3.6 % (10.8 % strong leakage).

  * Criterion A passes: qg + round flags beats the best known leakage-aware
    decoder 1.68x (z = +15.3) in the T1-dominated regime, 1.28x (z = +8.0)
    in the mixed one, and ties where the witness switches the reweighting
    off (0.97x, z = -1.5; 1.00x).
  * Criterion B passes: the flags add to qg in every regime (z = +8.8,
    +5.3, +4.0, +8.9).
  * Attribution. The combination beats both of its parts: in the
    T1-dominated regime flags alone give 1.07x over standard, the §64
    reweighting alone 1.45x, together 1.78x, more than the product (1.55x).
    Plausible mechanism: the reweighting trusts the final data bits, and a
    leaked qubit's final bit is random; erasing it removes a misleading
    input. So the flag helps qg more (z = +8.8) than it helps standard
    decoding (z = +4.2).
  * Replication (seed 71): 1.43x (z = +11.3) and 1.20x (z = +6.2); the
    effect holds, smaller than on seed 70.
  * Per-round detection adds almost nothing at d = 3: with h = 0 (only the
    final three-level readout) the gain is 1.39x instead of 1.43x (seed 71).
    The useful qutrit information is the final |2> readout.
  * Relation to §70: the flag is useless when a leaked qubit keeps its value
    (leak from |1> only, return to |1>) and useful when it does not. This
    model was chosen after §70's exploratory rerun, which is why the
    criterion was fixed and committed before this run.

  Verdict. With leakage that scrambles the qubit value, qang gains from a
  three-level readout: the |2> flag and the qg T1 reweighting combine
  better than either alone. The qg quantities themselves stay qubit
  quantities; the qutrit enters as an input to the qg decoder, not as a new
  qg. Limitations: d = 3, one leakage model (asymmetric leak 1:2, random
  return), erasure weights not tuned, no leakage-reduction units.
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


def simulate(code, rounds, shots, rng, p2, gamma, q, ell, r_seep, h, logical=0):
    """Returns (meas, final bits, round flags [shots, rounds, nd], final flags [shots, nd])."""
    nd, na = code.nd, code.na
    st = np.zeros((shots, code.nq), dtype=bool)
    st[:, :nd] = code.random_codewords(shots, rng, logical)
    leak = np.zeros((shots, nd), dtype=bool)
    meas = np.zeros((shots, rounds, na), dtype=bool)
    rflag = np.zeros((shots, rounds, nd), dtype=bool)
    g_idle, g_m = gamma, 2 * gamma

    def decay(cols, g):
        if g > 0:
            cols = np.asarray(cols)
            hit = rng.random((shots, len(cols))) < g
            dm = cols < nd  # a leaked data qubit does not decay to 0
            if dm.any():
                hit[:, dm] &= ~leak[:, cols[dm]]
            st[:, cols] = st[:, cols] & ~hit

    def seep():
        if r_seep > 0:
            back = leak & (rng.random((shots, nd)) < r_seep)
            st[:, :nd] = np.where(back, rng.random((shots, nd)) < 0.5, st[:, :nd])
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
                    pl = np.where(st[:, c], ell, ell / 2)
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
            rflag[:, op[1]] = leak & (rng.random((shots, nd)) < h)
        elif op[0] == "final":
            decay(list(range(nd)), g_m)
            final = np.where(leak, rng.random((shots, nd)) < 0.5, st[:, :nd])
            if q > 0:
                final = final ^ (rng.random((shots, nd)) < q)
    return meas, final, rflag, leak.copy()


def layer_of(e, na):
    u, v = e
    return {u // na} | ({v // na} if v >= 0 else set())


def erasure_sets(code, rounds):
    """Per data qubit and detector layer: the edges touching that layer."""
    full = Q.qubit_edges(code, rounds)
    na = code.na
    out = {}
    for x, edges in full.items():
        per = defaultdict(set)
        for e in edges:
            for L in layer_of(e, na):
                per[L].add(e)
        out[x] = per
    return out


def erased_edges(per, rounds, rflags_x, fflag_x, use_rounds):
    """Edges to erase for one qubit given its flags."""
    layers = set()
    if fflag_x:
        layers |= {rounds, rounds - 1}
    if use_rounds:
        R = [t for t in range(rounds) if rflags_x[t]]
        if R:
            layers |= set(range(max(min(R) - 1, 0), min(max(R) + 1, rounds) + 1))
    s = set()
    for L in layers:
        s |= per.get(L, set())
    return frozenset(s)


DECODERS = ("standard", "final flags", "round flags", "qg, no flags", "qg + round flags")


def run(p2, gamma, q, ell, r_seep, h, shots=60000, d=3, rounds=3, seed=70):
    rng = np.random.default_rng(seed)
    code = SC.Code(d)
    E = SC.prepare_edges(SC.fault_list(code, rounds, p2, gamma, q))
    per = erasure_sets(code, rounds)
    wit = SC.witness_ratio(d, rounds, p2, gamma, q)
    use_t1 = wit > 1
    cache = {}

    def dec(erase, bits):
        key = (erase, bits)
        if key not in cache:
            mult = None if bits is None else np.where(np.array(bits) == 0, 2.0, 2.0 * q)
            cache[key] = Q.build(E, q, mult, [erase] if erase else ())
        return cache[key]

    fails = {k: [] for k in DECODERS}
    leaked_end = 0.0
    for logical in (0, 1):
        meas, final, rflag, fflag = simulate(code, rounds, shots // 2, rng, p2, gamma, q, ell, r_seep, h, logical)
        det, lg = SC.detectors(code, meas, final)
        leaked_end += fflag.any(axis=1).mean() / 2
        f = {k: np.zeros(len(det), bool) for k in DECODERS}
        groups = defaultdict(list)
        for i in range(len(det)):
            ef = frozenset().union(*[erased_edges(per[x], rounds, rflag[i, :, x], fflag[i, x], False) for x in range(code.nd)])
            er = frozenset().union(*[erased_edges(per[x], rounds, rflag[i, :, x], fflag[i, x], True) for x in range(code.nd)])
            bits = tuple(final[i].astype(np.uint8)) if use_t1 else None
            groups[(ef, er, bits)].append(i)
        for (ef, er, bits), idx in groups.items():
            plan = {"standard": (frozenset(), None), "final flags": (ef, None), "round flags": (er, None),
                    "qg, no flags": (frozenset(), bits), "qg + round flags": (er, bits)}
            for name, (es, bk) in plan.items():
                m = dec(es, bk)
                f[name][idx] = m.decode_batch(det[idx])[:, 0].astype(bool) ^ lg[idx] ^ bool(logical)
        for k in DECODERS:
            fails[k].append(f[k])
    fails = {k: np.concatenate(v) for k, v in fails.items()}
    out = {k: float(v.mean()) for k, v in fails.items()}

    def z(a, b):  # positive when b fails less than a
        n1 = int(np.sum(fails[a] & ~fails[b]))
        n2 = int(np.sum(fails[b] & ~fails[a]))
        return (n1 - n2) / math.sqrt(n1 + n2) if n1 + n2 else 0.0

    best = "final flags" if out["final flags"] <= out["round flags"] else "round flags"
    out["best_known"] = best
    out["ratio_A"] = out[best] / max(out["qg + round flags"], 1e-12)
    out["z_A"] = z(best, "qg + round flags")
    out["z_B"] = z("qg, no flags", "qg + round flags")
    out["z_flags_vs_std"] = z("standard", "round flags")
    out["witness"] = wit
    out["leaked_at_end"] = float(leaked_end)
    return out


# same noise regimes as §70; herald efficiency h = 0.8 per round
CASES = [(name, dict(kw, h=0.8)) for name, kw in Q.CASES]
STRONG = {name for name, kw in CASES if kw["ell"] >= 0.002}  # all four regimes


def verdict(rows):
    a = any(r["ratio_A"] >= 1.3 and r["z_A"] >= 3 for _, r in rows) and all(r["z_A"] > -3 for _, r in rows)
    b = all(r["z_B"] >= 3 for name, r in rows if name in STRONG)
    return a, b


def main(shots=60000, seed=70):
    rows = []
    for name, kw in CASES:
        r = run(shots=shots, seed=seed, **kw)
        rows.append((name, r))
        print(f"{name} (witness {r['witness']:.2f}, shots ending leaked {r['leaked_at_end']:.3f})")
        for k in DECODERS:
            print(f"  {k:18s} p_L {r[k]:.4f}")
        print(f"  (A) best known = {r['best_known']}: ratio {r['ratio_A']:.2f}, z {r['z_A']:+.1f} | "
              f"(B) flags added to qg: z {r['z_B']:+.1f} | round flags vs standard z {r['z_flags_vs_std']:+.1f}")
    a, b = verdict(rows)
    print(f"criterion A: {'PASS' if a else 'FAIL'} | criterion B: {'PASS' if b else 'FAIL'}")
    return rows, a, b


if __name__ == "__main__":
    main()
