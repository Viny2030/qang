"""
The qutrit test: does treating the transmon as a qutrit give qang an
advantage, when leakage to |2> is present?

Question asked before extending qang beyond qubits. Leakage detection is
known to help decoding (erasure conversion, leakage-aware decoders), so a
gain of "seeing |2>" over "not seeing it" is not a qang result. The test is
whether qang's own tools add something ON TOP of standard leakage handling.

Model: the §64 circuit-level Z-memory (rotated surface code, d = 3, 3
rounds), extended with leakage of the data qubits:
  * after each CNOT a data qubit in |1> leaks to |2> with probability ell;
  * a leaked data qubit used as CNOT control flips its ancilla at random
    (probability 1/2), and returns to |1> with probability r per layer
    (decay 2 -> 1);
  * the final readout reads |2> as 1 (two-level discrimination) and, with a
    three-level readout, also flags it.
Other noise as §64: two-qubit depolarizing p2, decay gamma, readout q.

Decoders (minimum-weight matching on the §64 detector graph):
  standard        leakage ignored
  leakage-aware   the standard known method: for data qubits flagged as |2>
                  by the final three-level readout, their data-flip and
                  adjacent measurement edges are set to probability 1/2
                  (erasure)
  qg              leakage-aware + the §64 T1 reweighting by the final data
                  readout, switched on only when the §64 witness (register-
                  mean qg_Z / detector rate, from a calibration batch)
                  exceeds 1

Pre-registered criterion (written before the run, as agreed): qang gains
from the qutrit setting only if the qg decoder beats the leakage-aware
decoder by at least 1.3x in logical error, with a paired z >= 3, in at
least one leakage regime, and is never significantly worse. Otherwise qang
stays qubit-only and this is recorded as a negative result.

Needs pymatching.

Findings (python examples/qutrit_leakage_qg.py):

  Logical error, d = 3, 3 rounds, 60 000 shots per regime (same shots for
  every decoder); "flags" = data qubits read as |2> by the final three-level
  readout.

    regime (witness)              standard  leak-aware  qg      qg, no flags
    leakage + T1-dominated (1.68)  0.0156    0.0165      0.0100  0.0093
    leakage + mixed (1.27)         0.0099    0.0106      0.0089  0.0084
    leakage + depolarizing (0.34)  0.0032    0.0036      0.0036  0.0032
    strong leakage, depol. (0.46)  0.0021    0.0032      0.0032  0.0021
  Shots ending with a leaked data qubit: 2.4 % (7.3 % strong leakage).

  * As written, the criterion passes: qg beats the leakage-aware decoder
    1.65x (z = +14.8) in the T1-dominated regime and is never worse. The
    pass is an artefact and is NOT taken as a qutrit advantage:
      - the pre-registered leakage-aware baseline is worse than ignoring
        leakage in every regime (z = -3.8 to -6.0);
      - all of the gain comes from the §64 T1 reweighting, a qubit tool: qg
        WITHOUT any leakage flag beats standard 1.68x (z = +14.7), and
        adding the |2> flags to it makes it worse (z = -4.7, -3.0, -2.8,
        -4.9 across the four regimes).
  * Deviation, added after the first run: erasing only the final-readout
    layer of a flagged qubit (instead of its whole history) is less harmful
    but still worse than standard (z = -2.8 to -4.9). Why the flag hurts in
    this model: a qubit leaks only from |1> and seeps back to |1>, so
    reading |2> as 1 is usually the right value; declaring it erased throws
    that information away. The flag comes too late (end of the run) to
    locate the randomised syndromes of earlier rounds.
  * Verdict (as agreed before the run): no qutrit advantage for qang. qang
    stays a qubit construction; this is recorded as a negative result. The
    positive by-product: the §64 T1 reweighting keeps its full gain when
    leakage is present (1.68x here vs 1.5-1.7x in §64).

  What is new and what is not. Leakage-aware and erasure decoding are known
  (e.g. Suchara et al. 2015; Wu et al. 2022, erasure conversion); stronger
  leakage handling uses leakage-reduction units or per-round leakage
  detection, not modelled here. Ours: the test of whether qang's own tools
  gain from a three-level readout, with the answer no. Limitations: d = 3,
  a simple leakage model (leak from |1> after CNOT, random ancilla kick,
  seepage), flags only at the final readout, and erasure weights not tuned.
"""

import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import surface_code_circuit_t1_qg as SC  # noqa: E402


def simulate_leak(code, rounds, shots, rng, p2, gamma, q, ell, r_seep, logical=0):
    """As SC.simulate, plus data-qubit leakage. Returns (meas, final bits, leak flags)."""
    nd, na = code.nd, code.na
    st = np.zeros((shots, code.nq), dtype=bool)
    st[:, :nd] = code.random_codewords(shots, rng, logical)
    leak = np.zeros((shots, nd), dtype=bool)
    meas = np.zeros((shots, rounds, na), dtype=bool)
    g_idle, g_m = gamma, 2 * gamma

    def decay(cols, g):
        if g > 0:
            hit = rng.random((shots, len(cols))) < g
            st[:, cols] = st[:, cols] & ~hit

    def seep():
        if r_seep > 0:
            back = leak & (rng.random((shots, nd)) < r_seep)
            leak[back] = False  # 2 -> 1: the bit is already 1

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
                # decay acts on non-leaked qubits (a leaked qubit decays by seepage)
                if gamma > 0:
                    hit = rng.random(shots) < gamma
                    st[:, c] &= ~(hit & ~leak[:, c])
                    st[:, tq] &= ~(rng.random(shots) < gamma)
                if ell > 0:
                    new = st[:, c] & ~leak[:, c] & (rng.random(shots) < ell)
                    leak[:, c] |= new
            idle = [x for x in range(code.nq) if x not in busy]
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
            final = st[:, :nd] | leak  # |2> read as 1
            if q > 0:
                final = final ^ (rng.random((shots, nd)) < q)
    return meas, final, leak.copy()


def qubit_edges(code, rounds, window=None):
    """For each data qubit: the detector-graph edge keys its flips produce,
    plus the time-like edges of its stabilizers (erasure set). With window=w,
    only edges touching the last w detector layers are kept (window=1: the
    final-readout layer only)."""
    nd, na = code.nd, code.na
    ops = SC.ops_list(code, rounds)
    out = {x: set() for x in range(nd)}
    for k, op in enumerate(ops):
        if op[0] in ("reset", "cx", "meas"):
            for x in range(nd):
                dets, lg = SC._propagate(code, rounds, k, [x])
                if 0 < len(dets) <= 2:
                    u = dets[0]
                    v = dets[1] if len(dets) == 2 else -1
                    out[x].add(((min(u, v), max(u, v)) if v >= 0 else (u, -1)))
    for x in range(nd):
        stabs = [a for a, st in enumerate(code.zstabs) if x in st.values()]
        for a in stabs:
            for t in range(rounds):
                out[x].add((t * na + a, (t + 1) * na + a))
    if window is not None:
        lo = (rounds + 1 - window) * na
        out = {x: {e for e in s if max(e) >= lo or (e[1] == -1 and e[0] >= lo)} for x, s in out.items()}
    return out


def build(E, q, mult=None, erased=()):
    """SC.build_matching with erasure: edges of erased qubits get p = 1/2."""
    import pymatching

    p = E["p"].copy()
    if mult is not None:
        tq = E["q"]
        has = tq >= 0
        m = np.ones_like(p)
        m[has] = mult[tq[has]]
        dec = np.where(E["final"], p - q, p)
        p = np.where(E["final"], q + dec * m, p * m)
    p = np.clip(p, 1e-12, 0.499)
    logs = np.bincount(E["edge"], weights=np.log(1 - 2 * p), minlength=len(E["keys"]))
    pe = (1 - np.exp(logs)) / 2
    best = {}
    for k, (u, v, lg) in enumerate(E["keys"]):
        if (u, v) not in best or pe[k] > pe[best[(u, v)]]:
            best[(u, v)] = k
    erase = set().union(*erased) if erased else set()
    mt = pymatching.Matching()
    for (u, v), k in best.items():
        pk = 0.499 if (u, v) in erase else float(min(max(pe[k], 1e-12), 0.499))
        w = math.log((1 - pk) / pk)
        fid = {0} if E["keys"][k][2] else set()
        if v < 0:
            mt.add_boundary_edge(u, weight=w, fault_ids=fid, error_probability=pk)
        else:
            mt.add_edge(u, v, weight=w, fault_ids=fid, error_probability=pk)
    return mt


DECODERS = ("standard", "leakage-aware", "leakage-aware, last layer", "qg", "qg, no leakage flags", "qg, last layer")


def run(p2, gamma, q, ell, r_seep, shots=40000, d=3, rounds=3, seed=0):
    """Logical error of each decoder on the same shots.
    standard / leakage-aware / qg: as pre-registered (erasure over the whole
    history of a flagged qubit). Added after the first run (deviation, see
    findings): erasure of the final-readout layer only, and the T1
    reweighting without any leakage flag."""
    rng = np.random.default_rng(seed)
    code = SC.Code(d)
    E = SC.prepare_edges(SC.fault_list(code, rounds, p2, gamma, q))
    qe_all = qubit_edges(code, rounds)
    qe_last = qubit_edges(code, rounds, window=1)
    wit = SC.witness_ratio(d, rounds, p2, gamma, q)
    use_t1 = wit > 1
    cache = {}

    def dec(tag, fk, bk, erase_sets):
        key = (tag, fk, bk)
        if key not in cache:
            mult = None
            if bk is not None:
                mult = np.where(np.array(bk) == 0, 2.0, 2.0 * q)
            erased = [erase_sets[x] for x in range(code.nd) if fk[x]] if erase_sets is not None else ()
            cache[key] = build(E, q, mult, erased)
        return cache[key]

    fails = {k: [] for k in DECODERS}
    raw_leak = 0.0
    zero = tuple([0] * code.nd)
    for logical in (0, 1):
        meas, final, flag = simulate_leak(code, rounds, shots // 2, rng, p2, gamma, q, ell, r_seep, logical)
        det, lg = SC.detectors(code, meas, final)
        raw_leak += flag.any(axis=1).mean() / 2
        f = {k: np.zeros(len(det), bool) for k in DECODERS}
        groups = defaultdict(list)
        for i in range(len(det)):
            groups[(tuple(flag[i].astype(np.uint8)), tuple(final[i].astype(np.uint8)))].append(i)
        for (fk, bits), idx in groups.items():
            bk = bits if use_t1 else None
            plan = {
                "standard": ("s", zero, None, None),
                "leakage-aware": ("l", fk, None, qe_all),
                "leakage-aware, last layer": ("L", fk, None, qe_last),
                "qg": ("g", fk, bk, qe_all),
                "qg, no leakage flags": ("G", zero, bk, None),
                "qg, last layer": ("h", fk, bk, qe_last),
            }
            for name, (tag, fkk, bkk, es) in plan.items():
                m = dec(tag, fkk, bkk, es)
                f[name][idx] = m.decode_batch(det[idx])[:, 0].astype(bool) ^ lg[idx] ^ bool(logical)
        for k in DECODERS:
            fails[k].append(f[k])
    fails = {k: np.concatenate(v) for k, v in fails.items()}
    out = {k: float(v.mean()) for k, v in fails.items()}

    def z(a, b):  # paired: positive when b fails less than a
        n1 = int(np.sum(fails[a] & ~fails[b]))
        n2 = int(np.sum(fails[b] & ~fails[a]))
        return (n1 - n2) / math.sqrt(n1 + n2) if n1 + n2 else 0.0

    out["z_qg_vs_leak"] = z("leakage-aware", "qg")
    out["z_leak_vs_std"] = z("standard", "leakage-aware")
    out["z_leaklast_vs_std"] = z("standard", "leakage-aware, last layer")
    out["z_qgnoflag_vs_std"] = z("standard", "qg, no leakage flags")
    out["z_qglast_vs_qgnoflag"] = z("qg, no leakage flags", "qg, last layer")
    out["witness"] = wit
    out["shots_with_final_leak"] = float(raw_leak)
    out["shots"] = len(fails["qg"])
    return out


CASES = [
    ("leakage + T1-dominated", dict(p2=0.0005, gamma=0.003, q=0.002, ell=0.002, r_seep=0.05)),
    ("leakage + mixed", dict(p2=0.002, gamma=0.002, q=0.002, ell=0.002, r_seep=0.05)),
    ("leakage + depolarizing", dict(p2=0.004, gamma=0.0005, q=0.002, ell=0.002, r_seep=0.05)),
    ("strong leakage, depolarizing", dict(p2=0.002, gamma=0.0005, q=0.002, ell=0.005, r_seep=0.02)),
]


def main(shots=60000):
    rows = []
    for name, kw in CASES:
        r = run(shots=shots, **kw)
        rows.append((name, r))
        print(f"{name} (witness {r['witness']:.2f}, shots ending leaked {r['shots_with_final_leak']:.3f})")
        for k in DECODERS:
            print(f"  {k:28s} p_L {r[k]:.4f}")
        print(f"  pre-registered: leakage-aware/qg {r['leakage-aware'] / max(r['qg'], 1e-12):.2f} (z {r['z_qg_vs_leak']:+.1f});"
              f" leakage-aware vs standard z {r['z_leak_vs_std']:+.1f}")
        print(f"  deviation: last-layer erasure vs standard z {r['z_leaklast_vs_std']:+.1f};"
              f" qg without flags vs standard {r['standard'] / max(r['qg, no leakage flags'], 1e-12):.2f} (z {r['z_qgnoflag_vs_std']:+.1f});"
              f" adding last-layer flags to qg z {r['z_qglast_vs_qgnoflag']:+.1f}")
    return rows


if __name__ == "__main__":
    main()
