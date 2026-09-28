"""
Does the T1-aware decoder of §59 survive circuit-level noise and many
rounds?

§59 found, for the d = 3 rotated surface code with perfect syndrome
extraction, that knowing the noise is T1 (a decay only turns 1 into 0)
cuts the logical error 2.1-2.5x. Here the whole Z-memory circuit is
simulated: rotated codes of distance d = 3 and 5, (d^2 - 1)/2 Z-type
ancillas, R rounds of syndrome extraction (reset, four CNOT layers in the
N-order NW, NE, SW, SE, measurement), then a final readout of the data.

Why this can be simulated exactly without a density matrix: in a Z-memory
only X-type errors matter, the Z-stabilizer circuit is CNOTs from data to
ancillas, and amplitude damping keeps a state diagonal in the Z basis.
The Z-basis populations then evolve as a classical Markov chain on bits:
the data start as a uniformly random codeword (the Z-basis content of
|0_L>), a CNOT is an XOR, a decay turns a 1 into 0 with probability
gamma, and the X or Y part of a depolarizing error flips a bit. The
X-type stabilizers are left out (they do not affect a Z-memory's
X-error statistics except through their own faults, which are not
modelled): this is the main approximation.

Noise per location:
  every CNOT            two-qubit depolarizing p2, then decay gamma on both
  idle qubits per layer decay gamma_idle = gamma
  before each readout   decay gamma_m = 2 gamma, then readout flip q
  ancilla reset         wrong state with probability q

Decoders (minimum-weight perfect matching, pymatching, on the detector
graph built by enumerating every single fault through the circuit):
  standard   one weight per fault, its average probability (a decay at a
             location counts gamma * P(bit = 1 there), P from the
             noiseless circuit)
  T1-aware   per shot, the decay faults of each DATA qubit are reweighted
             by its final readout: read 1 -> it (almost) never decayed,
             read 0 -> twice the average. Depolarizing and measurement
             faults keep their weights, so the rule stays soft when the
             noise is not pure T1 (the §59 D lesson).

Needs pymatching (pip install pymatching).

Findings (python examples/surface_code_circuit_t1_qg.py --figure):

  Checks: noiseless circuits give no detector events and the prepared
  logical value for |0_L> and |1_L>; halving every error rate (d = 3,
  T1-dominated) lowers p_L 3.7x, the quadratic scaling expected at d = 3.

  Logical error after R = d rounds (100 000 shots at d = 3, 20 000 at
  d = 5; +- 95 % intervals; "paired" counts shots where only one decoder
  fails):

    noise (p2, gamma, q)              d   standard       T1-aware       ratio  paired z
    pure T1 (0, 0.003, 0.001)         3   0.0131(7)      0.0060(5)      2.19   +25
                                      5   0.0065(11)     0.0019(6)      3.33   +9
    T1-dominated (0.0005, 0.003,      3   0.0149(8)      0.0091(6)      1.63   +17
                  0.002)              5   0.0080(12)     0.0040(9)      2.04   +7
    mixed (0.002, 0.002, 0.002)       3   0.0095(6)      0.0084(6)      1.14   +5.5
                                      5   0.0050(10)     0.0027(7)      1.91   +5.7
    depolarizing-dominated            3   0.0030(3)      0.0034(4)      0.87   -4.3
      (0.004, 0.0005, 0.002)          5   0.0011(5)      0.0011(5)      0.96   -0.4

  * The §59 gain survives circuit-level noise and repeated rounds: 2.2x
    for pure T1 at d = 3 (the §59 code-capacity value was 2.1-2.5x), and
    it GROWS with distance: 3.3x at d = 5. With T1 dominant but not alone
    it is 1.6x (d = 3) and 2.0x (d = 5); for a mixed budget 1.1x and 1.9x.
  * It is not free. When depolarizing noise dominates, the reweighting
    costs 15 % at d = 3 (z = -4.3, significant) and nothing measurable at
    d = 5. Trying Bayes-corrected multipliers did not remove the loss.
  * The qg witness chooses the decoder without labels. On a separate
    calibration batch, the register-mean qg_Z of the final data readout
    (0 for any symmetric flip, > 0 under decay) divided by the detector
    firing rate is 1.3-2.6 where the T1-aware decoder wins and 0.34-0.54
    where it loses. The switch "T1-aware if the ratio exceeds 1" picks the
    better decoder in all eight cases. The threshold 1 was chosen after
    seeing these eight cases; it is a heuristic, not a derived bound.

  What is new and what is not. Decoders that use noise bias or erasure
  information are known (bias-tailored codes, erasure conversion,
  leakage-aware decoding). Ours: a T1-aware reweighting that needs only
  the final data readout, its circuit-level numbers at d = 3 and 5 with
  paired statistics, its cost when T1 does not dominate, and the qg
  witness as the switch. Limitations: a Z-memory with the X-stabilizer
  circuits left out (their faults would add X errors through hook
  propagation); amplitude damping, depolarizing and readout errors only
  (no leakage, crosstalk or correlated errors); per-location decay
  weights from the noiseless circuit; multipliers 2 (read 0) and 2q (read
  1) are a simple Bayes approximation; faults touching more than two
  detectors dropped from the matching graph.
"""

import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# --------------------------------------------------------------------- #
# code layout
# --------------------------------------------------------------------- #
class Code:
    def __init__(self, d):
        self.d = d
        self.nd = d * d
        dq = lambda r, c: r * d + c  # noqa: E731
        self.zstabs = []  # list of dict role -> data index
        self.xstabs = []
        for i in range(d + 1):
            for j in range(d + 1):
                roles = {}
                for role, (r, c) in (("NW", (i - 1, j - 1)), ("NE", (i - 1, j)), ("SW", (i, j - 1)), ("SE", (i, j))):
                    if 0 <= r < d and 0 <= c < d:
                        roles[role] = dq(r, c)
                if len(roles) < 2:
                    continue
                even = (i + j) % 2 == 0
                if len(roles) == 4:
                    (self.zstabs if even else self.xstabs).append(roles)
                elif even and j in (0, d):
                    self.zstabs.append(roles)
                elif not even and i in (0, d):
                    self.xstabs.append(roles)
        self.na = len(self.zstabs)
        self.nq = self.nd + self.na
        self.zl = [dq(0, c) for c in range(d)]  # logical Z: top row
        # CNOT layers: role order NW, NE, SW, SE
        self.layers = []
        for role in ("NW", "NE", "SW", "SE"):
            self.layers.append([(st[role], self.nd + a) for a, st in enumerate(self.zstabs) if role in st])
        self._codewords()

    def _codewords(self):
        gens = []
        for st in self.xstabs:
            v = np.zeros(self.nd, dtype=bool)
            v[list(st.values())] = True
            gens.append(v)
        self.xgens = np.array(gens)
        self.xl = np.zeros(self.nd, dtype=bool)
        self.xl[[r * self.d for r in range(self.d)]] = True  # logical X: left column

    def random_codewords(self, shots, rng, logical=0):
        coef = rng.integers(0, 2, size=(shots, len(self.xgens))).astype(np.uint8)
        cw = (coef @ self.xgens.astype(np.uint8)) % 2
        if logical:
            cw ^= self.xl.astype(np.uint8)
        return cw.astype(bool)


# --------------------------------------------------------------------- #
# circuit as an op list
# --------------------------------------------------------------------- #
def ops_list(code, rounds):
    ops = []
    for t in range(rounds):
        ops.append(("reset", t))
        for li, layer in enumerate(code.layers):
            ops.append(("cx", t, li, layer))
        ops.append(("meas", t))
    ops.append(("final",))
    return ops


def simulate(code, rounds, shots, rng, p2=0.0, gamma=0.0, q=0.0, logical=0, noise=True):
    """Returns (ancilla outcomes [shots, rounds, na], final data [shots, nd])."""
    nd, na = code.nd, code.na
    st = np.zeros((shots, code.nq), dtype=bool)
    st[:, :nd] = code.random_codewords(shots, rng, logical)
    meas = np.zeros((shots, rounds, na), dtype=bool)
    g_idle, g_m = gamma, 2 * gamma

    def decay(cols, g):
        if noise and g > 0:
            hit = rng.random((shots, len(cols))) < g
            sub = st[:, cols]
            st[:, cols] = sub & ~hit

    for op in ops_list(code, rounds):
        if op[0] == "reset":
            st[:, nd:] = False
            if noise and q > 0:
                st[:, nd:] = rng.random((shots, na)) < q
            decay(list(range(nd)), g_idle)
        elif op[0] == "cx":
            layer = op[3]
            busy = set()
            for c, tq in layer:
                st[:, tq] ^= st[:, c]
                busy |= {c, tq}
                if noise and p2 > 0:
                    # 15 two-qubit Paulis: X or Y on a qubit flips it
                    pick = rng.integers(1, 16, size=shots)
                    err = rng.random(shots) < p2
                    pa, pb = pick // 4, pick % 4  # 0=I,1=X,2=Y,3=Z on each
                    st[:, c] ^= err & ((pa == 1) | (pa == 2))
                    st[:, tq] ^= err & ((pb == 1) | (pb == 2))
                decay([c, tq], gamma)
            idle = [x for x in range(code.nq) if x not in busy]
            decay(idle, g_idle)
        elif op[0] == "meas":
            t = op[1]
            decay(list(range(nd, code.nq)), g_m)
            decay(list(range(nd)), g_m)
            out = st[:, nd:].copy()
            if noise and q > 0:
                out ^= rng.random((shots, na)) < q
            meas[:, t] = out
        elif op[0] == "final":
            decay(list(range(nd)), g_m)
            final = st[:, :nd].copy()
            if noise and q > 0:
                final ^= rng.random((shots, nd)) < q
    return meas, final


def detectors(code, meas, final):
    """Detector bits [shots, (rounds + 1) * na] and the raw logical bit."""
    shots, rounds, na = meas.shape
    fs = np.zeros((shots, na), dtype=bool)
    for a, stb in enumerate(code.zstabs):
        fs[:, a] = np.bitwise_xor.reduce(final[:, list(stb.values())], axis=1)
    rows = [meas[:, 0]]
    for t in range(1, rounds):
        rows.append(meas[:, t] ^ meas[:, t - 1])
    rows.append(fs ^ meas[:, rounds - 1])
    det = np.concatenate(rows, axis=1)
    logical = np.bitwise_xor.reduce(final[:, code.zl], axis=1)
    return det, logical


# --------------------------------------------------------------------- #
# fault enumeration -> detector graph
# --------------------------------------------------------------------- #
def _propagate(code, rounds, start_op, flips, meas_flip=None, final_flip=None):
    """Effect of one bit flip inserted after op index start_op (linear)."""
    nd, na = code.nd, code.na
    st = np.zeros((1, code.nq), dtype=bool)
    for x in flips:
        st[0, x] = True
    meas = np.zeros((1, rounds, na), dtype=bool)
    final = np.zeros((1, nd), dtype=bool)
    for k, op in enumerate(ops_list(code, rounds)):
        if k < start_op:
            continue
        if k == start_op and op[0] in ("reset", "cx", "meas", "final"):
            # the flip happens AFTER this op; except for measurement flips
            if op[0] == "meas" and meas_flip is not None:
                meas[0, op[1], meas_flip] = True
                continue
            if op[0] == "final" and final_flip is not None:
                final[0] = st[0, :nd]
                final[0, final_flip] ^= True
                continue
            if op[0] == "final":
                final[0] = st[0, :nd]
            continue
        if op[0] == "reset":
            st[:, nd:] = False
        elif op[0] == "cx":
            for c, tq in op[3]:
                st[:, tq] ^= st[:, c]
        elif op[0] == "meas":
            meas[:, op[1]] = st[:, nd:]
        elif op[0] == "final":
            final[0] = st[0, :nd]
    det, lg = detectors(code, meas, final)
    return tuple(np.flatnonzero(det[0])), bool(lg[0])


def bit_one_prob(code, rounds, shots=4000, seed=0):
    """P(bit = 1) after every op, from the noiseless circuit (for weighting
    decay faults)."""
    rng = np.random.default_rng(seed)
    nd, na = code.nd, code.na
    st = np.zeros((shots, code.nq), dtype=bool)
    st[:, :nd] = code.random_codewords(shots, rng)
    out = []
    for op in ops_list(code, rounds):
        if op[0] == "reset":
            st[:, nd:] = False
        elif op[0] == "cx":
            for c, tq in op[3]:
                st[:, tq] ^= st[:, c]
        out.append(st.mean(axis=0))
    return out


def fault_list(code, rounds, p2, gamma, q):
    """Every single fault: (detector tuple, logical flag, prob, tag), where
    tag = ('ad', data qubit) for decays of data qubits, else None."""
    P1 = bit_one_prob(code, rounds)
    nd = code.nd
    faults = []
    g_idle, g_m = gamma, 2 * gamma
    ops = ops_list(code, rounds)
    for k, op in enumerate(ops):
        if op[0] == "reset":
            for a in range(code.na):
                if q > 0:
                    faults.append((*_propagate(code, rounds, k, [nd + a]), q, None))
            for x in range(nd):
                p = g_idle * P1[k][x]
                if p > 0:
                    faults.append((*_propagate(code, rounds, k, [x]), p, ("ad", x)))
        elif op[0] == "cx":
            busy = set()
            for c, tq in op[3]:
                busy |= {c, tq}
                if p2 > 0:
                    for fl in ([c], [tq]):
                        faults.append((*_propagate(code, rounds, k, fl), p2 * 8 / 15, None))
                for x in (c, tq):
                    p = gamma * P1[k][x]
                    if p > 0:
                        faults.append((*_propagate(code, rounds, k, [x]), p, ("ad", x) if x < nd else None))
            for x in range(code.nq):
                if x not in busy:
                    p = g_idle * P1[k][x]
                    if p > 0:
                        faults.append((*_propagate(code, rounds, k, [x]), p, ("ad", x) if x < nd else None))
        elif op[0] == "meas":
            for a in range(code.na):
                p = q + g_m * P1[k - 1][nd + a]
                if p > 0:
                    faults.append((*_propagate(code, rounds, k, [], meas_flip=a), p, None))
            for x in range(nd):
                p = g_m * P1[k - 1][x]
                if p > 0:
                    faults.append((*_propagate(code, rounds, k, [x]), p, ("ad", x)))
        elif op[0] == "final":
            for x in range(nd):
                p = q + g_m * P1[k - 1][x]
                if p > 0:
                    tag = ("ad_final", x)
                    faults.append((*_propagate(code, rounds, k, [], final_flip=x), p, tag))
    return [f for f in faults if len(f[0]) > 0 or f[1]]


def prepare_edges(faults):
    """Group fault terms by graph edge (u, v, logical) for fast per-shot
    reweighting. Faults touching more than two detectors are dropped
    (rare; hyperedges)."""
    keys, index = [], {}
    t_edge, t_p, t_q, t_final = [], [], [], []
    for dets, lg, p, tag in faults:
        if len(dets) > 2:
            continue
        u = dets[0]
        v = dets[1] if len(dets) == 2 else -1
        key = (min(u, v), max(u, v), lg) if v >= 0 else (u, -1, lg)
        if key not in index:
            index[key] = len(keys)
            keys.append(key)
        t_edge.append(index[key])
        t_p.append(p)
        t_q.append(tag[1] if tag is not None else -1)
        t_final.append(tag is not None and tag[0] == "ad_final")
    return {"keys": keys, "edge": np.array(t_edge), "p": np.array(t_p), "q": np.array(t_q),
            "final": np.array(t_final, dtype=bool)}


def build_matching(E, mult=None, q=0.0):
    """mult: array over data qubits multiplying the decay part of their
    faults (None = standard weights)."""
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
    mt = pymatching.Matching()
    for (u, v), k in best.items():
        pk = float(min(max(pe[k], 1e-12), 0.499))
        w = math.log((1 - pk) / pk)
        fid = {0} if E["keys"][k][2] else set()
        if v < 0:
            mt.add_boundary_edge(u, weight=w, fault_ids=fid, error_probability=pk)
        else:
            mt.add_edge(u, v, weight=w, fault_ids=fid, error_probability=pk)
    return mt


def run(d, rounds, p2, gamma, q, shots, seed=0):
    rng = np.random.default_rng(seed)
    code = Code(d)
    n_det = (rounds + 1) * code.na
    faults = fault_list(code, rounds, p2, gamma, q)
    E = prepare_edges(faults)
    std = build_matching(E)
    res = {}
    fails = {"standard": 0, "t1_aware": 0, "raw": 0, "only_standard_fails": 0, "only_aware_fails": 0}
    cache = {}
    total = 0
    for logical in (0, 1):
        meas, final = simulate(code, rounds, shots // 2, rng, p2, gamma, q, logical)
        det, lg = detectors(code, meas, final)
        truth = lg ^ bool(logical)  # raw logical flip relative to the prepared value
        # final Z_L of a |1_L> codeword: X_L flips the top row's first qubit only
        pred = std.decode_batch(det)[:, 0].astype(bool)
        fail_std = pred ^ lg ^ bool(logical)
        fails["standard"] += int(np.sum(fail_std))
        fails["raw"] += int(np.sum(truth))
        keys = [tuple(r) for r in final.astype(np.uint8)]
        groups = defaultdict(list)
        for i, kk in enumerate(keys):
            groups[kk].append(i)
        for kk, idx in groups.items():
            if kk not in cache:
                mult = np.where(np.array(kk) == 0, 2.0, 2.0 * q)
                cache[kk] = build_matching(E, mult, q)
            pr = cache[kk].decode_batch(det[idx])[:, 0].astype(bool)
            fa = pr ^ lg[idx] ^ bool(logical)
            fails["t1_aware"] += int(np.sum(fa))
            fails["only_standard_fails"] += int(np.sum(fail_std[idx] & ~fa))
            fails["only_aware_fails"] += int(np.sum(fa & ~fail_std[idx]))
        total += shots // 2
    for k, v in fails.items():
        res[k] = v / total
    b, c = fails["only_standard_fails"], fails["only_aware_fails"]
    res["discordant"] = (b, c)
    res["mcnemar_z"] = (b - c) / math.sqrt(b + c) if b + c > 0 else 0.0
    res["shots"] = total
    res["faults"] = len(faults)
    return res


def witness_ratio(d, rounds, p2, gamma, q, shots=5000, seed=99):
    """Label-free switch from a separate calibration batch: register-mean
    qg_Z of the final data readout (0 for any symmetric flip, > 0 under
    decay) divided by the detector firing rate."""
    rng = np.random.default_rng(seed)
    code = Code(d)
    meas, final = simulate(code, rounds, shots, rng, p2, gamma, q)
    det, _ = detectors(code, meas, final)
    return float((1 - 2 * final).mean() / det.mean())


def ci(p, n):
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / n)


CASES = [
    ("pure T1 (+ readout)", dict(p2=0.0, gamma=0.003, q=0.001)),
    ("T1-dominated", dict(p2=0.0005, gamma=0.003, q=0.002)),
    ("mixed", dict(p2=0.002, gamma=0.002, q=0.002)),
    ("depolarizing-dominated", dict(p2=0.004, gamma=0.0005, q=0.002)),
]


def main(shots3=100000, shots5=20000):
    out = []
    for name, kw in CASES:
        for d, R, S in ((3, 3, shots3), (5, 5, shots5)):
            r = run(d, R, shots=S, **kw)
            r["witness"] = witness_ratio(d, R, **kw)
            r["switched"] = r["t1_aware"] if r["witness"] > 1 else r["standard"]
            out.append((name, d, R, r))
            ratio = r["standard"] / max(r["t1_aware"], 1e-9)
            print(f"{name:24s} d={d} R={R}: p_L standard {r['standard']:.4f} +- {ci(r['standard'], r['shots']):.4f}"
                  f"  T1-aware {r['t1_aware']:.4f} +- {ci(r['t1_aware'], r['shots']):.4f}  ratio {ratio:.2f}"
                  f"  paired: only std fails {r['discordant'][0]}, only aware fails {r['discordant'][1]}, z = {r['mcnemar_z']:+.1f}"
                  f"  (undecoded {r['raw']:.3f}, {r['faults']} faults)\n"
                  f"{'':24s}   witness mean qg_Z / detector rate = {r['witness']:.2f} -> switch picks "
                  f"{'T1-aware' if r['witness'] > 1 else 'standard'}: p_L {r['switched']:.4f}")
    return out


def make_figure(path, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    names = [c[0] for c in CASES]
    x = np.arange(len(names))
    w = 0.2
    for j, (d, col) in enumerate(((3, "#1f6fb2"), (5, "#8c2d04"))):
        rows = {r[0]: r[3] for r in out if r[1] == d}
        std = [rows[n]["standard"] for n in names]
        aw = [rows[n]["t1_aware"] for n in names]
        e_s = [ci(v, rows[n]["shots"]) for v, n in zip(std, names)]
        e_a = [ci(v, rows[n]["shots"]) for v, n in zip(aw, names)]
        ax.bar(x + (2 * j - 1.5) * w, std, w, yerr=e_s, color=col, alpha=0.45, label=f"d = {d}, standard MWPM")
        ax.bar(x + (2 * j - 0.5) * w, aw, w, yerr=e_a, color=col, label=f"d = {d}, T1-aware MWPM")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel(f"logical error after d rounds")
    ax.set_title("Circuit-level Z-memory: the T1-aware decoder helps when T1 dominates", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    res = main()
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"), res)
