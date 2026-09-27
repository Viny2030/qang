"""
The distance-3 rotated surface code read in qg: syndromes as multi-qubit
qg, a readout-free error-rate estimate, and a T1 witness that changes the
decoder.

Layout: 9 data qubits q = 3 r + c on a 3x3 grid. Z-type stabilizers
{0,1,3,4}, {4,5,7,8} (weight 4) and {2,5}, {3,6} (weight 2) detect X errors;
X-type {1,2,4,5}, {3,4,6,7}, {0,1}, {7,8}. Logical Z_L = Z0 Z1 Z2,
X_L = X0 X3 X6. We study a Z-memory: prepare |0_L>, let noise act, read all
data qubits in Z, rebuild the Z syndrome from the bits, decode, and read
Z_L. Each stabilizer average is a qg of a parity, qg_k = <S_k>; the logical
result is qg_L = 1 - 2 p_L.

A. Code capacity, independent bit flips p (exact, 512 error patterns).
   Minimum-weight decoding gives p_L ~ 18 p^2 and a pseudo-threshold where
   p_L = p.
B. Syndrome qg as an error-rate meter. With flips p and an ancilla readout
   gain b = 1 - 2q (§31), a weight-w stabilizer reads qg_w = b (1 - 2p)^w.
   The two weights of this code separate the two: (1 - 2p)^2 = qg_4/qg_2,
   b = qg_2^2/qg_4. Finite-shot bias and spread by Monte Carlo.
C. T1. Amplitude damping on the data is not a Pauli channel: in the Z basis
   it only turns 1 into 0. Its Pauli twirl is a symmetric bit flip of
   p = gamma/2, which leaves every data qubit at qg_Z = 0; the real channel
   moves the register-mean qg_Z of the final readout to +gamma. That mean
   costs nothing (it comes from the same bits as the syndrome) and tells
   the decoder that a bit read as 1 cannot have flipped. We compare the
   minimum-weight decoder, a T1-aware minimum-weight decoder (errors only
   on bits read 0) and the exact maximum-likelihood decoder.

Exact checks: 9-qubit density-matrix simulation of |0_L> with amplitude
damping reproduces <Z_i> = gamma and the classical model used in C.

Findings (python examples/surface_code_d3_qg.py):

  A. Bit flips, exact: p_L = 1.79e-5, 1.73e-3, 1.44e-2 at p = 0.001,
     0.01, 0.03 (p_L/p^2 -> 18: eighteen weight-2 patterns defeat the
     decoder); minimum weight equals maximum likelihood here to 1e-15.
     Pseudo-threshold p_L = p at p = 0.0753.
  B. Two stabilizer weights separate data errors from ancilla readout
     error. With q = 0.02, reading p from the weight-4 qg alone gives 0.015
     for a true 0.010 (50% high) and 0.035 for 0.030; the ratio
     qg_4/qg_2 gives 0.0100 and 0.0300 exactly, and b = 0.960. With 2000
     shots per stabilizer: p = 0.0098 +- 0.0025 and 0.0298 +- 0.0036, b to
     +-0.014-0.026, no visible bias. This is §31's calibrated readout gain
     b, obtained for free from the code's own geometry.
  C. T1 on the data. The 9-qubit density matrix confirms <Z_i> = gamma on
     every qubit and the classical decay model to 4e-17. A weight-w
     syndrome reads qg_w = (1-gamma)^w + gamma^w exactly (the stabilizer
     support of a codeword is a uniform even-weight string), against
     (1-gamma)^w for the Pauli twirl: 0.98020 vs 0.98010 for w = 2 at
     gamma = 0.01, a difference of gamma^w that needs ~1e8 shots to see.
     Syndromes are practically blind to T1-vs-Pauli; mean qg_Z = gamma is
     not (it is 0 for any symmetric flip) and is read from the same bits. The Pauli twirl underestimates the minimum-weight p_L by
     6-11 % (4.41e-4 vs 4.90e-4 at gamma = 0.01). Knowing it is T1, a
     decoder that places errors only on bits read 0 cuts p_L 2.1-2.5x for
     gamma <= 0.1 (1.6x at 0.3)
     (4.90e-4 -> 1.99e-4 at gamma = 0.01; 4.03e-2 -> 1.89e-2 at 0.1) and
     equals maximum likelihood (within 0.1 %) up to gamma = 0.3. Averaged over |0_L> and
     |1_L> (T1 treats them differently: 0.211 vs 0.251 at gamma = 0.3).
  D. The honest limit of C. Add symmetric flips p on top of the decay and
     the hard "bits read 0" rule breaks: at gamma = 0.03 it is already 1.8x
     WORSE than minimum weight with p = 0.003, and 2.8-2.9x worse with
     p = 0.01-0.03. The maximum-likelihood decoder with gamma and p
     estimated from mean qg_Z = gamma(1-2p) and qg_4 = ((1-gamma)(1-2p))^4
     (both recovered to 4 digits) never loses to minimum weight, but its
     gain shrinks fast: 2.4x at p = 0, 1 % at p = 0.1 gamma, 17 % at
     gamma = 0.1, p = 0.01. The qg readout tells you which decoder to use;
     the benefit is large only when T1 dominates the data-qubit noise.

  What is new and what is not. Syndrome-statistics noise estimation and
  asymmetric-channel (T1-aware) decoding are known ideas. Ours: the
  two-weight separation of p and b on the d = 3 code, the observation that
  the syndromes are blind to T1-vs-Pauli while the register-mean qg_Z of
  the same bits is not, and the exact d = 3 numbers with the robustness
  limit D. Limitations: code capacity only (perfect syndrome extraction
  except for the readout gain in B, a single round); Z-memory only (an
  X-memory reads in X and gets no qg_Z signal); no circuit-level noise,
  no leakage, no repeated rounds; d = 3 only.
"""

import itertools
import math

import numpy as np

N = 9
Z_STABS = [(0, 1, 3, 4), (4, 5, 7, 8), (2, 5), (3, 6)]
X_STABS = [(1, 2, 4, 5), (3, 4, 6, 7), (0, 1), (7, 8)]
Z_L = (0, 1, 2)
X_L = (0, 3, 6)


def _mask(qs):
    m = 0
    for q in qs:
        m |= 1 << q
    return m


ZS = [_mask(s) for s in Z_STABS]
XS = [_mask(s) for s in X_STABS]
ZL = _mask(Z_L)
XL = _mask(X_L)
POP = [bin(i).count("1") for i in range(1 << N)]


def syndrome(bits):
    return tuple(POP[bits & s] & 1 for s in ZS)


def zl(bits):
    return POP[bits & ZL] & 1


def codewords(logical=0):
    """Z-basis support of |0_L> (logical=0) or |1_L>: span of the X stabilizers."""
    out = []
    for k in range(1 << len(XS)):
        c = 0
        for j, s in enumerate(XS):
            if k >> j & 1:
                c ^= s
        out.append(c ^ (XL if logical else 0))
    return sorted(set(out))


# --------------------------------------------------------------------- #
# decoders
# --------------------------------------------------------------------- #
def _mw_table():
    table = {}
    for e in sorted(range(1 << N), key=lambda x: (POP[x], x)):
        table.setdefault(syndrome(e), e)
    return table


MW = _mw_table()


def decode_mw(bits):
    return bits ^ MW[syndrome(bits)]


def decode_t1(bits):
    """Minimum weight among flips on bits read 0 (a 1 -> 0 decay leaves a 0)."""
    zeros = ~bits & ((1 << N) - 1)
    s = syndrome(bits)
    best = None
    for e in range(1 << N):
        if e & ~zeros:
            continue
        if syndrome(e) == s and (best is None or POP[e] < POP[best]):
            best = e
    return bits ^ (best if best is not None else MW[s])


# --------------------------------------------------------------------- #
# A. code capacity
# --------------------------------------------------------------------- #
def logical_error_bitflip(p):
    pl = 0.0
    for e in range(1 << N):
        if zl(decode_mw(e)):
            pl += p ** POP[e] * (1 - p) ** (N - POP[e])
    return pl


def pseudo_threshold():
    lo, hi = 1e-4, 0.49
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if logical_error_bitflip(mid) < mid:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------- #
# B. syndrome qg as an error-rate meter
# --------------------------------------------------------------------- #
def syndrome_qg(p, q=0.0):
    b = 1 - 2 * q
    return {len(s): b * (1 - 2 * p) ** len(s) for s in Z_STABS}


def estimate_p_b(qg4, qg2):
    r = qg4 / qg2
    p = (1 - math.sqrt(max(r, 0.0))) / 2
    b = qg2 * qg2 / qg4
    return p, b


def sample_estimate(p, q, shots, reps, rng):
    """Monte Carlo of the ancilla readout: each shot flips data bits with p,
    each stabilizer readout with q. Returns arrays of (p_hat, b_hat)."""
    ps, bs = [], []
    for _ in range(reps):
        flips = rng.random((shots, N)) < p
        bits = (flips * (1 << np.arange(N))).sum(axis=1)
        vals = {}
        for s, m in zip(Z_STABS, ZS):
            par = np.array([POP[int(x) & m] & 1 for x in bits])
            par ^= rng.random(shots) < q
            vals.setdefault(len(s), []).append(1 - 2 * par.mean())
        qg4, qg2 = np.mean(vals[4]), np.mean(vals[2])
        ph, bh = estimate_p_b(qg4, qg2)
        ps.append(ph)
        bs.append(bh)
    return np.array(ps), np.array(bs)


# --------------------------------------------------------------------- #
# C. T1 on the data
# --------------------------------------------------------------------- #
def readout_dist(gamma, p=0.0, logical=0):
    """Exact distribution of the final Z readout of |0_L> or |1_L> after
    amplitude damping gamma (each 1 decays to 0) followed by bit flips p.
    The Z-basis populations of amplitude damping evolve classically, so a
    uniform codeword plus classical decay is exact (checked below against
    the 9-qubit density matrix)."""
    cw = codewords(logical)
    dist = {}
    for c in cw:
        ones = [q for q in range(N) if c >> q & 1]
        for k in range(len(ones) + 1):
            for sub in itertools.combinations(ones, k):
                r = c ^ _mask(sub)
                dist[r] = dist.get(r, 0.0) + gamma**k * (1 - gamma) ** (len(ones) - k) / len(cw)
    if p > 0:
        out = {}
        for r, pr in dist.items():
            for e in range(1 << N):
                out[r ^ e] = out.get(r ^ e, 0.0) + pr * p ** POP[e] * (1 - p) ** (N - POP[e])
        dist = out
    return dist


def decode_ml(bits, gamma, p=0.0):
    """Exact maximum likelihood for decay gamma followed by flips p: per bit,
    P(0|1) = a = gamma (1-p) + (1-gamma) p, P(1|0) = p. Returns the logical."""
    a = gamma * (1 - p) + (1 - gamma) * p
    tr = {(1, 0): a, (1, 1): 1 - a, (0, 0): 1 - p, (0, 1): p}
    lik = []
    for logical in (0, 1):
        tot = 0.0
        for c in codewords(logical):
            w = 1.0
            for q in range(N):
                w *= tr[(c >> q & 1, bits >> q & 1)]
            tot += w
        lik.append(tot)
    return 0 if lik[0] >= lik[1] else 1


def estimate_gamma_p(mean_qg_z, qg4):
    """Invert mean qg_Z = gamma (1-2p) and qg_4 ~ ((1-gamma)(1-2p))^4
    (exact up to O(gamma^4); see findings C)."""
    u = qg4 ** 0.25  # (1-gamma)(1-2p)
    one_minus_2p = u + mean_qg_z
    return mean_qg_z / one_minus_2p, (1 - one_minus_2p) / 2


def t1_study(gamma, p=0.0):
    """Logical error averaged over |0_L> and |1_L> for three decoders. The ML
    decoder uses gamma and p estimated from the qg of the |0_L> readout."""
    d0 = readout_dist(gamma, p, 0)
    mean_qg = sum(pr * np.mean([1 - 2 * (r >> q & 1) for q in range(N)]) for r, pr in d0.items())
    stab = {s: sum(pr * (1 - 2 * (POP[r & _mask(s)] & 1)) for r, pr in d0.items()) for s in Z_STABS}
    g_hat, p_hat = estimate_gamma_p(mean_qg, stab[Z_STABS[0]])
    g_hat, p_hat = max(g_hat, 0.0), max(p_hat, 0.0)
    res = {"mean_qg_z": mean_qg, "stab_qg": stab, "gamma_hat": g_hat, "p_hat": p_hat}
    for name in ("mw", "t1", "ml"):
        per = []
        for logical in (0, 1):
            d = d0 if logical == 0 else readout_dist(gamma, p, 1)
            if name == "mw":
                fail = sum(pr for r, pr in d.items() if zl(decode_mw(r)) != logical)
            elif name == "t1":
                fail = sum(pr for r, pr in d.items() if zl(decode_t1(r)) != logical)
            else:
                fail = sum(pr for r, pr in d.items() if decode_ml(r, g_hat, p_hat) != logical)
            per.append(fail)
        res["pL_" + name] = 0.5 * (per[0] + per[1])
        res["pL_" + name + "_01"] = tuple(per)
    res["pL_twirl"] = logical_error_bitflip(gamma / 2 + p - gamma * p)
    return res



def t1_density_matrix_check(gamma):
    """9-qubit density matrix: |0_L> then amplitude damping on every qubit.
    Returns (<Z_i> list, Z-basis populations)."""
    dim = 1 << N
    psi = np.zeros(dim)
    for c in codewords(0):
        psi[c] = 1.0
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi)
    k0 = np.array([[1, 0], [0, math.sqrt(1 - gamma)]])
    k1 = np.array([[0, math.sqrt(gamma)], [0, 0]])
    rho = rho.reshape([2] * (2 * N))
    for q in range(N):
        ax = N - 1 - q  # qubit q is bit q of the index (little endian)
        new = 0
        for K in (k0, k1):
            t = np.tensordot(K, rho, axes=([1], [ax]))
            t = np.moveaxis(t, 0, ax)
            t = np.tensordot(t, K.conj(), axes=([N + ax], [1]))
            t = np.moveaxis(t, -1, N + ax)
            new = new + t
        rho = new
    rho = rho.reshape(dim, dim)
    pops = np.real(np.diag(rho))
    zs = [float(sum(pops[i] * (1 - 2 * (i >> q & 1)) for i in range(dim))) for q in range(N)]
    return zs, pops


def make_figure(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ps = np.geomspace(1e-3, 0.3, 40)
    ax1.loglog(ps, [logical_error_bitflip(p) for p in ps], color="#1f6fb2", lw=2, label="d = 3, minimum weight")
    ax1.loglog(ps, ps, "k:", lw=1, label="unencoded qubit")
    pt = pseudo_threshold()
    ax1.axvline(pt, color="#8c2d04", lw=1, ls="--", label=f"pseudo-threshold {pt:.3f}")
    ax1.set_xlabel("bit-flip probability p per data qubit")
    ax1.set_ylabel("logical error p_L = (1 - qg_L)/2")
    ax1.set_title("Code capacity, exact over 512 patterns", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    gs = np.linspace(0.01, 0.4, 25)
    res = [t1_study(g) for g in gs]
    ax2.semilogy(gs, [r["pL_twirl"] for r in res], "k:", lw=1.2, label="Pauli twirl (p = gamma/2), MW")
    ax2.semilogy(gs, [r["pL_mw"] for r in res], color="#1f6fb2", lw=2, label="T1, minimum weight")
    ax2.semilogy(gs, [r["pL_t1"] for r in res], color="#e0a030", lw=2, label="T1, MW on bits read 0")
    ax2.semilogy(gs, [r["pL_ml"] for r in res], "--", color="#8c2d04", lw=1.5, label="T1, max likelihood")
    ax2.set_xlabel("decay probability gamma per data qubit")
    ax2.set_ylabel("logical error p_L")
    ax2.set_title("T1: mean qg_Z = gamma flags it; a T1-aware decoder uses it", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    import sys

    print("Code: Z stabilizers", Z_STABS, " codewords of |0_L>:", len(codewords(0)))
    print("\nA. code capacity (bit flips), minimum-weight decoder")
    for p in (0.001, 0.01, 0.03, 0.1):
        pl = logical_error_bitflip(p)
        print(f"  p = {p:5.3f}: p_L = {pl:.3e}  p_L/p^2 = {pl / p**2:.2f}")
    print(f"  pseudo-threshold (p_L = p): {pseudo_threshold():.4f}")

    print("\nB. syndrome qg as an error-rate meter: qg_w = b (1-2p)^w")
    rng = np.random.default_rng(7)
    for p, q in ((0.01, 0.0), (0.01, 0.02), (0.03, 0.02), (0.03, 0.05)):
        sq = syndrome_qg(p, q)
        naive = (1 - sq[4] ** 0.25) / 2
        pe, be = estimate_p_b(sq[4], sq[2])
        ph, bh = sample_estimate(p, q, 2000, 200, rng)
        print(f"  p={p}, q={q}: qg_4={sq[4]:.4f} qg_2={sq[2]:.4f}  naive p from qg_4 alone {naive:.4f}"
              f"  two-weight p={pe:.4f} b={be:.4f};  2000 shots: p {ph.mean():.4f}+-{ph.std():.4f}"
              f" b {bh.mean():.4f}+-{bh.std():.4f}")

    print("\nC. T1 on the data qubits (exact over the readout distribution)")
    zs, pops = t1_density_matrix_check(0.2)
    dist = readout_dist(0.2)
    maxdiff = max(abs(pops[i] - dist.get(i, 0.0)) for i in range(1 << N))
    print(f"  density matrix, gamma 0.2: <Z_i> = {min(zs):.6f}..{max(zs):.6f};"
          f" classical model max |diff| {maxdiff:.1e}")
    print(f"  {'gamma':>6} | {'mean qg_Z':>9} | {'qg w4':>7} {'qg w2':>7} | {'twirl p=g/2':>11} {'MW':>9} {'MW on 0s':>9} {'ML':>9}")
    for g in (0.01, 0.03, 0.1, 0.2, 0.3):
        r = t1_study(g)
        st = r["stab_qg"]
        print(f"  {g:6.2f} | {r['mean_qg_z']:+9.4f} | {st[Z_STABS[0]]:7.4f} {st[Z_STABS[2]]:7.4f} |"
              f" {r['pL_twirl']:11.3e} {r['pL_mw']:9.3e} {r['pL_t1']:9.3e} {r['pL_ml']:9.3e}"
              f"   MW |0_L>/|1_L> {r['pL_mw_01'][0]:.2e}/{r['pL_mw_01'][1]:.2e}")
    print("  T1 plus symmetric flips p (p_L averaged over |0_L>, |1_L>; ML uses gamma, p estimated from qg)")
    for g, p in ((0.03, 0.0), (0.03, 0.003), (0.03, 0.01), (0.03, 0.03), (0.1, 0.01), (0.1, 0.05)):
        r = t1_study(g, p)
        print(f"    gamma {g}, p {p}: mean qg_Z {r['mean_qg_z']:+.4f} -> gamma^ {r['gamma_hat']:.4f} p^ {r['p_hat']:.4f}"
              f"  MW {r['pL_mw']:.3e}  MW on 0s {r['pL_t1']:.3e}  ML(qg-estimated) {r['pL_ml']:.3e}")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"))
