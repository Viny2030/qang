"""
Bell pairs in a quantum network, read in qg: fidelity, noise diagnosis,
which distillation to run, and how long a pair may wait in memory.

A shared pair meant to be |Phi+> = (|00> + |11>)/sqrt 2 is characterised
by three two-qubit qg correlations and two single-qubit polar biases:

    c_x = <XX>,  c_y = <YY>,  c_z = <ZZ>,     m_A = <Z_A>,  m_B = <Z_B>

(|Phi+> has c = (+1, -1, +1), m = 0). Three measurement settings (XX, YY,
ZZ; the ZZ shots also give m_A, m_B) are enough for:

  * the fidelity, exact for any state:  F = (1 + c_x - c_y + c_z)/4;
  * the Bell-diagonal weights of the twirled pair
        Phi+ (1 + c_x - c_y + c_z)/4,   Phi- (1 - c_x + c_y + c_z)/4,
        Psi+ (1 + c_x + c_y - c_z)/4,   Psi- (1 - c_x - c_y - c_z)/4,
    i.e. which error dominates: Phi- = phase flip, Psi+ = bit flip,
    Psi- = both;
  * a T1 witness: amplitude damping in the memories moves m_A, m_B
    towards +1, while dephasing, bit flips and depolarizing leave them at
    0 (the §40 idea, now on a shared pair).

A. Diagnosis. Each noise type leaves a distinct signature in (c, m).
B. Distillation. The DEJMPS recurrence (Deutsch et al. 1996) keeps the
   Phi+ weight A, and its output depends on which error weight sits in
   the slot B that is combined with A (A' = (A^2 + B^2)/N). Local
   rotations choose which error goes there. The rule: put the SMALLEST
   error in slot B. Fixed protocol (Psi- in slot B, the textbook choice)
   vs the slot chosen from qg data estimated with a finite number of
   shots, vs the oracle.
C. Memory cutoff. A pair waiting in two memories with T1 and T2 used for
   entanglement-based QKD (BBM92): QBER_Z = (1 - c_z)/2 and
   QBER_X = (1 - c_x)/2, secret fraction 1 - h(QBER_X) - h(QBER_Z). The
   latest usable time, with and without one distillation round first, in
   closed form from the aged qg correlations.

Findings (python examples/bell_pairs_network_qg.py):

  A. Signatures (noise with p = 0.1 on each half; F from the three
     correlations equals the exact fidelity in every case):

       noise              c_x     c_y     c_z     m_A = m_B   F       largest error
       dephasing          +0.64   -0.64   +1.00   0           0.820   Phi- (phase)
       bit flip           +1.00   -0.64   +0.64   0           0.820   Psi+ (bit)
       Y flip             +0.64   -1.00   +0.64   0           0.820   Psi- (both)
       depolarizing       +0.81   -0.81   +0.81   0           0.858   all equal
       amplitude damping  +0.90   -0.90   +0.82   +0.10       0.905   Psi+

     Each Pauli noise keeps one correlation at its ideal value; only
     amplitude damping (memory T1) moves the local polar biases, which
     makes m_A, m_B a T1 witness on a shared pair.

  B. One DEJMPS round (output fidelity, success probability):

       input               F_in    textbook slot   best slot       qg rule, 200 shots/setting
       dephasing 0.1       0.820   0.954 (0.705)   0.954 (0.705)   0.954 (right 100 %)
       bit flip 0.1        0.820   0.954 (0.705)   0.954 (0.705)   0.954 (100 %)
       Y flip 0.1          0.820   0.705 (1.000)   0.954 (0.705)   0.954 (100 %)
       depolarizing 0.1    0.858   0.891 (0.828)   0.891 (0.828)   0.891 (100 %)
       ampl. damping 0.2   0.820   0.828 (0.820)   0.920 (0.731)   0.915 (94 %)

     * The textbook slot order fails when the dominant error sits in the
       slot combined with Phi+: for Y-flip noise one round LOWERS the
       fidelity (0.820 -> 0.705); for amplitude damping it gains almost
       nothing (0.828 vs 0.920 possible).
     * The rule "put the smallest estimated error in slot B", read from the
       three qg correlations with 200 shots per setting, reaches the best
       output in every case (94 % of the time for amplitude damping, where
       two errors are close).

  C. Memory cutoff for BBM92 (latest storage time with a positive secret
     fraction; the aged correlations have closed forms
     c_x = -c_y = e^{-2t/T2}, c_z = 1 - 2g + 2g^2, m = g, g = 1 - e^{-t/T1}):

       T2/T1                2       1       0.5     0.2
       no distillation      0.184   0.129   0.088   0.051   (units of T1)
       one DEJMPS round     0.395   0.232   0.144   0.076

     * One distillation round (best slot) roughly doubles the usable
       storage time when T1 limits it, and gains less as dephasing
       dominates.
     * It costs half the pairs, so for fresh pairs it lowers the key per
       input pair: distil only after 0.085 T1 (T2 = 2 T1), 0.083 (T2 = T1),
       0.056 (0.5) or 0.032 (0.2), all read from the aged qg correlations.

Honest scope. The fidelity formula, the Bell-diagonal reading, DEJMPS
with local rotations and memory cutoffs are all known; the contributions
are the qg signatures (including the T1 witness on a shared pair), the
slot rule applied from finite-shot qg data, and the closed-form cutoff.
Distillation is computed on the twirled (Bell-diagonal) state; perfect
local gates and classical communication; Markovian memories.
"""

import math

import numpy as np

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
PHI_PLUS = np.array([1, 0, 0, 1]) / math.sqrt(2)


def bell(name):
    v = {"Phi+": [1, 0, 0, 1], "Phi-": [1, 0, 0, -1], "Psi+": [0, 1, 1, 0], "Psi-": [0, 1, -1, 0]}[name]
    v = np.array(v, dtype=complex) / math.sqrt(2)
    return np.outer(v, v.conj())


def correlations(rho):
    """(c_x, c_y, c_z, m_A, m_B); qubit A is the left tensor factor."""
    e = lambda O: float(np.real(np.trace(rho @ O)))  # noqa: E731
    return (e(np.kron(X, X)), e(np.kron(Y, Y)), e(np.kron(Z, Z)), e(np.kron(Z, I2)), e(np.kron(I2, Z)))


def fidelity_from_qg(cx, cy, cz):
    return (1 + cx - cy + cz) / 4


def bell_weights(cx, cy, cz):
    return {"Phi+": (1 + cx - cy + cz) / 4, "Phi-": (1 - cx + cy + cz) / 4,
            "Psi+": (1 + cx + cy - cz) / 4, "Psi-": (1 - cx - cy - cz) / 4}


# --------------------------------------------------------------------- #
# channels on one qubit of the pair
# --------------------------------------------------------------------- #
def _apply(rho, kraus, qubit):
    out = np.zeros_like(rho)
    for K in kraus:
        full = np.kron(K, I2) if qubit == 0 else np.kron(I2, K)
        out = out + full @ rho @ full.conj().T
    return out


def pauli_channel(p, P):
    return [math.sqrt(1 - p) * I2, math.sqrt(p) * P]


def depolarizing(p):
    return [math.sqrt(1 - 3 * p / 4) * I2, math.sqrt(p / 4) * X, math.sqrt(p / 4) * Y, math.sqrt(p / 4) * Z]


def amplitude_damping(g):
    return [np.array([[1, 0], [0, math.sqrt(1 - g)]], dtype=complex), np.array([[0, math.sqrt(g)], [0, 0]], dtype=complex)]


def noisy_pair(kind, p, both=True):
    rho = bell("Phi+").astype(complex)
    K = {"dephasing": pauli_channel(p, Z), "bit flip": pauli_channel(p, X), "Y flip": pauli_channel(p, Y),
         "depolarizing": depolarizing(p), "amplitude damping": amplitude_damping(p)}[kind]
    rho = _apply(rho, K, 0)
    if both:
        rho = _apply(rho, K, 1)
    return rho


# --------------------------------------------------------------------- #
# B. DEJMPS with a chosen error in slot B
# --------------------------------------------------------------------- #
ERRORS = ("Phi-", "Psi+", "Psi-")


def dejmps_full(w, perm):
    """One DEJMPS round with the three errors assigned to slots (B, C, D) = perm
    (any permutation is reachable with bilateral local rotations). Returns the
    output Bell weights and the success probability. In the textbook slot
    order (B, C, D) = (Psi-, Psi+, Phi-):
        A' = (A^2 + B^2)/N,  B' = 2CD/N,  C' = (C^2 + D^2)/N,  D' = 2AB/N,
        N = (A + B)^2 + (C + D)^2."""
    b, c, d = perm
    A, B, C, D = w["Phi+"], w[b], w[c], w[d]
    N = (A + B) ** 2 + (C + D) ** 2
    return {"Phi+": (A * A + B * B) / N, b: 2 * C * D / N, c: (C * C + D * D) / N, d: 2 * A * B / N}, N


def dejmps(w, slot_b):
    """Output fidelity and success probability; they depend only on the slot-B error."""
    others = [e for e in ERRORS if e != slot_b]
    out, N = dejmps_full(w, (slot_b, others[0], others[1]))
    return out["Phi+"], N


def best_slot(w):
    return max(ERRORS, key=lambda e: dejmps(w, e)[0])


def estimate(rho, shots, rng):
    """Finite-shot estimate of (c_x, c_y, c_z) from `shots` shots per setting."""
    cx, cy, cz, _, _ = correlations(rho)
    return tuple(2 * rng.binomial(shots, (1 + c) / 2) / shots - 1 for c in (cx, cy, cz))


def distillation_table(cases, shots=200, trials=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for label, (kind, p) in cases.items():
        rho = noisy_pair(kind, p)
        w = bell_weights(*correlations(rho)[:3])
        f_fixed, s_fixed = dejmps(w, "Psi-")
        oracle = best_slot(w)
        f_orc, s_orc = dejmps(w, oracle)
        picks, fs = [], []
        for _ in range(trials):
            we = bell_weights(*estimate(rho, shots, rng))
            e = min(ERRORS, key=lambda k: we[k])  # the rule: smallest error in slot B
            picks.append(dejmps(w, e)[0] >= f_orc - 1e-9)  # ties (equal errors) count as right
            fs.append(dejmps(w, e)[0])
        out[label] = {"F_in": w["Phi+"], "weights": w, "fixed": f_fixed, "p_fixed": s_fixed, "oracle_slot": oracle,
                      "oracle": f_orc, "p_oracle": s_orc, "qg_rule": float(np.mean(fs)), "qg_right": float(np.mean(picks))}
    return out


# --------------------------------------------------------------------- #
# C. memory cutoff for entanglement-based QKD
# --------------------------------------------------------------------- #
def h(x):
    return 0.0 if x <= 0 or x >= 1 else -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def aged_pair(t, t2_over_t1):
    """Phi+ with both halves stored for time t (units of T1): amplitude damping and pure dephasing."""
    g = 1 - math.exp(-t)
    lam = math.exp(-t / t2_over_t1) / math.sqrt(1 - g)  # extra coherence factor from pure dephasing
    p = max(0.0, (1 - lam) / 2)
    rho = bell("Phi+").astype(complex)
    for q in (0, 1):
        rho = _apply(rho, amplitude_damping(g), q)
        rho = _apply(rho, pauli_channel(p, Z), q)
    return rho


def aged_correlations_closed(t, t2_over_t1):
    """Closed forms for Phi+ stored in two memories for time t (units of T1):
    c_x = -c_y = e^{-2t/T2},  c_z = 1 - 2 g + 2 g^2,  m_A = m_B = g,  g = 1 - e^{-t}."""
    g = 1 - math.exp(-t)
    cx = math.exp(-2 * t / t2_over_t1)
    return cx, -cx, 1 - 2 * g + 2 * g * g, g


def key_fraction(rho):
    cx, cy, cz, _, _ = correlations(rho)
    return max(0.0, 1 - h((1 - cx) / 2) - h((1 - cz) / 2))


def _key_from_weights(wt):
    ez = wt["Psi+"] + wt["Psi-"]  # Z-basis error
    ex = wt["Phi-"] + wt["Psi-"]  # X-basis error
    return max(0.0, 1 - h(ex) - h(ez))


def key_fraction_after_dejmps(rho):
    """Secret fraction per INPUT pair after one DEJMPS round (two pairs in,
    one out with probability N), maximised over the slot permutations."""
    from itertools import permutations

    w = bell_weights(*correlations(rho)[:3])
    best = 0.0
    for perm in permutations(ERRORS):
        out, N = dejmps_full(w, perm)
        best = max(best, _key_from_weights(out) * N / 2)
    return best


def cutoff(t2_over_t1, distill=False, t_max=3.0):
    """Latest storage time with a positive secret fraction per stored pair."""
    f = (lambda t: key_fraction_after_dejmps(aged_pair(t, t2_over_t1))) if distill else \
        (lambda t: key_fraction(aged_pair(t, t2_over_t1)))
    lo, hi = 0.0, t_max
    if f(hi) > 0:
        return hi
    for _ in range(50):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def make_figure(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ps = np.linspace(0.005, 0.25, 50)
    for kind, col in (("Y flip", "#8c2d04"), ("amplitude damping", "#e0a030"), ("dephasing", "#1f6fb2")):
        ws = [bell_weights(*correlations(noisy_pair(kind, p))[:3]) for p in ps]
        ax1.plot(ps, [w["Phi+"] for w in ws], color=col, lw=0.8, ls=":")
        ax1.plot(ps, [dejmps(w, "Psi-")[0] for w in ws], color=col, lw=1.5, ls="--")
        ax1.plot(ps, [dejmps(w, best_slot(w))[0] for w in ws], color=col, lw=2, label=kind)
    ax1.plot([], [], "k:", label="input")
    ax1.plot([], [], "k--", label="DEJMPS, textbook slot")
    ax1.plot([], [], "k-", label="DEJMPS, qg-chosen slot")
    ax1.set_xlabel("noise strength on each half")
    ax1.set_ylabel("fidelity with Phi+")
    ax1.set_title("One distillation round: the slot must hold the smallest error", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    ts = np.linspace(0, 0.5, 60)
    for r, col in ((2.0, "#1f6fb2"), (1.0, "#e0a030"), (0.5, "#8c2d04")):
        ax2.plot(ts, [key_fraction(aged_pair(t, r)) for t in ts], color=col, ls="--", lw=1.5)
        ax2.plot(ts, [key_fraction_after_dejmps(aged_pair(t, r)) for t in ts], color=col, lw=2,
                 label=f"T2 = {r} T1")
    ax2.plot([], [], "k--", label="no distillation")
    ax2.plot([], [], "k-", label="one DEJMPS round (per input pair)")
    ax2.set_xlabel("storage time in both memories, t / T1")
    ax2.set_ylabel("BBM92 secret fraction")
    ax2.set_title("Memory cutoff for entanglement-based QKD", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    import sys

    print("A. signatures of noise on both halves of Phi+ (p = 0.1): c_x, c_y, c_z, m_A, m_B | F | largest error")
    for kind in ("dephasing", "bit flip", "Y flip", "depolarizing", "amplitude damping"):
        rho = noisy_pair(kind, 0.1)
        c = correlations(rho)
        w = bell_weights(*c[:3])
        err = max(ERRORS, key=lambda e: w[e])
        print(f"  {kind:18s} ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}, {c[3]:+.3f}, {c[4]:+.3f}) | F "
              f"{fidelity_from_qg(*c[:3]):.4f} (exact {np.real(PHI_PLUS @ rho @ PHI_PLUS):.4f}) | {err}")
    cases = {"dephasing 0.1": ("dephasing", 0.1), "bit flip 0.1": ("bit flip", 0.1), "Y flip 0.1": ("Y flip", 0.1),
             "depolarizing 0.1": ("depolarizing", 0.1), "amplitude damping 0.2": ("amplitude damping", 0.2)}
    d = distillation_table(cases)
    print("\nB. one DEJMPS round: F_in -> F_out (success prob); fixed = Psi- in slot B; qg rule = smallest "
          "estimated error in slot B, 200 shots per setting")
    for k, v in d.items():
        print(f"  {k:24s} F_in {v['F_in']:.4f} | fixed {v['fixed']:.4f} ({v['p_fixed']:.3f}) | oracle "
              f"[{v['oracle_slot']}] {v['oracle']:.4f} ({v['p_oracle']:.3f}) | qg rule {v['qg_rule']:.4f} "
              f"(right slot {v['qg_right']:.3f})")
    print("\nC. memory cutoff for BBM92 (positive secret fraction), units of T1")
    for r in (2.0, 1.0, 0.5, 0.2):
        print(f"  T2 = {r} T1: no distillation {cutoff(r):.3f}   one DEJMPS round {cutoff(r, True):.3f}")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"))
