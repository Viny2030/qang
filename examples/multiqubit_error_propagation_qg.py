"""
§5 beyond one pure qubit: shot-noise propagation for mixed states and for
multi-qubit registers (qang.statistics).

A. Mixed qubit. §5 found Var(theta_hat) = 1/N for every theta: the
   vanishing shot noise at a pole exactly cancels the diverging Jacobian
   -1/sin(theta). For a qubit with Bloch length r < 1 (a noisy qubit, or
   one qubit of an entangled register) the outcome is never certain, so
   the cancellation fails:

       Var(theta_hat) = [1 + (1 - r^2) / (r^2 sin^2 theta)] / N,

   which equals the quantum Cramer-Rao bound 1/(N r^2) only at the
   equator and diverges at the poles. (It is the inverse of the Ramsey
   Fisher information of §36, F = (V^2 - qg^2)/(1 - qg^2) with V = r.)

B. Register. The per-qubit qg_i of an n-qubit register come from the
   same joint shots, so their estimators are correlated:

       Cov(qg_i_hat, qg_j_hat) = (<Z_i Z_j> - qg_i qg_j) / N,

   and any aggregate (the symmetry witness of §20-§30, a parity, an
   energy) needs the full covariance. The naive error bar (independent
   qubits) can be wrong in either direction: for a GHZ state it is n
   times too small, for a fixed-particle-number (Dicke) state the true
   variance of the witness is exactly zero.

C. Consequence for the symmetry witness. On a register that should hold
   exactly K excitations (the §20-§30 filters), the witness mean(qg_i)
   has no shot noise at all except from the leak itself, so a small
   leak is detected with far fewer shots than the naive error bar says.

Findings (python examples/multiqubit_error_propagation_qg.py; Monte Carlo
from multinomial sampling):

  A. N * Var(theta_hat), N = 1e4, 4000 trials (delta formula / sampled,
     % of trials clipped at the pole):

      theta     r = 1           r = 0.95             r = 0.8
      pi/2      1.00 / 0.99     1.11 / 1.10          1.56 / 1.55
      pi/4      1.00 / 1.01     1.22 / 1.22          2.13 / 2.13
      pi/8      1.00 / 1.00     1.74 / 1.74          4.84 / 4.90
      pi/16     1.00 / 1.01     3.84 / 3.93 (0 %)    15.8 / 19.4 (1 %)
      pi/64     1.00 / 1.11     45.9 / 18.6 (38 %)   235 / 38 (45 %)

     The cancellation is a pure-state property: with 5 % less purity the
     angular error at pi/16 is already twice as large (variance x 3.8),
     and at pi/64 the estimator sticks to the pole in 38-45 % of the
     trials (the sampled variance is then smaller than the formula only
     because the estimate is biased onto the pole). At the equator the
     error equals the Cramer-Rao bound 1/(N r^2).

  B. 4 qubits, N = 1000 shots, std of the register witness mean(qg_i)
     (x 1e-3), correct (covariance) / naive (independent) / sampled:

      product, theta = pi/3     13.7 / 13.7 / 13.8
      GHZ                       31.6 / 15.8 / 32.1
      W state                    0.0 / 13.7 /  0.0
      Dicke D(4, 2)              0.0 / 15.8 /  0.0
      D(4, 1) + 2 % bit flips    4.4 / 13.9 /  4.4

  C. Leak detection with the witness on D(4, 1) (target mean(qg) = 0.5),
     bit-flip rate 2 % per qubit: witness shift 0.020. Shots for a 3-sigma
     detection: 441 with the correct error bar, 4329 with the naive one
     (about 10 times more). If the full bitstrings are kept, counting the
     shots with the wrong excitation number (the kept fraction of the qg
     filter) is better still: 7.6 % of shots leak, so on an ideal device
     the first wrong-weight shot (about 13 shots) is conclusive; with
     readout errors the null rate is not zero and the count needs its own
     calibration. The witness error bar matters when only aggregate or
     per-qubit averages are recorded.

Honest scope. These are standard delta-method statistics; the content
is where §5's statement stops holding (mixed states) and which
covariance the register witnesses need. The leak numbers use simple
independent bit flips.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qang import statistics as S  # noqa: E402

THETAS = (("pi/2", math.pi / 2), ("pi/4", math.pi / 4), ("pi/8", math.pi / 8), ("pi/16", math.pi / 16),
          ("pi/64", math.pi / 64))
RADII = (1.0, 0.95, 0.8)
N_QUBITS = 4


# --------------------------------------------------------------------- #
# A. mixed qubit
# --------------------------------------------------------------------- #
def sampled_theta_variance(theta, r, n_shots, trials=4000, seed=0):
    rng = np.random.default_rng(seed)
    p0 = (1 + r * math.cos(theta)) / 2
    q = 2 * rng.binomial(n_shots, p0, size=trials) / n_shots - 1
    x = q / r
    clipped = np.mean(np.abs(x) >= 1)
    th = np.arccos(np.clip(x, -1, 1))
    return float(np.var(th)), float(clipped)


def table_mixed(n_shots=10000, trials=4000):
    out = {}
    for name, th in THETAS:
        for r in RADII:
            v, c = sampled_theta_variance(th, r, n_shots, trials)
            out[(name, r)] = (n_shots * S.propagated_theta_variance_mixed(th, r, n_shots), n_shots * v, c)
    return out


# --------------------------------------------------------------------- #
# B. registers
# --------------------------------------------------------------------- #
def _weight(n):
    return np.array([bin(i).count("1") for i in range(2 ** n)])


def product_state(theta, n=N_QUBITS):
    p1 = math.sin(theta / 2) ** 2
    w = _weight(n)
    return p1 ** w * (1 - p1) ** (n - w)


def ghz(n=N_QUBITS):
    p = np.zeros(2 ** n)
    p[0] = p[-1] = 0.5
    return p


def dicke(k, n=N_QUBITS):
    w = _weight(n)
    p = (w == k).astype(float)
    return p / p.sum()


def bit_flips(p, eta, n=N_QUBITS):
    """Independent bit flips with probability eta on every qubit."""
    p = np.asarray(p, dtype=float).copy()
    idx = np.arange(2 ** n)
    for i in range(n):
        p = (1 - eta) * p + eta * p[idx ^ (1 << i)]
    return p


STATES = [("product, theta = pi/3", lambda: product_state(math.pi / 3)), ("GHZ", ghz), ("W state", lambda: dicke(1)),
          ("Dicke D(4, 2)", lambda: dicke(2)), ("D(4, 1) + 2 % bit flips", lambda: bit_flips(dicke(1), 0.02))]


def sampled_witness_std(p, n_shots, trials=4000, seed=1, n=N_QUBITS):
    rng = np.random.default_rng(seed)
    z = S._z_values(n)
    counts = rng.multinomial(n_shots, p, size=trials)  # (trials, 2^n)
    q = counts @ z.T / n_shots  # (trials, n)
    return float(np.std(q.mean(axis=1)))


def table_registers(n_shots=1000, trials=4000):
    out = {}
    for name, f in STATES:
        p = f()
        out[name] = (math.sqrt(S.register_witness_variance(p, N_QUBITS, n_shots)),
                     math.sqrt(S.register_witness_variance(p, N_QUBITS, n_shots, independent=True)),
                     sampled_witness_std(p, n_shots, trials))
    return out


# --------------------------------------------------------------------- #
# C. leak detection
# --------------------------------------------------------------------- #
def shots_for_detection(eta=0.02, k=1, sigmas=3.0, n=N_QUBITS):
    """Shots for the witness shift to reach `sigmas` standard errors, with
    the correct and the naive (independent-qubit) error bars."""
    target = 1 - 2 * k / n
    p = bit_flips(dicke(k, n), eta, n)
    q, _ = S.qg_covariance(p, n, 1)
    shift = abs(q.mean() - target)
    v_true = S.register_witness_variance(p, n, 1)
    v_naive = S.register_witness_variance(p, n, 1, independent=True)
    return shift, math.ceil(sigmas**2 * v_true / shift**2), math.ceil(sigmas**2 * v_naive / shift**2)


def make_figure(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    th = np.linspace(0.02, math.pi / 2, 300)
    cols = {1.0: "#1f6fb2", 0.95: "#e0a030", 0.8: "#8c2d04"}
    for r in RADII:
        ax1.plot(th, [S.propagated_theta_variance_mixed(t, r, 1) for t in th], color=cols[r], lw=2, label=f"r = {r}")
        ax1.axhline(1 / r**2, color=cols[r], lw=0.8, ls=":")
    ax1.set_yscale("log")
    ax1.set_ylim(0.8, 300)
    ax1.set_xlabel("polar angle theta (rad)")
    ax1.set_ylabel("N * Var(theta_hat)")
    ax1.set_title("§5 cancellation holds only for pure states (dotted: Cramer-Rao 1/r^2)", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    t = table_registers(trials=2000)
    names = [n for n, _ in STATES]
    x = np.arange(len(names))
    ax2.bar(x - 0.2, [1e3 * t[n][0] for n in names], 0.4, color="#1f6fb2", label="correct (covariance)")
    ax2.bar(x + 0.2, [1e3 * t[n][1] for n in names], 0.4, color="#8c8c8c", label="naive (independent qubits)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["product", "GHZ", "W", "Dicke\nD(4,2)", "D(4,1)\n+2% flips"], fontsize=8)
    ax2.set_ylabel("std of mean(qg_i) x 1e3, N = 1000")
    ax2.set_title("Register witness: error bar with and without correlations", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    import sys

    a = table_mixed()
    print("A. N * Var(theta_hat), N = 1e4: delta formula / sampled (clipped fraction)")
    for name, _ in THETAS:
        print(f"  {name:6s} " + "   ".join(f"r={r}: {a[(name, r)][0]:7.2f} / {a[(name, r)][1]:6.2f} ({a[(name, r)][2]:.2f})"
                                          for r in RADII))
    b = table_registers()
    print("\nB. std of the register witness x 1e3 (N = 1000): correct / naive / sampled")
    for name, _ in STATES:
        print(f"  {name:28s} {1e3 * b[name][0]:6.1f} / {1e3 * b[name][1]:6.1f} / {1e3 * b[name][2]:6.1f}")
    shift, n_true, n_naive = shots_for_detection()
    print(f"\nC. leak on D(4,1), 2 % bit flips: shift {shift:.4f}; shots for 3 sigma: {n_true} (correct), {n_naive} (naive)")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"))
