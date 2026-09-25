"""
Grover amplitude estimation in qg units: Chebyshev polynomials, a flat
Fisher factor, and what noise does to the quadratic speed-up.

A prepares cos(t)|bad> + sin(t)|good>, amplitude a = sin^2 t. Read the
good/bad flag as a qubit: its qg is qg_0 = 1 - 2a = cos 2t. After k
Grover iterations Q = A S_0 A^dag S_good the flag reads

    qg_k = cos(2 (2k + 1) t) = T_m(qg_0),   m = 2k + 1,

the Chebyshev polynomial of degree m (the same structure as the arccos
encoding of §16). Measuring qg_k with one shot gives, since
T_m'(q) = m U_{m-1}(q) and 1 - T_m(q)^2 = (1 - q^2) U_{m-1}(q)^2,

    F_m(qg_0) = m^2 / (1 - qg_0^2)          (exact, any q)

i.e. depth m multiplies the one-shot qg information of §15.1 by m^2,
uniformly along the fringe. This is the qg form of the known Fisher
information 4 m^2 of the angle t [Suzuki et al., Quantum Inf. Process.
19, 75 (2020)], and it is why maximum-likelihood amplitude estimation
(MLAE) reaches ~1/queries instead of the Monte Carlo 1/sqrt(queries).

Experiment (analytic sampling; a Qiskit circuit check is in tests/):
estimate a with a budget of oracle queries (a shot at depth k costs
2k + 1 applications of A):
  Monte Carlo   depth 0 only
  MLAE          depths k = 0, 1, 2, 4, ..., 2^j (N shots each), maximum
                likelihood over qg_0 in [-1, 1]
Noise: depolarizing noise of strength p per Grover iteration shrinks the
flag towards qg = 0: qg_k = (1 - p)^k T_m(qg_0). Two MLAE variants:
  naive         ignores the noise
  noise-aware   knows p (from a calibration) and uses it in the likelihood

Findings (300 repetitions, 100 shots per depth):

  * F_m (1 - qg_0^2) / m^2 = 1 to machine precision for every depth and
    every qg_0: Grover depth multiplies the qg information by m^2,
    uniformly along the fringe.
  * Noiseless: MLAE error falls as queries^(-0.93 ... -1.12) against
    queries^(-0.5) for Monte Carlo, 8.4-9.3x lower at 26,200 queries
    for a = 0.1, 0.3, 0.5. With only depths 0 and 1 (j = 0) MLAE can
    be worse than Monte Carlo (0.6x at a = 0.3): T_3 is not
    one-to-one, and the likelihood has several peaks.
  * Depolarizing noise p = 0.01 per iteration: MLAE that ignores it
    stops improving and ends worse than Monte Carlo (6.8e-3 vs 2.7e-3
    at 26,200 queries); the noise-aware likelihood keeps going (6.7e-4,
    4x below Monte Carlo).
  * p = 0.05: even noise-aware MLAE plateaus at the Monte Carlo level
    (3.2e-3 vs 2.8e-3): depths beyond ~1/p carry almost no signal
    ((1 - p)^64 = 0.04), and the quadratic speed-up is gone.

Honest summary: the Chebyshev form and the m^2/(1 - qg^2) Fisher factor
are a clean qg reading of known amplitude estimation results (Suzuki et
al. 2020 for MLAE; the noise-aware likelihood is also known). No new
algorithm; the useful practical point is that the noise must be in the
likelihood, and that the advantage ends near depth 1/p.
"""

import math

import numpy as np
from numpy.polynomial import chebyshev as C

A_TRUE = (0.1, 0.3, 0.5)
Q_GRID = np.linspace(-1, 1, 40001)


def chebyshev_t(m, q):
    return C.chebval(q, [0] * m + [1])


def flag_qg(q0, k, p=0.0):
    return (1 - p) ** k * chebyshev_t(2 * k + 1, q0)


def fisher_depth(m, q0):
    """One-shot Fisher information on qg_0 at depth m (noiseless)."""
    tm = chebyshev_t(m, q0)
    d = C.chebval(q0, C.chebder([0] * m + [1]))
    return d**2 / (1 - tm**2)


def schedule(j):
    return [0] + [2**i for i in range(j + 1)]


def queries(ks, shots):
    return shots * sum(2 * k + 1 for k in ks)


def mle(counts, ks, shots, p=0.0):
    """counts[i] = number of +1 (bad) outcomes at depth ks[i]."""
    ll = np.zeros_like(Q_GRID)
    for c, k in zip(counts, ks):
        pp = np.clip((1 + flag_qg(Q_GRID, k, p)) / 2, 1e-15, 1 - 1e-15)
        ll += c * np.log(pp) + (shots - c) * np.log(1 - pp)
    return float(Q_GRID[np.argmax(ll)])


def run(a, j, shots, reps=300, p_true=0.0, p_model=0.0, seed=0):
    """RMSE of a_hat for MLAE with schedule(j) and for Monte Carlo with the
    same number of queries."""
    rng = np.random.default_rng(seed)
    q0 = 1 - 2 * a
    ks = schedule(j)
    n_q = queries(ks, shots)
    err_ae, err_mc = [], []
    for _ in range(reps):
        counts = [rng.binomial(shots, (1 + flag_qg(q0, k, p_true)) / 2) for k in ks]
        a_ae = (1 - mle(counts, ks, shots, p_model)) / 2
        c0 = rng.binomial(n_q, (1 + q0) / 2)
        a_mc = (1 - (2 * c0 / n_q - 1)) / 2
        err_ae.append(a_ae - a)
        err_mc.append(a_mc - a)
    return n_q, float(np.sqrt(np.mean(np.square(err_ae)))), float(np.sqrt(np.mean(np.square(err_mc))))


def scaling(a=0.3, shots=100, js=range(0, 7), p=0.0, aware=True, reps=300):
    rows = []
    for j in js:
        n_q, e_ae, e_mc = run(a, j, shots, reps, p_true=p, p_model=p if aware else 0.0)
        rows.append((j, n_q, e_ae, e_mc))
    return rows


def slope(rows, idx):
    x = np.log([r[1] for r in rows])
    y = np.log([r[idx] for r in rows])
    return float(np.polyfit(x, y, 1)[0])


def make_figure(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    q = np.linspace(-1, 1, 600)
    for m, c in ((1, "#8c8c8c"), (3, "#9ecae1"), (5, "#1f6fb2"), (9, "#08306b")):
        ax1.plot(q, chebyshev_t(m, q), color=c, lw=1.8, label=f"k = {(m - 1) // 2}: T_{m}(qg_0)")
    ax1.set_xlabel("qg_0 = 1 - 2a")
    ax1.set_ylabel("flag qg after k Grover iterations")
    ax1.set_title("Grover iterations = Chebyshev polynomials in qg", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    base = scaling()
    ax2.loglog([r[1] for r in base], [r[3] for r in base], "o-", color="#8c8c8c", label="Monte Carlo")
    ax2.loglog([r[1] for r in base], [r[2] for r in base], "o-", color="#1f6fb2", label="MLAE, noiseless")
    for p, c in ((0.01, "#e0a030"), (0.05, "#c0392b")):
        naive = scaling(p=p, aware=False)
        aware = scaling(p=p, aware=True)
        ax2.loglog([r[1] for r in naive], [r[2] for r in naive], "s--", color=c, label=f"MLAE naive, p = {p}")
        ax2.loglog([r[1] for r in aware], [r[2] for r in aware], "o-", color=c, alpha=0.6,
                   label=f"MLAE noise-aware, p = {p}")
    ax2.set_xlabel("oracle queries")
    ax2.set_ylabel("RMSE of a (a = 0.3)")
    ax2.set_title("100 shots per depth, depths 0, 1, 2, 4, ..., 2^j", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    import sys

    q = np.linspace(-0.95, 0.95, 7)
    for m in (1, 3, 5, 9):
        ratio = fisher_depth(m, q) * (1 - q**2) / m**2
        print(f"F_m (1 - q^2) / m^2 for m = {m}: " + " ".join(f"{x:.6f}" for x in ratio))
    for a in A_TRUE:
        rows = scaling(a=a)
        print(f"\na = {a}: queries, RMSE MLAE, RMSE Monte Carlo")
        for j, n_q, e_ae, e_mc in rows:
            print(f"  j={j}  {n_q:6d}  {e_ae:.2e}  {e_mc:.2e}  (MC/MLAE {e_mc / e_ae:.1f}x)")
        print(f"  log-log slopes: MLAE {slope(rows, 2):.2f}, Monte Carlo {slope(rows, 3):.2f}")
    for p in (0.01, 0.05):
        naive, aware = scaling(p=p, aware=False), scaling(p=p, aware=True)
        print(f"\ndepolarizing p = {p} per iteration, a = 0.3:")
        for (j, n_q, e_n, e_mc), (_, _, e_a, _) in zip(naive, aware):
            print(f"  j={j}  {n_q:6d}  naive {e_n:.2e}  aware {e_a:.2e}  MC {e_mc:.2e}")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"))
