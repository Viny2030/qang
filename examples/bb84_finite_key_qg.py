"""
BB84 with finite keys: what a realistic block size costs, and what the qg
diagnosis of §40 adds at those sizes.

Protocol (Tomamichel, Lim, Gisin, Renner, Nat. Commun. 3, 634 (2012)).
The key comes from n sifted bits in one basis (here Z, the basis in which
T1 errors appear), the phase error is estimated from k bits in the other
basis, and the protocol aborts if the test error rate lambda exceeds a
tolerance Q_tol fixed in advance. It is eps-secure with key length

    l = n [1 - h(Q_tol + mu)] - leak_EC - log2( 2 / (eps_sec^2 eps_cor) ),
    mu = sqrt( (n + k)/(n k) * (k + 1)/k * ln(2/eps_sec) ),

leak_EC = xi n h(Q_key) with xi = 1.1, and eps_sec = eps_cor = 5e-11
(eps = 1e-10). With basis probabilities chosen to minimise the number of
signals, M = (sqrt(n) + sqrt(k))^2, and the rate is
r = (1 - eps_rob) l / M, with eps_rob = P(lambda > Q_tol) the abort
probability on the expected channel (exact binomial tail). k and Q_tol
are optimised for each n.

Channel: the §40 model (gamma0 = 0.02, p0 = 0.01, detector e01 = 0.005,
e10 = 0.01): Q_Z = 1.74 %, Q_X = 2.22 % on the baseline.

What qg adds, at no key cost. After error correction and its
verification, Bob holds Alice's key bits, so he knows the four error
counts (0->1, 1->0 on all n key bits; +->-, -->+ on the k test bits)
without disclosing anything; announcing one alarm bit costs at most one
key bit. When a block aborts, its key bits are discarded anyway, so Alice
can announce them and Bob gets the same counts. Two nested likelihood
ratios on the joint model (T1 damping gamma and intercept-resend f, both
free) are computed on these counts:

    attack-like   symmetric (intercept-resend-like) error beyond any T1
    drift-like    T1 beyond the baseline, whatever the attack

each calibrated to 1 % false alarms on the baseline for the same (n, k).
The four counts are the four qg values of §40 (qg(0), qg(1), qg(+),
qg(-)); the drift flag is driven by the T1 witness A_Z = qg(0) + qg(1),
which a symmetric attack leaves unchanged.

Findings (python examples/bb84_finite_key_qg.py):

  1. Finite-key cost. Minimum block size for a positive key, and the
     optimised rate at n = 1e4 ... 1e7 key bits (asymptotic in the last
     column):

                          n_min    1e4    1e5    1e6    1e7    asym
     baseline             1.2e3   0.120  0.268  0.407  0.516  0.707
     T1 drift 0.04        1.6e3   0.091  0.223  0.348  0.446  0.622
     T1 drift 0.06        2.0e3   0.068  0.182  0.294  0.383  0.544
     intercept f = 0.05   1.7e3   0.079  0.201  0.318  0.410  0.575

     A realistic block keeps 17 % (1e4), 38 % (1e5), 58 % (1e6) and 73 %
     (1e7) of the asymptotic rate. T1 drift and a naive attack cost key
     alike: security charges every error to Eve.

  2. The operator's problem. A protocol tuned on the baseline at
     n = 1e5 (k = 7609, Q_tol = 2.67 %) aborts in 1 % of baseline blocks,
     but in 10 %, 56 % and 99.5 % of blocks under T1 drift 0.03, 0.04 and
     0.06, and in 50 % under a naive intercept-resend at f = 0.02. The
     abort alone does not say why.

  3. The qg diagnosis of the same blocks (fraction of 400 blocks):

                                abort   attack-like  drift-like
     baseline                   0.010   0.020        0.015
     T1 drift 0.03              0.100   0.005        1.000
     T1 drift 0.04              0.557   0.005        1.000
     T1 drift 0.06              0.995   0.007        1.000
     intercept f = 0.02         0.495   1.000        0.043
     intercept f = 0.05         1.000   1.000        0.040
     drift 0.04 + intercept .02 0.995   1.000        1.000
     T1-mimicking attack        0.995   0.013        1.000

     At a realistic block size the attribution is essentially exact:
     every drifting block, aborted or not, is flagged as drift and not
     as an attack, and every intercept-resend block is flagged as an
     attack, including the half that did not abort. At n = 1e4 (k = 3353)
     it is already 0.97-1.00 for drift >= 0.04 and intercept f >= 0.02
     (0.81 for drift 0.03).
  4. An attack hidden under drift is caught. The statistics are nested
     likelihood ratios on the joint (gamma, f) model: "attack-like" asks
     for symmetric error beyond ANY T1 drift, "drift-like" for T1 beyond
     the baseline whatever the attack. A small intercept-resend (f = 0.02)
     on a drifting memory (gamma = 0.04) raises both flags in 100 % of
     blocks (97 % at n = 1e4). The one-parameter GLRT of §40 (attack vs
     baseline or pure drift) misses it: 0.5 % attack flags, 99 % "drift".
  5. Early warning. At gamma = 0.03 only 10 % of blocks abort, but the
     drift flag is up in every block, so the memory can be serviced
     before the link stops. A Q_Z trend on the same counts would also
     warn; what qg adds is the attribution (T1-like vs symmetric).

Honest scope. The key length is the standard one: qg does not change
it, and every error is still charged to Eve. An attacker who mimics T1
exactly is flagged as "drift", by construction. The diagnosis costs no
key bits (at most one announced bit per block). Only two error families
(T1 and intercept-resend) are modelled.
"""

import math

import numpy as np
from scipy.stats import binom

import bb84_qg_eve_vs_noise as B

EPS = 1e-10
EPS_SEC = EPS_COR = EPS / 2
XI = 1.1
N_GRID = (1e4, 1e5, 1e6, 1e7)

SCENARIOS = [("baseline", {}), ("T1 drift 0.03", {"gamma": 0.03}), ("T1 drift 0.04", {"gamma": 0.04}),
             ("T1 drift 0.06", {"gamma": 0.06}), ("intercept f = 0.02", {"f": 0.02}),
             ("intercept f = 0.05", {"f": 0.05}), ("drift 0.04 + intercept 0.02", {"gamma": 0.04, "f": 0.02}),
             ("T1-mimicking attack", {"gamma": 0.06})]


def h(x):
    if x <= 0:
        return 0.0
    if x >= 0.5:
        return 1.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def mu(n, k, eps_sec=EPS_SEC):
    return math.sqrt((n + k) / (n * k) * (k + 1) / k * math.log(2 / eps_sec))


def key_length(n, k, q_tol, q_key):
    return n * (1 - h(q_tol + mu(n, k))) - XI * n * h(q_key) - math.log2(2 / (EPS_SEC**2 * EPS_COR))


def eps_rob(k, q_test, q_tol):
    return float(binom.sf(math.floor(q_tol * k), int(k), q_test))


def signals(n, k):
    return (math.sqrt(n) + math.sqrt(k)) ** 2


def rate(n, k, q_tol, q_key, q_test):
    ell = key_length(n, k, q_tol, q_key)
    if ell <= 0:
        return 0.0
    return (1 - eps_rob(k, q_test, q_tol)) * ell / signals(n, k)


def asymptotic_rate(q_key, q_test):
    return max(0.0, 1 - h(q_test) - XI * h(q_key))


def optimize(n, q_key, q_test):
    """Best (rate, k, Q_tol) on a grid; k up to n, Q_tol above the expected test error."""
    best = (0.0, None, None)
    for k in np.unique(np.round(np.logspace(2, math.log10(n), 60))):
        for q_tol in q_test + np.linspace(0.0, 0.08, 161):
            r = rate(n, k, q_tol, q_key, q_test)
            if r > best[0]:
                best = (r, int(k), float(q_tol))
    return best


def channel(**kw):
    e = B.error_probabilities(**kw)
    q_z, q_x = B.qber(e)
    return e, q_z, q_x


def n_min(q_key, q_test, lo=1e2, hi=1e7):
    """Smallest n (to ~3 %) with a positive optimised rate."""
    if optimize(hi, q_key, q_test)[0] <= 0:
        return math.inf
    while hi / lo > 1.03:
        mid = math.sqrt(lo * hi)
        if optimize(mid, q_key, q_test)[0] > 0:
            hi = mid
        else:
            lo = mid
    return hi


# --------------------------------------------------------------------- #
# qg diagnosis on post-error-correction counts
# --------------------------------------------------------------------- #
G2 = np.round(np.arange(0.0, 0.2001, 0.0025), 4)
F2 = np.round(np.arange(0.0, 0.3001, 0.0025), 4)
_JOINT = np.array([[B.error_probabilities(gamma=g, f=f) for f in F2] for g in G2])  # (gamma, f, 4)
_I_G0 = int(np.argmin(np.abs(G2 - B.BASE["gamma"])))


def _loglik_grid(kc, nc):
    pr = np.clip(_JOINT, 1e-12, 1 - 1e-12)
    return (kc * np.log(pr) + (nc - kc) * np.log(1 - pr)).sum(axis=-1)


def statistics(kc, nc):
    """Two nested likelihood-ratio statistics on the joint (gamma, f) model:
    attack-like = extra symmetric error beyond any T1 drift (f > 0, gamma free);
    drift-like  = extra T1 beyond the baseline, whatever the attack (gamma free vs gamma0)."""
    ll = _loglik_grid(np.asarray(kc), np.asarray(nc))
    best = ll.max()
    return float(best - ll[:, 0].max()), float(best - ll[_I_G0, :].max())


def block(probs, n, k, rng):
    """Error counts of one block: n key bits (Z), k test bits (X)."""
    nc = np.array([n // 2, n - n // 2, k // 2, k - k // 2])
    return rng.binomial(nc, probs), nc


def calibrate(n, k, reps=400, seed=0):
    rng = np.random.default_rng(seed)
    s = np.array([statistics(*block(B._BASE_PROBS, n, k, rng)) for _ in range(reps)])
    return float(np.quantile(s[:, 0], 0.99)), float(np.quantile(s[:, 1], 0.99))


def diagnose(n=100000, reps=400, seed=1):
    """Tune the protocol on the baseline at this n, then run every scenario."""
    _, q_z, q_x = channel()
    _, k, q_tol = optimize(n, q_z, q_x)
    th = calibrate(n, k, reps)
    rng = np.random.default_rng(seed)
    out = {"k": k, "q_tol": q_tol}
    for name, kw in SCENARIOS:
        probs = B.error_probabilities(**kw)
        ab = at = dr = 0
        for _ in range(reps):
            kc, nc = block(probs, n, k, rng)
            ab += kc[2:].sum() / nc[2:].sum() > q_tol
            sa, sd = statistics(kc, nc)
            at += sa > th[0]
            dr += sd > th[1]
        out[name] = {"abort": ab / reps, "attack": at / reps, "drift": dr / reps}
    return out


def rate_table():
    out = {}
    for name, kw in SCENARIOS:
        _, q_z, q_x = channel(**kw)
        out[name] = {"asym": asymptotic_rate(q_z, q_x), "n_min": n_min(q_z, q_x)}
        for n in N_GRID:
            out[name][n] = optimize(n, q_z, q_x)
    return out


def make_figure(path, diag):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ns = np.logspace(3, 7.5, 28)
    style = {"baseline": ("#1f6fb2", "-"), "T1 drift 0.06": ("#e0a030", "--"),
             "intercept f = 0.05": ("#8c2d04", ":")}
    for name, (col, ls) in style.items():
        _, q_z, q_x = channel(**dict(SCENARIOS)[name])
        ax1.plot(ns, [optimize(n, q_z, q_x)[0] for n in ns], ls, color=col, lw=2, label=name)
        ax1.axhline(asymptotic_rate(q_z, q_x), color=col, lw=0.8, alpha=0.6)
    ax1.set_xscale("log")
    ax1.set_xlabel("key bits per block, n")
    ax1.set_ylabel("secret key per signal")
    ax1.set_title("Finite-key rate (eps = 1e-10); thin lines: asymptotic", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    names = [n for n, _ in SCENARIOS]
    short = ["baseline", "drift\n0.03", "drift\n0.04", "drift\n0.06", "IR\nf=.02", "IR\nf=.05", "drift .04\n+ IR .02", "mimic\nattack"]
    x = np.arange(len(names))
    for off, key, col, lab in ((-0.27, "abort", "#8c8c8c", "protocol aborts"),
                               (0.0, "attack", "#8c2d04", "qg: attack-like"),
                               (0.27, "drift", "#e0a030", "qg: drift-like")):
        ax2.bar(x + off, [diag[n][key] for n in names], 0.26, color=col, label=lab)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, fontsize=7)
    ax2.set_ylabel("fraction of blocks")
    ax2.set_title("Protocol tuned on the baseline, n = 1e5: abort vs qg diagnosis", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    import sys

    t = rate_table()
    print(f"{'scenario':22s} {'n_min':>8} " + " ".join(f"{n:>8.0e}" for n in N_GRID) + "     asym")
    for name, _ in SCENARIOS:
        r = t[name]
        print(f"{name:22s} {r['n_min']:8.2e} " + " ".join(f"{r[n][0]:8.3f}" for n in N_GRID) + f" {r['asym']:8.3f}")
    for n in (10000, 100000):
        d = diagnose(n)
        print(f"\nn = {n}: tuned k = {d['k']}, Q_tol = {d['q_tol']:.4f}")
        for name, _ in SCENARIOS:
            v = d[name]
            print(f"  {name:22s} abort {v['abort']:.3f}  attack-like {v['attack']:.3f}  drift-like {v['drift']:.3f}")
        if n == 100000 and "--figure" in sys.argv:
            make_figure(__file__.replace(".py", ".png"), d)
