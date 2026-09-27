"""
Error bars for qg_S from finite shots.

§10.2 fixed the BIAS of the joint qg_S estimator (Miller-Madow). What was
still missing is an honest ERROR BAR: the Shannon entropy is flat at its
maximum, so the delta method gives a zero standard error exactly where
qg_S is near 1 (an equatorial qubit, a heavily scrambled register) and the
Wald interval collapses to a point.

Single qubit. The §7 identity qg_S = H((1 + qg_Z)/2) gives a qg-native
fix: take any good interval for qg_Z (Wilson, §15) and map it through H.
H is unimodal with its maximum at qg_Z = 0, so the image is
[min H(ends), 1] if the qg_Z interval contains 0 and [H(end), H(end)]
otherwise; its coverage is at least that of the qg_Z interval
(`qang.statistics.qg_s_estimate`, method "qg_wilson").

Register (4 qubits, 16 outcomes). No such identity; compared are the
delta method around Miller-Madow, the bootstrap, and a Bayesian interval
with the Haar prior. For a Haar-random pure state the Z-basis outcome
distribution is exactly Dirichlet(1, ..., 1) (Porter-Thomas), so that is
the natural prior here, the register analogue of the uniform-in-qg_Z
prior of §15; its posterior mean has the Wolpert-Wolf closed form.

Truth: single qubit at qg_Z = 0, 0.3, 0.6, 0.9, 0.99; register with a
Porter-Thomas distribution mixed with the uniform one (weight lambda =
0, 0.5, 0.9, 0.99, i.e. increasingly depolarized). 95 % intervals,
coverage over 2000 trials (register: 1000).

Findings (python examples/qg_s_error_bars.py):

  Single qubit, coverage of the 95 % interval (target 0.95):

                      N = 20                      N = 100                    N = 1000
    qg_Z   qg_S    Wald boot Bayes qg-Wil |  Wald boot Bayes qg-Wil |  Wald boot Bayes qg-Wil
    0      1.000   1.00 .93  .00   .95    |  1.00 .85  .00   .93    |  1.00 .20  .00   .95
    0.3    0.934   .92  .96  .98   .99    |  .91  .95  .95   .95    |  .94  .96  .95   .96
    0.6    0.722   .85  .93  .95   .95    |  .93  .95  .94   .93    |  .95  .96  .96   .95
    0.9    0.286   .62  .64  .93   .92    |  .95  .95  .96   .96    |  .94  .94  .94   .94
    0.99   0.045   .09  .10  .90   .90    |  .39  .39  .91   .91    |  .95  .95  .96   .96

  (Wald's 1.00 at qg_Z = 0 is clipping at the boundary qg_S = 1, not a
  real interval: its width there is 0.003 at N = 1000.)

  * The qg-mapped Wilson interval is the only one that never fails:
    0.90-0.99 everywhere. Wald and the bootstrap collapse near a pole with
    few shots (9-39 % at qg_Z = 0.99), where the minority outcome is
    rarely seen; the bootstrap also fails at the maximum, qg_Z = 0 (20 %
    at N = 1000), because every resample sits below qg_S = 1; the
    equal-tailed Bayesian interval can never contain the boundary value 1
    (0 %).
  * Point estimate: Miller-Madow has the smallest bias everywhere (at most
    0.02, mostly below 0.005); the plug-in is biased low by about
    1/(2 N ln 2); the Haar-prior posterior mean is pulled towards the
    prior (+0.21 at qg_Z = 0.99 with 20 shots).

  4-qubit register (normalized qg_S), coverage (bias of the point
  estimates at N = 100):

    lambda  qg_S     N = 100: Wald-MM boot-MM Bayes |  N = 1000: Wald-MM boot-MM Bayes
    0       0.858            .91     .81     .97    |            .95     .93     .95
    0.5     0.964            .94     .73     .41    |            .95     .92     .89
    0.9     0.998            .97     .55     .00    |            .97     .77     .00
    0.99    1.000            .97     .50     .00    |            .99     .63     .00

  * For the register the delta method around Miller-Madow works (0.91-
    0.99): with 16 outcomes the sample distribution is never flat enough
    for the delta variance to vanish. The percentile bootstrap fails as
    the register is depolarized (50-77 %), because resampling doubles the
    entropy's downward bias.
  * The Haar (Porter-Thomas) prior is the best choice when the state
    really is Haar-like (0.97 coverage, bias -0.001 at 100 shots, against
    -0.028 for the plug-in) and the worst when the register is near
    uniform (0 %): the posterior cannot reach the maximum. A prior
    helps only when it matches the physics.

Recommendation, now in the library: single qubit, qg_s_estimate(k0, n)
(Miller-Madow point, Wilson interval for qg_Z mapped through the §7
identity); register, Miller-Madow with the delta-method interval. Neither
the bootstrap nor a Haar-prior credible interval should be used for qg_S
near its maximum.

Honest scope. Standard estimators; the contribution is the single-qubit
interval built on the qg_Z <-> qg_S identity and the map of where each
method fails. One Porter-Thomas instance per lambda.
"""

import math
import os
import sys

import numpy as np
from scipy.special import digamma

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qang import statistics as S  # noqa: E402

LN2 = math.log(2)
QG_VALUES = (0.0, 0.3, 0.6, 0.9, 0.99)
SHOTS = (20, 100, 1000)
LAMBDAS = (0.0, 0.5, 0.9, 0.99)
N_REG = 4


def h2v(p):
    p = np.clip(p, 1e-300, 1)
    q = np.clip(1 - p, 1e-300, 1)
    out = -(p * np.log2(p) + q * np.log2(q))
    return np.where((p <= 1e-300) | (q <= 1e-300), 0.0, out)


def entropy_bits(p, axis=-1):
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, -p * np.log2(p), 0.0)
    return t.sum(axis=axis)


# --------------------------------------------------------------------- #
# single qubit
# --------------------------------------------------------------------- #
def bayes_mean_single(k0, n):
    """Posterior mean of H(p) (bits) under the uniform (Haar) prior, Beta(k0+1, n-k0+1)."""
    a, b = k0 + 1.0, n - k0 + 1.0
    return (digamma(a + b + 1) - a / (a + b) * digamma(a + 1) - b / (a + b) * digamma(b + 1)) / LN2


def single_qubit(qg, n, trials=2000, seed=0, boot=400, post=2000):
    rng = np.random.default_rng(seed)
    p0 = (1 + qg) / 2
    truth = float(h2v(np.array(p0)))
    k = rng.binomial(n, p0, size=trials)
    p = k / n
    plug = h2v(p)
    mm = plug + np.where((k > 0) & (k < n), 1 / (2 * n * LN2), 0.0)
    bay = np.array([bayes_mean_single(x, n) for x in k])
    cov = {m: [] for m in ("wald", "bootstrap", "bayes_haar", "qg_wilson")}
    wid = {m: [] for m in cov}
    for x in k:
        rows = {}
        _, lo, hi = S.qg_s_estimate(int(x), n, "wald")
        rows["wald"] = (lo, hi)
        bs = h2v(rng.binomial(n, x / n, size=boot) / n)
        rows["bootstrap"] = tuple(np.quantile(bs, [0.025, 0.975]))
        ps = rng.beta(x + 1, n - x + 1, size=post)
        rows["bayes_haar"] = tuple(np.quantile(h2v(ps), [0.025, 0.975]))
        _, lo, hi = S.qg_s_estimate(int(x), n, "qg_wilson")
        rows["qg_wilson"] = (lo, hi)
        for m, (lo, hi) in rows.items():
            cov[m].append(lo - 1e-12 <= truth <= hi + 1e-12)
            wid[m].append(hi - lo)
    return {
        "truth": truth,
        "bias": {"plug-in": float(plug.mean() - truth), "miller-madow": float(mm.mean() - truth),
                 "bayes_haar": float(bay.mean() - truth)},
        "coverage": {m: float(np.mean(v)) for m, v in cov.items()},
        "width": {m: float(np.mean(v)) for m, v in wid.items()},
    }


# --------------------------------------------------------------------- #
# register
# --------------------------------------------------------------------- #
def register_distribution(lam, seed=7, n=N_REG):
    rng = np.random.default_rng(seed)
    pt = rng.dirichlet(np.ones(2**n))
    return (1 - lam) * pt + lam / 2**n


def bayes_mean_dirichlet(counts, alpha=1.0):
    a = counts + alpha
    A = a.sum()
    return float((digamma(A + 1) - np.sum(a / A * digamma(a + 1))) / LN2)


def register(lam, n_shots, trials=1000, seed=1, boot=400, post=1000, n=N_REG):
    rng = np.random.default_rng(seed)
    p = register_distribution(lam)
    truth = float(entropy_bits(p)) / n
    counts = rng.multinomial(n_shots, p, size=trials)
    est = {"plug-in": [], "miller-madow": [], "bayes_haar": []}
    cov = {m: [] for m in ("wald_mm", "bootstrap_mm", "bayes_haar")}
    wid = {m: [] for m in cov}
    for c in counts:
        ph = c / n_shots
        plug = float(entropy_bits(ph))
        mm = plug + (np.count_nonzero(c) - 1) / (2 * n_shots * LN2)
        bay = bayes_mean_dirichlet(c)
        est["plug-in"].append(plug / n)
        est["miller-madow"].append(mm / n)
        est["bayes_haar"].append(bay / n)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.where(ph > 0, np.log2(ph), 0.0)
        var = max(float(np.sum(ph * lp**2) - plug**2), 0.0) / n_shots
        se = math.sqrt(var)
        rows = {"wald_mm": (mm - 1.96 * se, mm + 1.96 * se)}
        bc = rng.multinomial(n_shots, ph, size=boot)
        bs = entropy_bits(bc / n_shots) + (np.count_nonzero(bc, axis=1) - 1) / (2 * n_shots * LN2)
        rows["bootstrap_mm"] = tuple(np.quantile(bs, [0.025, 0.975]))
        ds = rng.dirichlet(c + 1.0, size=post)
        rows["bayes_haar"] = tuple(np.quantile(entropy_bits(ds), [0.025, 0.975]))
        for m, (lo, hi) in rows.items():
            cov[m].append(lo / n - 1e-12 <= truth <= hi / n + 1e-12)
            wid[m].append((hi - lo) / n)
    return {
        "truth": truth,
        "bias": {m: float(np.mean(v) - truth) for m, v in est.items()},
        "coverage": {m: float(np.mean(v)) for m, v in cov.items()},
        "width": {m: float(np.mean(v)) for m, v in wid.items()},
    }


def make_figure(path, single, reg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    style = {"wald": ("#8c8c8c", "o"), "bootstrap": ("#e0a030", "^"), "bayes_haar": ("#8c2d04", "D"),
             "qg_wilson": ("#1f6fb2", "s")}
    for m, (c, mk) in style.items():
        ax1.plot(QG_VALUES, [single[(q, 100)]["coverage"][m] for q in QG_VALUES], marker=mk, color=c, label=m)
    ax1.axhline(0.95, color="k", lw=0.8, ls=":")
    ax1.set_xlabel("true qg_Z")
    ax1.set_ylabel("coverage of the 95 % interval")
    ax1.set_title("Single qubit, 100 shots", fontsize=9)
    ax1.set_ylim(0, 1.02)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    style2 = {"wald_mm": ("#8c8c8c", "o"), "bootstrap_mm": ("#e0a030", "^"), "bayes_haar": ("#8c2d04", "D")}
    for m, (c, mk) in style2.items():
        ax2.plot(LAMBDAS, [reg[(l, 100)]["coverage"][m] for l in LAMBDAS], marker=mk, color=c, label=f"{m}, 100 shots")
        ax2.plot(LAMBDAS, [reg[(l, 1000)]["coverage"][m] for l in LAMBDAS], marker=mk, color=c, ls="--",
                 label=f"{m}, 1000 shots")
    ax2.axhline(0.95, color="k", lw=0.8, ls=":")
    ax2.set_xlabel("depolarizing weight lambda (Porter-Thomas -> uniform)")
    ax2.set_title("4-qubit register qg_S", fontsize=9)
    ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=6)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    single = {(q, n): single_qubit(q, n) for q in QG_VALUES for n in SHOTS}
    print("Single qubit: coverage (mean width) of 95 % intervals; bias of point estimates")
    for n in SHOTS:
        print(f"  N = {n}")
        for q in QG_VALUES:
            r = single[(q, n)]
            cv = "  ".join(f"{m} {r['coverage'][m]:.3f} ({r['width'][m]:.3f})" for m in r["coverage"])
            bs = "  ".join(f"{m} {v:+.4f}" for m, v in r["bias"].items())
            print(f"    qg {q:4.2f} qg_S {r['truth']:.4f} | {cv} | {bs}")
    reg = {(l, n): register(l, n) for l in LAMBDAS for n in (100, 1000)}
    print("\n4-qubit register (normalized qg_S): coverage (mean width); bias")
    for n in (100, 1000):
        print(f"  N = {n}")
        for l in LAMBDAS:
            r = reg[(l, n)]
            cv = "  ".join(f"{m} {r['coverage'][m]:.3f} ({r['width'][m]:.4f})" for m in r["coverage"])
            bs = "  ".join(f"{m} {v:+.4f}" for m, v in r["bias"].items())
            print(f"    lambda {l:4.2f} qg_S {r['truth']:.4f} | {cv} | {bs}")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"), single, reg)
