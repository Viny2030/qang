"""
Few-shot single-qubit tomography in qg units: is the Haar (uniform-in-qg)
prior per axis better than the standard estimators?

A qubit state is its Bloch vector r = (qg_X, qg_Y, qg_Z), |r| <= 1. With
N shots in each Pauli basis and k_a outcomes "+1" on axis a:

  LI         linear inversion r_a = 2 k_a / N - 1 (can leave the ball)
  LI + proj  the same, rescaled onto the ball when |r| > 1
  MLE        maximum likelihood over the ball
  qg-Haar    per-axis posterior mean under the Haar prior, which is
             uniform in each qg (Archimedes, §15.3):
             r_a = (2 k_a - N)/(N + 2), then projected onto the ball
  Jeffreys   per-axis Jeffreys posterior mean: r_a = (2 k_a - N)/(N + 1),
             projected (the control asked for by §25)
  Bayes      full Bloch-ball posterior mean under the prior that matches
             the test ensemble (uniform sphere for pure, uniform ball for
             mixed), by importance sampling: the best possible for MSE
             when the prior is right

Test ensembles: Haar-random pure states, uniform-in-ball mixed states,
pure states near a pole (polar angle < 0.3 rad), and strongly mixed
states (|r| < 0.3). Loss: mean squared Bloch error E|r_hat - r|^2
(= 2 x squared Hilbert-Schmidt distance); for pure states also the
infidelity of the estimated direction, (1 - r . r_hat / |r_hat|)/2.

Findings (2000 states per cell; mean squared Bloch error; Bayes by
importance sampling, reported up to 100 shots):

  10 shots per axis    LI     LI+proj  MLE    qg-Haar  Jeffreys  Bayes
  Haar pure           0.203   0.157   0.150   0.165    0.159    0.143
  uniform mixed       0.247   0.211   0.210   0.185    0.196    0.171
  pure near pole      0.199   0.159   0.109   0.163    0.159    0.036
  strongly mixed      0.298   0.293   0.293   0.208    0.245    0.046

  * Linear inversion is unphysical in 52-88% of runs on pure states;
    every other estimator is physical by construction.
  * The per-axis qg-Haar estimator is the best of the cheap closed
    forms on mixed states: 12% below MLE on uniform mixed states and
    29% below it on strongly mixed ones at 10 shots, and 6-15% below
    Jeffreys. It shrinks towards the centre, which is right for mixed
    states.
  * On pure states that shrinkage is wrong: qg-Haar is 10% worse than
    MLE on Haar pure states and 50% worse near a pole. The per-axis
    prior is the Haar marginal, not the joint Haar prior, which lives
    on the sphere; the full-sphere Bayes estimator uses it and is the
    best on pure states.
  * By 100 shots the physical estimators agree within ~10%, except near
    a pole, where MLE stays 25% ahead of the others (still 15% at 1000
    shots): there the truth sits on the boundary of the ball.
  * A mismatched prior is expensive: Bayes with a ball prior on
    near-pole pure states gives 0.145 (matched 0.036), and a sphere
    prior on strongly mixed states gives 0.591 (matched 0.046, LI 0.298).

Same verdict as §15.3 and §25: at few shots the prior does the work,
not the qg coordinates. qg-Haar is a good, free default when the state
is expected to be mixed (noisy hardware); MLE or a sphere prior when it
is expected to be pure.
"""

import math

import numpy as np
from scipy.optimize import minimize

ESTIMATORS = ("LI", "LI+proj", "MLE", "qg-Haar", "Jeffreys", "Bayes")
ENSEMBLES = ("Haar pure", "uniform mixed", "pure near pole", "strongly mixed")


def sample_states(ensemble, m, rng):
    if ensemble == "Haar pure":
        v = rng.normal(size=(m, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    if ensemble == "uniform mixed":
        v = rng.normal(size=(m, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v * rng.random((m, 1)) ** (1 / 3)
    if ensemble == "pure near pole":
        th = np.arccos(rng.uniform(math.cos(0.3), 1.0, m))
        ph = rng.uniform(0, 2 * math.pi, m)
        return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    if ensemble == "strongly mixed":
        v = rng.normal(size=(m, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v * 0.3 * rng.random((m, 1)) ** (1 / 3)
    raise ValueError(ensemble)


def _project(r):
    n = np.linalg.norm(r, axis=-1, keepdims=True)
    return np.where(n > 1, r / np.maximum(n, 1e-300), r)


def _mle_one(k, n):
    """Constrained MLE for one state (k: 3 counts of +1 out of n)."""
    li = 2 * k / n - 1
    if np.linalg.norm(li) <= 1:
        return li

    def nll(ang):
        th, ph = ang
        r = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
        p = np.clip((1 + r) / 2, 1e-12, 1 - 1e-12)
        return -float((k * np.log(p) + (n - k) * np.log(1 - p)).sum())

    d = li / np.linalg.norm(li)
    x0 = [math.acos(np.clip(d[2], -1, 1)), math.atan2(d[1], d[0])]
    res = minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10})
    th, ph = res.x
    return np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])


def _prior_samples(kind, m, rng):
    return sample_states("Haar pure" if kind == "sphere" else "uniform mixed", m, rng)


def bayes_mean(k, n, prior, chunk=500):
    """Posterior mean of r for counts k (reps x 3) under prior samples."""
    p = np.clip((1 + prior) / 2, 1e-12, 1 - 1e-12)  # (M, 3)
    lp, lq = np.log(p), np.log(1 - p)
    out = np.empty((len(k), 3))
    for s in range(0, len(k), chunk):
        kk = k[s:s + chunk]
        ll = kk @ lp.T + (n - kk) @ lq.T  # (c, M)
        ll -= ll.max(axis=1, keepdims=True)
        w = np.exp(ll)
        out[s:s + chunk] = (w @ prior) / w.sum(axis=1, keepdims=True)
    return out


def estimate_all(k, n, prior):
    li = 2 * k / n - 1
    return {
        "LI": li,
        "LI+proj": _project(li),
        "MLE": np.array([_mle_one(kk, n) for kk in k]),
        "qg-Haar": _project((2 * k - n) / (n + 2)),
        "Jeffreys": _project((2 * k - n) / (n + 1)),
        # importance sampling needs the prior to cover the posterior: fine up to
        # ~100 shots with 6000 samples, not at 1000 (posterior width ~0.03)
        "Bayes": bayes_mean(k, n, prior) if n <= 100 else np.full_like(li, np.nan),
    }


def benchmark(ensemble, n_shots, reps=2000, seed=0, prior_size=6000):
    rng = np.random.default_rng(seed)
    r = sample_states(ensemble, reps, rng)
    k = rng.binomial(n_shots, (1 + r) / 2).astype(float)
    prior_kind = "sphere" if "pure" in ensemble else "ball"
    prior = _prior_samples(prior_kind, prior_size, rng)
    if ensemble == "strongly mixed":
        prior = sample_states("strongly mixed", prior_size, rng)
    if ensemble == "pure near pole":
        prior = sample_states("pure near pole", prior_size, rng)
    est = estimate_all(k, n_shots, prior)
    out = {}
    for name, e in est.items():
        mse = float(np.mean(np.sum((e - r) ** 2, axis=1)))
        row = {"mse": mse, "unphysical": float(np.mean(np.linalg.norm(e, axis=1) > 1 + 1e-9))}
        if "pure" in ensemble:
            d = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-300)
            row["infidelity"] = float(np.mean((1 - np.sum(d * r, axis=1)) / 2))
        out[name] = row
    return out


def mismatched_prior(n_shots=10, reps=2000, seed=0):
    """Bayes with a uniform-ball prior on near-pole pure states, and with a
    sphere prior on strongly mixed states (prior not matching the truth)."""
    rng = np.random.default_rng(seed)
    res = {}
    for ens, kind in (("pure near pole", "ball"), ("strongly mixed", "sphere")):
        r = sample_states(ens, reps, rng)
        k = rng.binomial(n_shots, (1 + r) / 2).astype(float)
        e = bayes_mean(k, n_shots, _prior_samples(kind, 6000, rng))
        res[ens] = float(np.mean(np.sum((e - r) ** 2, axis=1)))
    return res


SHOTS = (10, 30, 100, 1000)


def make_figure(tables, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"LI": "#bdbdbd", "LI+proj": "#8c8c8c", "MLE": "#e0a030", "qg-Haar": "#1f6fb2",
           "Jeffreys": "#9ecae1", "Bayes": "#2e8b57"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
    for ax, ens in zip(axes, ENSEMBLES):
        for e in ESTIMATORS:
            ax.loglog(SHOTS, [tables[(ens, n)][e]["mse"] if np.isfinite(tables[(ens, n)][e]["mse"]) else np.nan
                              for n in SHOTS], "o-", color=col[e], lw=2 if e == "qg-Haar" else 1.3,
                      label=e + (" (matched prior)" if e == "Bayes" else ""))
        ax.set_title(ens, fontsize=9)
        ax.set_xlabel("shots per axis")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("mean squared Bloch error")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    import sys

    tables = {}
    for ens in ENSEMBLES:
        print(f"\n{ens}: mean squared Bloch error (fraction unphysical)")
        print(f"{'shots':>6} " + "".join(f"{e:>17}" for e in ESTIMATORS))
        for n in SHOTS:
            t = tables[(ens, n)] = benchmark(ens, n)
            print(f"{n:6d} " + "".join(f"{t[e]['mse']:9.4f} ({t[e]['unphysical']:.2f})" for e in ESTIMATORS))
        if "pure" in ens:
            print("  direction infidelity:")
            for n in SHOTS:
                t = tables[(ens, n)]
                print(f"{n:6d} " + "".join(f"{t[e]['infidelity']:17.4f}" for e in ESTIMATORS))
    print("\nBayes with a mismatched prior (10 shots):", mismatched_prior())
    if "--figure" in sys.argv:
        make_figure(tables, __file__.replace(".py", ".png"))
