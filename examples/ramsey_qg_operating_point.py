"""
Ramsey phase sensing in qg units: where to operate, and what few shots do.

A Ramsey sequence turns a phase phi into the measured
    qg = a + b V cos(phi - alpha),
with fringe visibility V (dephasing), a control phase alpha (the
operating point), and the affine readout map a, b of §31.

A. Fisher information per shot. For a Bernoulli outcome with mean qg,
F_qg = 1/(1 - qg^2) (§15.1), and dqg/dphi = -b V sin x, x = phi - alpha:

    F(phi) = b^2 V^2 sin^2 x / (1 - qg^2).

With a = 0 and b = 1 this is
    F = (V^2 - qg^2)/(1 - qg^2),
a one-line function of the operating qg: the 1/(1 - qg^2) gain near the
poles is exactly cancelled by the Jacobian when V = 1 (F = 1 at every
operating point), and with V < 1 the optimum is the mid-fringe qg = 0
(the known rule) while F -> 0 at the bright fringe. With an asymmetric
readout (a != 0, from §31) the optimum moves off quadrature; it is
found numerically in optimal_operating_point().

B. Few shots. The Cramer-Rao bound is flat in the operating point when
V = 1, but a finite number of shots is not: near a pole all shots
often give the same outcome and the arccos inversion sticks at the pole.
We estimate a small phase phi in [0, 0.5] rad (a weak signal) with N
shots at a fixed operating point:
  "pole"        alpha = 0 (qg near V: bright fringe)
  "mid-fringe"  alpha = -pi/2 (qg near 0)
and two estimators: plug-in inversion (arccos / arcsin of the measured
qg) and the Bayesian posterior mean with a uniform prior on [0, 0.5].
Score: sqrt(N) x RMSE of phi, against the Cramer-Rao value 1/sqrt(F).

Findings:

  * F = (V^2 - qg^2)/(1 - qg^2) is exact, and it reproduces the known
    rule in one line: with V = 1 every operating point gives F = 1;
    with V < 1 the mid-fringe (qg = 0) is optimal, and F -> 0 at the
    bright fringe.
  * Asymmetric readout moves the optimum off quadrature, but the gain
    is negligible: +0.17% with the §31 readout (e01 0.015, e10 0.04,
    V = 0.9, optimum at qg* = +0.09) and +1.6% even with e10 = 0.10.
  * Few shots, V = 1 (phi in [0, 0.5] rad, sqrt(N) x RMSE, CRB = 1):
    plug-in inversion is within 0.89-1.12 at both operating points,
    although at the pole 82% of 10-shot runs give identical outcomes;
    the Bayesian posterior mean is ~2x better at 10 shots (0.41-0.42)
    because the prior range is information, and converges to the CRB
    by 1000 shots.
  * Few shots, V = 0.9: the pole is the wrong place and gets worse
    with more shots (plug-in 1.09, 1.38, 1.85, 2.50 at N = 10 ... 1000,
    because F -> 0 there and small phases become invisible), while the
    mid-fringe tracks its bound (1.11-1.24, CRB 1.12).

Honest summary: a clean qg form of known Ramsey facts. No gain over
standard practice (operate at mid-fringe, use a prior at few shots).
"""

import math

import numpy as np
from scipy.optimize import minimize_scalar

PHI_RANGE = (0.0, 0.5)
OPERATING = {"pole": 0.0, "mid-fringe": -math.pi / 2}


def measured_qg(phi, alpha, V=1.0, a=0.0, b=1.0):
    return a + b * V * np.cos(phi - alpha)


def fisher(phi, alpha, V=1.0, a=0.0, b=1.0):
    q = measured_qg(phi, alpha, V, a, b)
    return (b * V * np.sin(phi - alpha)) ** 2 / np.maximum(1 - q**2, 1e-300)


def fisher_from_qg(qg, V):
    """a = 0, b = 1: F as a function of the operating qg."""
    return (V**2 - qg**2) / (1 - qg**2)


def optimal_operating_point(V, a, b=1.0):
    """b < 1 - |a| keeps the fringe inside (-1, 1) (a physical readout)."""
    res = minimize_scalar(lambda x: -fisher(x, 0.0, V, a, b), bounds=(1e-6, math.pi - 1e-6), method="bounded",
                          options={"xatol": 1e-12})
    x = float(res.x)
    return x, float(measured_qg(x, 0.0, V, a, b)), float(-res.fun), float(fisher(math.pi / 2, 0.0, V, a, b))


def _plugin(k, n, alpha, V):
    q = np.clip((2 * k - n) / n / V, -1, 1)
    if alpha == 0.0:
        return np.arccos(q)                      # phi in [0, pi]
    return np.arcsin(-q)                         # alpha = -pi/2: qg = -V sin(phi)


def _bayes(k, n, alpha, V, grid=np.linspace(*PHI_RANGE, 801)):
    p0 = np.clip((1 + measured_qg(grid, alpha, V)) / 2, 1e-12, 1 - 1e-12)
    ll = k[:, None] * np.log(p0)[None] + (n - k)[:, None] * np.log(1 - p0)[None]
    ll -= ll.max(axis=1, keepdims=True)
    w = np.exp(ll)
    return (w @ grid) / w.sum(axis=1)


def few_shot(n_shots, V=1.0, reps=4000, seed=0):
    """sqrt(N) * RMSE for each (operating point, estimator), plus the
    Cramer-Rao value averaged over the prior, and the fraction of runs
    where every shot gave the same outcome."""
    rng = np.random.default_rng(seed)
    phi = rng.uniform(*PHI_RANGE, reps)
    out = {}
    for name, alpha in OPERATING.items():
        p0 = (1 + measured_qg(phi, alpha, V)) / 2
        k = rng.binomial(n_shots, p0).astype(float)
        crb = float(np.sqrt(np.mean(1 / fisher(phi, alpha, V))))
        for est in ("plug-in", "Bayes"):
            e = _plugin(k, n_shots, alpha, V) if est == "plug-in" else _bayes(k, n_shots, alpha, V)
            out[(name, est)] = float(np.sqrt(np.mean((e - phi) ** 2)) * math.sqrt(n_shots))
        out[(name, "CRB")] = crb
        out[(name, "all same")] = float(np.mean((k == 0) | (k == n_shots)))
    return out


SHOTS = (10, 30, 100, 1000)


def make_figure(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    q = np.linspace(-0.999, 0.999, 400)
    for V, c in ((1.0, "#1f6fb2"), (0.9, "#2e8b57"), (0.6, "#e0a030")):
        qq = q[np.abs(q) <= V]
        ax1.plot(qq, fisher_from_qg(qq, V), color=c, lw=2, label=f"V = {V}")
    ax1.set_xlabel("operating qg")
    ax1.set_ylabel("Fisher information per shot")
    ax1.set_title("F = (V^2 - qg^2)/(1 - qg^2)", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    for V, ls in ((1.0, "-"), (0.9, "--")):
        rows = [few_shot(n, V, reps=3000) for n in SHOTS]
        for (name, est), c in ((("pole", "plug-in"), "#c0392b"), (("pole", "Bayes"), "#f1948a"),
                               (("mid-fringe", "plug-in"), "#1f6fb2"), (("mid-fringe", "Bayes"), "#9ecae1")):
            ax2.semilogx(SHOTS, [r[(name, est)] for r in rows], ls, color=c, marker="o",
                         label=f"{name}, {est}" if V == 1.0 else None)
        ax2.semilogx(SHOTS, [r[("mid-fringe", "CRB")] for r in rows], ls, color="k", lw=1,
                     label="Cramer-Rao (mid-fringe)" if V == 1.0 else None)
    ax2.set_xlabel("shots")
    ax2.set_ylabel("sqrt(N) x RMSE(phi)")
    ax2.set_title("phi in [0, 0.5] rad; solid V = 1, dashed V = 0.9", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    import sys

    print("A. Operating point: F at quadrature vs optimum")
    # readouts: ideal; §31 (e01 0.015, e10 0.04); strongly asymmetric (e01 0.01, e10 0.10)
    for V, (e01, e10) in ((1.0, (0, 0)), (0.9, (0, 0)), (0.9, (0.015, 0.04)), (0.9, (0.01, 0.10)),
                          (0.6, (0.01, 0.10))):
        a, b = e10 - e01, 1 - e01 - e10
        x, q, f, fq = optimal_operating_point(V, a, b)
        print(f"  V {V}, e01 {e01}, e10 {e10}: optimum at x = {x:.3f} rad (qg* = {q:+.3f}), F* = {f:.4f}, "
              f"F(quadrature) = {fq:.4f}, gain {100 * (f / fq - 1):.2f}%")
    for V in (1.0, 0.9):
        print(f"\nB. Few shots, V = {V}: sqrt(N) x RMSE (CRB); fraction of runs with all shots equal")
        for n in SHOTS:
            r = few_shot(n, V)
            print(f"  N={n:5d}  " + "  ".join(
                f"{name}: plug-in {r[(name, 'plug-in')]:.2f}, Bayes {r[(name, 'Bayes')]:.2f} "
                f"(CRB {r[(name, 'CRB')]:.2f}, same {r[(name, 'all same')]:.2f})" for name in OPERATING))
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"))
