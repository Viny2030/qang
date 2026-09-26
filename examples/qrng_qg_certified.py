"""
Certified quantum random numbers in qg units: how many of the output bits
are really random?

A qubit QRNG prepares |+> and measures Z. Ideally every bit is uniform
and unpredictable. On a real device three things go wrong:
  * dephasing (T2) during the wait shrinks the coherence;
  * thermal population (§31) makes the start state mixed;
  * the readout flips 0 <-> 1 asymmetrically (e01, e10; §31).
Dephasing and thermal mixing leave the OUTPUT looking unbiased, but the
randomness is then classical noise, which an adversary holding the
environment (the purification) can know.

The certified quantity (trusted measurement, adversary with quantum side
information E) has a one-line qg form. Measuring Z on a qubit with Bloch
vector (qg_X, qg_Y, qg_Z), the adversary's two conditional states are
pure with overlap |rho_01| = r_perp / 2, so by Helstrom

    P_guess(Z | E) = (1 + sqrt(1 - r_perp^2)) / 2,
    H_min(Z | E)   = -log2 P_guess,      r_perp = sqrt(qg_X^2 + qg_Y^2),

independent of qg_Z: only the coherence in the measured basis is private.
(For a pure state r_perp = sqrt(1 - qg_Z^2) and this reduces to the
classical -log2 max(p0, p1).) Readout flips are classical noise that the
adversary may know, so they add no certified randomness.

Estimators of the certified bits per shot (from N test rounds):
  naive          -log2 max(p0_hat, p1_hat) of the output bits (the usual
                 bias-based health estimate; ignores side information)
  qg, one-sided  r_perp from X and Y test rounds with one rotation each
                 (measured qg = a + b qg: the readout offset a can inflate
                 r_perp -- unsafe)
  qg, +/- pairs  each axis measured with both rotations, so the offset
                 cancels: (m_+ - m_-)/2 = b qg (safe, reduced by b)
  qg, calibrated the +/- estimate divided by b from the §31 heralded
                 calibration (safe and tight)
All qg estimates use a 3-sigma lower confidence bound on r_perp.

Findings (10^4 test rounds per setting, 300 repetitions; readout of
§31, e01 = 0.015, e10 = 0.04; certified bits per shot, mean estimate and
fraction of runs ABOVE the truth, i.e. unsafe):

  scenario                  truth  naive        qg +/- pairs  qg calibrated
  ideal |+>                 1.000  0.965 (0)    0.512 (0)     0.681 (0)
  T2: V = 0.8               0.322  0.964 (1.00) 0.245 (0)     0.286 (0)
  T2: V = 0.5               0.100  0.965 (1.00) 0.077 (0)     0.087 (0)
  T2: V = 0.2               0.015  0.965 (1.00) 0.009 (0)     0.010 (0)
  thermal p = 0.05, V = 0.9 0.334  0.965 (1.00) 0.254 (0)     0.297 (0)
  e10 = 0.10, V = 0.95      0.608  0.876 (1.00) 0.341 (0)     0.513 (0)

  * The usual output-bias estimate is unsafe whenever the state is not
    pure: with V = 0.2 it certifies 0.97 bits per shot where only 0.015
    are private. Dephasing and thermal mixing leave the output unbiased.
  * The qg estimate from the X and Y test rounds is safe in every run
    when each axis is measured with both rotations (the readout offset
    a cancels). With a single rotation the offset inflates r_perp and
    the estimate exceeds the truth in 7% of runs at V = 0.2.
  * The §31 calibration of b recovers 10-50% more certified bits
    (0.286 vs 0.245 at V = 0.8; 0.513 vs 0.341 with strong asymmetry).
  * Near a perfect source the certification is shot-hungry: H_min has an
    infinite slope at r_perp = 1 (the qg pole again), so a 3-sigma bound
    gives 0.68, 0.81, 0.89 bits per shot with 10^4, 10^5, 10^6 test
    rounds (truth 1).

Honest scope: device-dependent QRNG (trusted measurement, i.i.d. rounds,
qubit model), not device-independent randomness; the min-entropy formula
is standard (Helstrom discrimination of the adversary's two conditional
states); qg makes it one line in measured quantities.
"""

import math

import numpy as np

READOUT = {"e01": 0.015, "e10": 0.04}


def h_min_certified(r_perp):
    r = min(max(float(r_perp), 0.0), 1.0)
    return -math.log2((1 + math.sqrt(1 - r * r)) / 2)


def h_min_naive(p0):
    return -math.log2(max(p0, 1 - p0))


def device_state(visibility=1.0, p_thermal=0.0, phase=0.0):
    """Bloch vector after Ry(pi/2) on the thermal state and dephasing:
    the thermal qg_eq = 1 - 2p is rotated onto the equator and shrunk by
    the visibility V; phase is a small rotation error about Z."""
    q = 1 - 2 * p_thermal
    r = q * visibility
    return np.array([r * math.cos(phase), r * math.sin(phase), 0.0])


def readout_map(e01=READOUT["e01"], e10=READOUT["e10"]):
    return e10 - e01, 1 - e01 - e10  # measured qg = a + b qg


def output_p0(bloch, **ro):
    a, b = readout_map(**ro)
    return (1 + a + b * bloch[2]) / 2


def _measure(q_true, n, rng, **ro):
    a, b = readout_map(**ro)
    return 2 * rng.binomial(n, (1 + a + b * q_true) / 2) / n - 1


def estimates(bloch, n_test, rng, b_calibrated=None, sigmas=3.0, **ro):
    """Certified-bit estimates from n_test test rounds per setting."""
    a, b = readout_map(**ro)
    n_out = 4 * n_test
    p0_hat = rng.binomial(n_out, output_p0(bloch, **ro)) / n_out
    x1, y1 = _measure(bloch[0], n_test, rng, **ro), _measure(bloch[1], n_test, rng, **ro)
    xp, xm = _measure(bloch[0], n_test // 2, rng, **ro), _measure(-bloch[0], n_test // 2, rng, **ro)
    yp, ym = _measure(bloch[1], n_test // 2, rng, **ro), _measure(-bloch[1], n_test // 2, rng, **ro)
    x_pm, y_pm = (xp - xm) / 2, (yp - ym) / 2
    se = 1 / math.sqrt(n_test)

    def lcb(x, y, scale=1.0):
        r = math.hypot(x, y) / scale
        return max(0.0, r - sigmas * se / scale)

    bc = b if b_calibrated is None else b_calibrated
    return {
        "naive": h_min_naive(p0_hat),
        "qg one-sided": h_min_certified(lcb(x1, y1)),
        "qg +/- pairs": h_min_certified(lcb(x_pm, y_pm)),
        "qg calibrated": h_min_certified(lcb(x_pm, y_pm, bc)),
    }


def truth(bloch):
    return h_min_certified(math.hypot(bloch[0], bloch[1]))


SCENARIOS = [
    ("ideal |+>, ideal readout", dict(visibility=1.0), dict(e01=0.0, e10=0.0)),
    ("ideal |+>, §31 readout", dict(visibility=1.0), {}),
    ("T2: V = 0.8", dict(visibility=0.8), {}),
    ("T2: V = 0.5", dict(visibility=0.5), {}),
    ("T2: V = 0.2", dict(visibility=0.2), {}),
    ("thermal p = 0.05, V = 0.9", dict(visibility=0.9, p_thermal=0.05), {}),
    ("strong readout asymmetry", dict(visibility=0.95), dict(e01=0.01, e10=0.10)),
]
ESTIMATORS = ("naive", "qg one-sided", "qg +/- pairs", "qg calibrated")


def table(n_test=10000, reps=300, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name, dev, ro in SCENARIOS:
        bloch = device_state(**dev)
        rows = [estimates(bloch, n_test, rng, **ro) for _ in range(reps)]
        t = truth(bloch)
        out[name] = {"truth": t}
        for e in ESTIMATORS:
            v = np.array([r[e] for r in rows])
            out[name][e] = (float(v.mean()), float(np.mean(v > t + 1e-12)))  # mean, fraction unsafe
    return out


def make_figure(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    vs = np.linspace(0.05, 1.0, 20)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    rp = np.linspace(0, 1, 200)
    ax1.plot(rp, [h_min_certified(r) for r in rp], color="#1f6fb2", lw=2, label="certified, H_min(Z|E)")
    ax1.plot(rp, [1.0 for _ in rp], color="#8c8c8c", lw=2, ls="--", label="naive (output looks unbiased)")
    ax1.set_xlabel("coherence r_perp = sqrt(qg_X^2 + qg_Y^2)")
    ax1.set_ylabel("random bits per shot")
    ax1.set_title("A qubit measured in Z, qg_Z = 0", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    style = {"naive": "#8c8c8c", "qg one-sided": "#e0a030", "qg +/- pairs": "#9ecae1", "qg calibrated": "#1f6fb2"}
    res = {e: [] for e in ESTIMATORS}
    for v in vs:
        rows = [estimates(device_state(v), 10000, rng) for _ in range(40)]
        for e in ESTIMATORS:
            res[e].append(np.mean([r[e] for r in rows]))
    ax2.plot(vs, [truth(device_state(v)) for v in vs], "k-", lw=2, label="truth")
    for e in ESTIMATORS:
        ax2.plot(vs, res[e], "o", color=style[e], ms=4, label=e)
    ax2.set_xlabel("visibility V (T2 dephasing)")
    ax2.set_title("Estimated certified bits per shot (§31 readout, 10^4 test rounds)", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    import sys

    t = table()
    print("certified random bits per shot: mean estimate (fraction of runs above the truth = unsafe)")
    print(f"{'scenario':30s} {'truth':>6} | " + " | ".join(f"{e:>20s}" for e in ESTIMATORS))
    for name, _, _ in SCENARIOS:
        r = t[name]
        print(f"{name:30s} {r['truth']:6.3f} | " + " | ".join(f"{r[e][0]:12.3f} ({r[e][1]:.2f})" for e in ESTIMATORS))
    print("\nNear-perfect source (V = 1, §31 readout): calibrated estimate vs test rounds")
    rng = np.random.default_rng(2)
    for n in (10**4, 10**5, 10**6):
        v = np.mean([estimates(device_state(1.0), n, rng)["qg calibrated"] for _ in range(100)])
        print(f"  {n:8d} test rounds per setting: {v:.3f} certified bits per shot (truth 1.000)")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"))
