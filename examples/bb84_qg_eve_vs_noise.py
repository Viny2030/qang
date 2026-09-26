"""
BB84 on a noisy qubit channel: eavesdropper or natural noise? A qg
monitor vs the standard QBER monitor.

Setting. Alice sends |0>, |1> (Z basis) or |+>, |-> (X basis); the qubit
waits in a noisy memory or channel (amplitude damping gamma = T1, then
dephasing p) and Bob measures with an asymmetric detector (e01, e10, as
calibrated in §31). After sifting, Alice and Bob compare a sample of the
key. The standard monitor looks only at the total error rate (QBER) and
raises an alarm when it exceeds its calibrated baseline.

In qg units each sent state gives one number:
    qg(0) = 1 - 2 e(0->1),   qg(1) = -1 + 2 e(1->0)   (Z basis)
and the same for |+>, |-> in the X basis. Their sum
    A_Z = qg(0) + qg(1) = 2 [e(1->0) - e(0->1)]
is the T1 witness of §24: natural relaxation pushes both towards +1
(only 1 -> 0 errors), while a naive intercept-resend attack adds
SYMMETRIC errors (0 -> 1 as often as 1 -> 0, and in both bases).

Scenarios, all at the same baseline channel (gamma0 = 0.02, p0 = 0.01,
detector e01 = 0.005, e10 = 0.01):
  baseline        nothing changes
  T1 drift        the memory's T1 gets worse (gamma = 0.04, 0.06)
  intercept-resend  Eve measures a fraction f of the qubits in a random
                  basis and resends (f = 0.02 ... 0.2)
  T1-mimicking attack  Eve couples each qubit to her ancilla through an
                  amplitude-damping interaction (extra gamma_E); from
                  Bob's side this is identical to T1 drift

Monitors (one test window of N sifted and compared bits, 1% false alarms
on the baseline):
  QBER            one-sided binomial test on the total error count
  qg (GLRT)       the four error counts (0->1, 1->0, +->-, -->+) and a
                  generalized likelihood ratio: "attack" (f > 0) must
                  beat both "baseline" and "T1 drift" (gamma free)

Security note, stated up front: a QKD security proof must attribute ALL
errors to Eve; the secret-key fraction (Shor-Preskill,
1 - h(Q_Z) - h(Q_X)) does not improve because errors look like T1, and a
clever Eve can mimic T1 exactly (the last scenario). The qg monitor is an
operational diagnostic (alarm, hardware drift vs tampering), not a
security proof.

Findings (alarm rate of the QBER monitor / the qg monitor, 1000 windows,
1% false alarms on the baseline):

                               N = 500        N = 2000       N = 10000
  baseline                   0.007/0.004    0.007/0.006    0.010/0.007
  T1 drift, gamma 0.04       0.147/0.009    0.478/0.009    0.993/0.000
  T1 drift, gamma 0.06       0.463/0.009    0.962/0.006    1.000/0.000
  intercept-resend f = 0.02  0.072/0.065    0.218/0.271    0.847/0.848
  intercept-resend f = 0.05  0.337/0.338    0.861/0.823    1.000/1.000
  intercept-resend f = 0.10  0.821/0.730    1.000/0.994    1.000/1.000
  T1-mimicking attack        0.463/0.009    0.962/0.006    1.000/0.000

  * The qg monitor detects the naive intercept-resend attack as well as
    the QBER monitor (within the sampling spread at every N and f), but
    it does NOT raise alarms when the memory's T1 gets worse (false
    alarms stay at the 1% level where the QBER monitor fires in 48-100%
    of windows at N = 2000). The reason is the qg asymmetry
    A_Z = qg(0) + qg(1): it grows with T1 (0.049 -> 0.128) and stays
    fixed under intercept-resend, whose errors are symmetric.
  * The price, by design: a T1-mimicking attack is invisible to the qg
    monitor (0.6% alarms) and caught by the QBER monitor (96%). The two
    monitors are complementary: QBER says "something changed", qg says
    "symmetric (attack-like) or T1-like (hardware drift, or an attacker
    who hides as one)".
  * The secret fraction (Shor-Preskill) falls the same way in both cases
    (0.72 -> 0.57 for T1 drift to 0.06, 0.72 -> 0.59 for f = 0.05):
    security must still attribute every error to Eve.

Honest summary: an operational diagnostic that tells hardware drift
from naive tampering, which the QBER monitor cannot, not a security gain. The
setting (a qubit stored with T1 and T2 before Bob measures) fits
matter-qubit links and on-chip BB84 demonstrations, not photon
polarization links, whose noise is mostly unital.
"""

import math

import numpy as np

BASE = {"gamma": 0.02, "p": 0.01, "e01": 0.005, "e10": 0.01}
ALPHA = 0.01
F_GRID = np.linspace(0.0, 0.6, 301)
G_GRID = np.linspace(0.0, 0.4, 401)

I2 = np.eye(2)
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.diag([1.0, -1.0])
H = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2)
KETS = {"0": np.array([1.0, 0.0]), "1": np.array([0.0, 1.0]),
        "+": np.array([1.0, 1.0]) / math.sqrt(2), "-": np.array([1.0, -1.0]) / math.sqrt(2)}
CATEGORIES = (("Z", "0"), ("Z", "1"), ("X", "+"), ("X", "-"))


def _dephase_in(basis, rho):
    """Measure-and-prepare in the given basis (what intercept-resend does)."""
    if basis == "Z":
        return np.diag(np.diag(rho))
    r = H @ rho @ H
    return H @ np.diag(np.diag(r)) @ H


def _channel(rho, gamma, p):
    a0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1 - gamma)]])
    a1 = np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]])
    rho = a0 @ rho @ a0.T + a1 @ rho @ a1.T
    return (1 - p) * rho + p * Z @ rho @ Z


def error_probabilities(gamma=BASE["gamma"], p=BASE["p"], e01=BASE["e01"], e10=BASE["e10"], f=0.0):
    """P(Bob's bit != Alice's bit) for each (basis, sent state)."""
    out = []
    for basis, s in CATEGORIES:
        rho = np.outer(KETS[s], KETS[s])
        rho = (1 - f) * rho + f * 0.5 * (_dephase_in("Z", rho) + _dephase_in("X", rho))
        rho = _channel(rho, gamma, p)
        if basis == "X":
            rho = H @ rho @ H
        p0 = float(np.real(rho[0, 0]))
        p_read0 = p0 * (1 - e01) + (1 - p0) * e10
        wrong = 1 - p_read0 if s in ("0", "+") else p_read0
        out.append(wrong)
    return np.array(out)


def qg_per_state(errors):
    """qg of Bob's outcome for each sent state, read in its own basis."""
    e = np.asarray(errors)
    return np.array([1 - 2 * e[0], -1 + 2 * e[1], 1 - 2 * e[2], -1 + 2 * e[3]])


def z_asymmetry(errors):
    q = qg_per_state(errors)
    return float(q[0] + q[1])


def qber(errors):
    e = np.asarray(errors)
    return float(e[:2].mean()), float(e[2:].mean())


def shor_preskill(qz, qx):
    h = lambda x: 0.0 if x <= 0 or x >= 1 else -x * math.log2(x) - (1 - x) * math.log2(1 - x)  # noqa: E731
    return max(0.0, 1 - h(qz) - h(qx))


# --------------------------------------------------------------------- #
# monitors
# --------------------------------------------------------------------- #
def _loglik(k, n, probs):
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    return float((k * np.log(probs) + (n - k) * np.log(1 - probs)).sum())


_ATTACK_TABLE = np.array([error_probabilities(f=f) for f in F_GRID])
_DRIFT_TABLE = np.array([error_probabilities(gamma=g) for g in G_GRID])
_BASE_PROBS = error_probabilities()


def glrt_statistic(k, n):
    """log L(attack, best f) - max(log L(baseline), log L(drift, best gamma))."""
    ll_attack = max(_loglik(k, n, pr) for pr in _ATTACK_TABLE)
    ll_drift = max(_loglik(k, n, pr) for pr in _DRIFT_TABLE)
    return ll_attack - max(ll_drift, _loglik(k, n, _BASE_PROBS))


def sample_counts(probs, n_bits, rng):
    """n_bits sifted and compared bits, split evenly over the 4 categories."""
    n = np.full(4, n_bits // 4)
    return rng.binomial(n, probs), n


def calibrate_thresholds(n_bits, reps=2000, seed=0):
    """1% false-alarm thresholds on the baseline for both monitors."""
    rng = np.random.default_rng(seed)
    tot, stat = [], []
    for _ in range(reps):
        k, n = sample_counts(_BASE_PROBS, n_bits, rng)
        tot.append(k.sum())
        stat.append(glrt_statistic(k, n))
    return float(np.quantile(tot, 1 - ALPHA)), float(np.quantile(stat, 1 - ALPHA))


def alarm_rates(probs, n_bits, thresholds, reps=1000, seed=1):
    rng = np.random.default_rng(seed)
    t_tot, t_glrt = thresholds
    a_q = a_g = 0
    for _ in range(reps):
        k, n = sample_counts(probs, n_bits, rng)
        a_q += k.sum() > t_tot
        a_g += glrt_statistic(k, n) > t_glrt
    return a_q / reps, a_g / reps


SCENARIOS = [("baseline", {}), ("T1 drift, gamma 0.04", {"gamma": 0.04}), ("T1 drift, gamma 0.06", {"gamma": 0.06}),
             ("intercept-resend f=0.02", {"f": 0.02}), ("intercept-resend f=0.05", {"f": 0.05}),
             ("intercept-resend f=0.10", {"f": 0.10}), ("intercept-resend f=0.20", {"f": 0.20}),
             ("T1-mimicking attack, +0.04", {"gamma": 0.06})]


def scenario_table(n_bits_list=(500, 2000, 10000), reps=1000):
    out = {}
    for n_bits in n_bits_list:
        th = calibrate_thresholds(n_bits)
        for name, kw in SCENARIOS:
            out[(name, n_bits)] = alarm_rates(error_probabilities(**kw), n_bits, th, reps)
    return out


def make_figure(table, path, n_bits=2000):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fs = [0.02, 0.05, 0.10, 0.20]
    for nb, ls in ((500, ":"), (2000, "--"), (10000, "-")):
        th = calibrate_thresholds(nb)
        rows = [alarm_rates(error_probabilities(f=f), nb, th, reps=600) for f in fs]
        ax1.plot(fs, [r[0] for r in rows], ls, color="#8c8c8c", marker="o", label=f"QBER monitor, N={nb}")
        ax1.plot(fs, [r[1] for r in rows], ls, color="#1f6fb2", marker="s", label=f"qg monitor, N={nb}")
    ax1.set_xlabel("fraction of qubits intercepted (f)")
    ax1.set_ylabel("detection rate")
    ax1.set_title("Naive intercept-resend attack", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    names = [n for n, _ in SCENARIOS]
    short = ["baseline", "T1 drift\n0.04", "T1 drift\n0.06", "IR f=.02", "IR f=.05", "IR f=.10", "IR f=.20",
             "mimic\nattack"]
    x = np.arange(len(names))
    ax2.bar(x - 0.2, [table[(n, n_bits)][0] for n in names], 0.38, color="#8c8c8c", label="QBER monitor")
    ax2.bar(x + 0.2, [table[(n, n_bits)][1] for n in names], 0.38, color="#1f6fb2", label="qg monitor")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, fontsize=7)
    ax2.set_ylabel("alarm rate")
    ax2.set_title(f"Alarms per scenario, N = {n_bits} compared bits", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    import sys

    print("Error probabilities (0->1, 1->0, +->-, -->+), QBER_Z, QBER_X, A_Z, secret fraction:")
    for name, kw in SCENARIOS:
        e = error_probabilities(**kw)
        qz, qx = qber(e)
        print(f"  {name:28s} {np.round(e, 4)}  Q_Z {qz:.4f} Q_X {qx:.4f}  A_Z {z_asymmetry(e):+.4f}  "
              f"r {shor_preskill(qz, qx):.3f}")
    table = scenario_table()
    print("\nAlarm rates (QBER monitor / qg monitor), 1% false alarms on the baseline:")
    for n_bits in (500, 2000, 10000):
        print(f"  N = {n_bits}")
        for name, _ in SCENARIOS:
            a, b = table[(name, n_bits)]
            print(f"    {name:28s} {a:.3f} / {b:.3f}")
    if "--figure" in sys.argv:
        make_figure(table, __file__.replace(".py", ".png"))
