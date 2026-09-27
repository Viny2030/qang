"""
BB84 beyond intercept-resend: partial cloning, one-basis attacks and T2
drift, against the qg flags of §43.

The §43 flags know two families: T1 drift (asymmetric 1 -> 0 errors in Z)
and intercept-resend in both bases (symmetric errors in both bases). Here
the attacker and the hardware get more options:

  phase-covariant cloner  Eve keeps an optimal clone of each qubit (the
                          best individual attack on BB84); Bob's qubit
                          suffers a symmetric disturbance D in BOTH bases
  Z-only intercept        Eve measures a fraction f in Z only: errors in
                          the X basis only (f/2), none in Z
  X-only intercept        the same in X: symmetric errors in Z only
  T2 drift                the memory's dephasing grows (p0 = 0.01 -> 0.03):
                          symmetric errors in the X basis only

From Bob's side every attack enters as a Pauli channel (p_x, p_y, p_z)
applied before the memory:

  intercept-resend, both bases, fraction f   (f/4, 0, f/4)
  phase-covariant cloner with disturbance D  (0, D, 0)
  Z-only intercept, fraction f               (0, 0, f/2)
  X-only intercept, fraction f               (f/2, 0, 0)

Z-basis errors come from p_x + p_y, X-basis errors from p_z + p_y, and
Bob's BB84 statistics cannot see which Pauli produced them. So, before
running:

  * the cloner with D gives exactly the counts of intercept-resend with
    f = 4D: it is flagged as an attack exactly as often (it does leak more
    information per error to Eve, which no error statistic can show);
  * a Z-only intercept gives exactly the counts of extra dephasing
    p' = f/2: it is indistinguishable from T2 drift. This is a second blind
    spot, the twin of the T1-mimicking attack of §40;
  * the §43 flag model has no T2 family, so real T2 drift should raise its
    "attack" flag (a false alarm). Adding a dephasing family fixes the false
    alarm and turns it into an honest "T2-like" verdict, which a Z-only
    intercept shares.

Setup as §43: n = 1e5 key bits (Z), k = 7609 test bits (X), 1 % false
alarms on the baseline, 400 blocks per scenario. The extended model has
three nested flags on a (gamma, f, p) grid:

  attack-like  symmetric both-basis error beyond any T1 and T2 drift
  T1-like      gamma above gamma0, whatever f and p   (one-sided)
  T2-like      p above p0, whatever gamma and f       (one-sided)

Findings (python examples/bb84_attacks_beyond_ir_qg.py):

  Identities checked: the cloner with D gives Bob the same four error
  probabilities as intercept-resend with f = 4D, and a Z-only intercept
  with f the same as extra dephasing (1 - 2p') = (1 - 2p0)(1 - f), to 1e-16.

  Flag rates (400 blocks; Q_Z, Q_X and the asymptotic key fraction
  1 - h(Q_Z) - h(Q_X) for reference):

                                 Q_Z    Q_X    key  | attack  T1    T2   | §43: attack  T1
    baseline                     .0174  .0222  .720 | .020    .005  .033 |      .020   .015
    T1 drift 0.04                .0272  .0271  .640 | .018    1.00  .033 |      .015   1.00
    T2 drift p = 0.02            .0174  .0320  .670 | .007    .010  1.00 |      .075   .022
    T2 drift p = 0.03            .0174  .0417  .624 | .020    .013  1.00 |      .268   .030
    intercept f = 0.05           .0294  .0341  .594 | 1.00    .055  .028 |      1.00   .052
    cloner D = 0.0125            .0294  .0341  .594 | 1.00    .055  .022 |      1.00   .050
    cloner D = 0.025             .0415  .0461  .481 | 1.00    .055  .040 |      1.00   .025
    Z-only intercept f = 0.02    .0174  .0318  .671 | .013    .022  .995 |      .060   .028
    Z-only intercept f = 0.04    .0174  .0413  .625 | .013    .030  1.00 |      .265   .058
    X-only intercept f = 0.04    .0367  .0222  .620 | 1.00    .048  .000 |      1.00   .033
    T2 drift 0.02 + cloner       .0294  .0437  .550 | 1.00    .043  .995 |      1.00   .037

  * Partial cloning is caught. The optimal phase-covariant cloner is
    flagged as an attack in every block, exactly like intercept-resend
    with f = 4D, because Bob sees the same counts. What the flags cannot
    show is that the cloner gives Eve more information per error; the key
    rate (which charges every error to Eve) already accounts for that.
  * The §43 model raises false attack alarms under T2 drift: 7.5 % of
    blocks at p = 0.02 and 27 % at p = 0.03. It has no dephasing family,
    so X-only errors are best explained by an attack. Adding the family
    removes the false alarm (0.7-2 %) and gives the right verdict, T2-like,
    in every block.
  * The second blind spot: a Z-only intercept produces exactly the counts
    of T2 drift, so it is flagged T2-like (99.5-100 %) and not as an attack
    (1.3 %). It is the twin of the T1-mimicking attack of §40: an attacker
    who hides in the error type the hardware makes naturally. An X-only
    intercept has no natural twin (symmetric Z errors with no X errors)
    and is caught in every block.
  * Mixtures are resolved: T2 drift plus a cloner raises both flags in
    97-100 % of blocks.
  * False alarms on the baseline stay at 0.5-3.3 % (thresholds set at 1 %
    on 400 bootstrap blocks); cross-flags under attack at 2-6 %.

Honest scope. Individual attacks only, as Pauli channels on Bob's side;
coherent (collective) attacks and detector attacks are not modelled.
Diagnosis, not security: the key rate is unchanged, and the Z-only
intercept shows that an attacker who mimics a natural error type is not
flagged as an attack by any error statistic.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bb84_finite_key_qg as F  # noqa: E402
import bb84_qg_eve_vs_noise as B  # noqa: E402

N_KEY, K_TEST = 100000, 7609
BASE = dict(B.BASE)  # gamma 0.02, p 0.01, e01 0.005, e10 0.01


def error_probabilities(gamma=BASE["gamma"], p=BASE["p"], e01=BASE["e01"], e10=BASE["e10"],
                        px=0.0, py=0.0, pz=0.0):
    """P(Bob's bit != Alice's) for (0, 1, +, -): Pauli attack (px, py, pz),
    then amplitude damping gamma, dephasing p, readout (e01 = P(1|0), e10 = P(0|1))."""
    flip_z = px + py  # flips Z-basis states
    flip_x = pz + py  # flips X-basis states
    # Z basis: population of |1> after the attack and damping
    p1_from0 = flip_z * (1 - gamma)
    p1_from1 = (1 - flip_z) * (1 - gamma)
    err0 = p1_from0 * (1 - e10) + (1 - p1_from0) * e01
    err1 = (1 - p1_from1) * (1 - e01) + p1_from1 * e10
    # X basis: Bloch x of |+> after attack, damping, dephasing; z after damping
    x = (1 - 2 * flip_x) * math.sqrt(1 - gamma) * (1 - 2 * p)
    # measured after H: outcome 0 <-> +x ; readout acts on that outcome
    p0_plus = (1 + x) / 2
    err_plus = 1 - (p0_plus * (1 - e01) + (1 - p0_plus) * e10)
    p0_minus = (1 - x) / 2
    err_minus = p0_minus * (1 - e01) + (1 - p0_minus) * e10
    return np.array([err0, err1, err_plus, err_minus])


SCENARIOS = [
    ("baseline", {}),
    ("T1 drift 0.04", {"gamma": 0.04}),
    ("T2 drift p = 0.02", {"p": 0.02}),
    ("T2 drift p = 0.03", {"p": 0.03}),
    ("intercept f = 0.05 (both bases)", {"px": 0.0125, "pz": 0.0125}),
    ("cloner D = 0.0125", {"py": 0.0125}),
    ("cloner D = 0.025", {"py": 0.025}),
    ("Z-only intercept f = 0.02", {"pz": 0.01}),
    ("Z-only intercept f = 0.04", {"pz": 0.02}),
    ("X-only intercept f = 0.04", {"px": 0.02}),
    ("T2 drift 0.02 + cloner 0.0125", {"p": 0.02, "py": 0.0125}),
]

G3 = np.round(np.arange(0.0, 0.1201, 0.0025), 4)
F3 = np.round(np.arange(0.0, 0.2001, 0.0025), 4)
P3 = np.round(np.arange(0.0, 0.0801, 0.0025), 4)
_TABLE = None


def table3():
    global _TABLE
    if _TABLE is None:
        _TABLE = np.array([[[error_probabilities(gamma=g, p=p, px=f / 4, pz=f / 4) for p in P3] for f in F3]
                           for g in G3])  # (gamma, f, p, 4)
    return _TABLE


_I_G0 = int(np.argmin(np.abs(G3 - BASE["gamma"])))
_I_P0 = int(np.argmin(np.abs(P3 - BASE["p"])))


def statistics3(kc, nc):
    pr = np.clip(table3(), 1e-12, 1 - 1e-12)
    ll = (kc * np.log(pr) + (nc - kc) * np.log(1 - pr)).sum(axis=-1)
    best = ll.max()
    # one-sided nested tests: f > 0, gamma > gamma0, p > p0
    return (float(best - ll[:, 0, :].max()),  # attack-like
            float(best - ll[: _I_G0 + 1, :, :].max()),  # T1-like
            float(best - ll[:, :, : _I_P0 + 1].max()))  # T2-like


def calibrate3(n=N_KEY, k=K_TEST, reps=400, seed=0):
    rng = np.random.default_rng(seed)
    base = error_probabilities()
    s = np.array([statistics3(*F.block(base, n, k, rng)) for _ in range(reps)])
    return tuple(float(np.quantile(s[:, j], 0.99)) for j in range(3))


def diagnose(n=N_KEY, k=K_TEST, reps=400, seed=1):
    th3 = calibrate3(n, k)
    th2 = F.calibrate(n, k, reps)
    rng = np.random.default_rng(seed)
    out = {}
    for name, kw in SCENARIOS:
        probs = error_probabilities(**kw)
        hits = np.zeros(5)
        for _ in range(reps):
            kc, nc = F.block(probs, n, k, rng)
            a, t1, t2 = statistics3(kc, nc)
            a2, d2 = F.statistics(kc, nc)
            hits += [a > th3[0], t1 > th3[1], t2 > th3[2], a2 > th2[0], d2 > th2[1]]
        qz, qx = probs[:2].mean(), probs[2:].mean()
        out[name] = {"attack": hits[0] / reps, "T1": hits[1] / reps, "T2": hits[2] / reps,
                     "attack_s43": hits[3] / reps, "drift_s43": hits[4] / reps,
                     "q_z": float(qz), "q_x": float(qx), "key_fraction": B.shor_preskill(qz, qx)}
    return out


def make_figure(path, d):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n, _ in SCENARIOS]
    short = ["base", "T1\n0.04", "T2\n0.02", "T2\n0.03", "IR\nf=.05", "clone\n.0125", "clone\n.025",
             "Z-only\n.02", "Z-only\n.04", "X-only\n.04", "T2 +\nclone"]
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.bar(x - 0.2, [d[n]["attack_s43"] for n in names], 0.4, color="#8c2d04", label="§43 model: attack-like")
    ax1.bar(x + 0.2, [d[n]["drift_s43"] for n in names], 0.4, color="#e0a030", label="§43 model: T1-like")
    ax1.set_ylabel("fraction of blocks")
    ax1.set_title("Two-family model (§43): T2 drift raises a false attack alarm", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", alpha=0.3)
    for off, key, col, lab in ((-0.27, "attack", "#8c2d04", "attack-like"), (0.0, "T1", "#e0a030", "T1-like"),
                               (0.27, "T2", "#1f6fb2", "T2-like")):
        ax2.bar(x + off, [d[n][key] for n in names], 0.26, color=col, label=lab)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, fontsize=7)
    ax2.set_ylabel("fraction of blocks")
    ax2.set_title("Three-family model: T2 drift and a Z-only intercept get the same verdict", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    d = diagnose()
    print(f"n = {N_KEY}, k = {K_TEST}; flag rates over 400 blocks")
    print(f"{'scenario':32s} {'Q_Z':>6} {'Q_X':>6} {'key':>5} | {'attack':>6} {'T1':>5} {'T2':>5} | "
          f"{'§43 att':>7} {'§43 T1':>6}")
    for name, _ in SCENARIOS:
        v = d[name]
        print(f"{name:32s} {v['q_z']:6.4f} {v['q_x']:6.4f} {v['key_fraction']:5.3f} | {v['attack']:6.3f} "
              f"{v['T1']:5.3f} {v['T2']:5.3f} | {v['attack_s43']:7.3f} {v['drift_s43']:6.3f}")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"), d)
