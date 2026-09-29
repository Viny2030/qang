"""
Removing the arccos range limit of qg-space optimization with a branch sign.

The limit (paper, Sec. 3; RESEARCH_NOTES §8.2). A qg-space update is mapped
back through theta = arccos(qg_Z), which only covers [0, pi]. On H2 (0.735 A,
STO-3G, 2 qubits) the optimum lies at theta ~ -0.22, and raw, clipped and
Tikhonov qg-space descent all stop at the Hartree-Fock energy.

The proposal. qg_Z = cos(theta) is two-to-one; the pair (qg_Z, s) with
s = sign(qg_X) = sign(sin theta) is one-to-one on the whole circle, and s is
measurable (the X-basis polar bias of the same rotation). qang.gradients now
has two signed updates:

  signed_qg_step          clipped inverse-Jacobian step at the signed angle,
                          reflected through a pole with a sign flip when it
                          leaves [-1, 1]
  signed_natural_qg_step  the quantum-natural-gradient step in qg,
                          dq = lr * dE/dtheta * sin(theta), with the same
                          reflection (to first order a plain theta step)

Prediction, written before the run:
  P1 both signed updates reach the FCI energy of H2 from Hartree-Fock, where
     every unsigned qg update is trapped;
  P2 the signed natural step behaves like theta-space descent (same steps to
     chemical accuracy within +-2 on H2, same success rates on random
     landscapes within 5 points);
  P3 the signed clipped step inherits the 1/sin^2(theta) preconditioning of
     Euclidean qg coordinates: it is fast at small learning rates and
     unstable at the learning rates where theta-space descent works.
So the range limit is a representation artefact, not a property of qg; what
remains is a choice of metric, and the natural metric in qg IS theta-space.

Findings (python examples/signed_qg_range_qg.py):

  A. H2 from Hartree-Fock (error vs FCI; first step after which the error
     stays below chemical accuracy):
       lr     theta        qg_clipped    signed_qg      signed_natural_qg   pole-damped
       0.03   8e-9 (52)    2.0e-2 never  9e-10 (8)      9e-9 (54)           9.4e-3 never
       0.1    9e-10 (15)   2.0e-2 never  9e-10 (8)      9e-10 (17)          4e-5 (183)
       0.3    9e-10 (5)    2.0e-2 never  0.11 never     9e-10 (6)           9e-10 (61)
       1.0    9e-10 (1)    2.0e-2 never  0.083 never    9e-10 (1)           9e-10 (19)
  B. 300 random near-pole landscapes, success rate:
       lr     theta  qg_clipped  signed_qg  signed_natural  pole-damped
       0.1    1.00   0.20        0.84       1.00            0.91
       0.5    1.00   0.02        0.49       1.00            1.00
       1.0    0.79   0.01        0.27       0.68            1.00
       2.0    0.18   0.01        0.09       0.16            0.51
       3.0    0.08   0.01        0.04       0.08            0.37
  C. LiH (4 parameters, all optima within 0.04 rad of a pole): signed
     natural converges at lr = 0.3 in 21 steps (theta 17); at lr = 1.0
     theta converges in 5 steps and signed natural does not; the unsigned
     and signed clipped steps never converge; pole-damped converges at
     every lr (355, 106, 53 steps).

  * P1 holds. The branch sign removes the trap: signed qg-space descent
    reaches the FCI energy of H2 from Hartree-Fock (error 1e-9 Ha), where
    every unsigned qg update stops at Hartree-Fock (2.0e-2 Ha). The arccos
    range limit of the paper is a representation artefact of using qg_Z
    alone, not a property of qg: (qg_Z, sign qg_X) covers the circle.
  * P3 holds. The signed clipped (Euclidean) step is 2-7x faster than theta
    at small lr on H2 (8 steps vs 15-52) and unstable from lr = 0.3; on
    the landscapes it succeeds 84 % at lr = 0.1 and falls fast.
  * P2 holds at small and moderate learning rates (H2 within 2 steps at
    every lr; landscapes equal at lr <= 0.5 and within 2 points at
    lr >= 2) and FAILS at lr = 1.0: 0.68 vs 0.79 on the landscapes, and
    no convergence on LiH where theta converges. The reflection at a pole
    is exact only to first order in the step.
  * Verdict: the sign fixes the documented limit but gives qg-space
    optimization no advantage. The natural metric in qg reduces to theta
    space, and at aggressive learning rates pole-damped theta descent
    remains the robust choice (1.00 and 0.51 where theta has 0.79 and
    0.18). The paper's limitation (ii) should read "removed by a branch
    sign, with no gain over theta-space".
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from qang.gradients import (  # noqa: E402
    pole_damping_factor,
    signed_natural_qg_step,
    signed_qg_step,
    signed_theta,
    toy_vqe_energy,
    toy_vqe_grad_theta,
)

SPACES = ("theta", "qg_clipped", "signed_qg", "signed_natural_qg", "theta_pole_damped")


def descend(energy, grad, theta0, space, lr, steps, eps=0.05):
    """Generic multi-parameter descent. energy/grad take a list of angles."""
    th = list(theta0)
    q = [math.cos(t) for t in th]
    s = [1 if math.sin(t) >= 0 else -1 for t in th]
    es = [energy(th)]
    for _ in range(steps):
        g = grad(th)
        if space == "theta":
            th = [t - lr * gi for t, gi in zip(th, g)]
        elif space == "theta_pole_damped":
            th = [t - lr * pole_damping_factor(t, eps) * gi for t, gi in zip(th, g)]
        elif space == "qg_clipped":  # unsigned: the trapped baseline
            new = []
            for t, gi in zip(th, g):
                sn = math.sin(t)
                sin_c = math.copysign(max(abs(sn), eps), sn if sn != 0 else 1.0)
                qn = max(-1.0, min(1.0, math.cos(t) - lr * gi * (-1.0 / sin_c)))
                new.append(math.acos(qn))
            th = new
        else:
            for i, gi in enumerate(g):
                if space == "signed_qg":
                    q[i], s[i] = signed_qg_step(q[i], s[i], gi, lr, eps)
                else:
                    q[i], s[i] = signed_natural_qg_step(q[i], s[i], gi, lr)
            th = [signed_theta(qi, si) for qi, si in zip(q, s)]
        e = energy(th)
        es.append(e)
        if not math.isfinite(e) or any(abs(t) > 1e6 for t in th):
            break
    return es


def first_within(es, target, tol):
    for i, e in enumerate(es):
        if abs(e - target) < tol:
            return i
    return None


def sustained_within(es, target, tol):
    ok = [abs(e - target) < tol for e in es]
    i = len(es)
    for k in range(len(es) - 1, -1, -1):
        if ok[k]:
            i = k
        else:
            break
    return i if i < len(es) else None


# --------------------------------------------------------------------- #
def h2_study(lrs=(0.03, 0.1, 0.3, 1.0), steps=300):
    import vqe_h2_qg_vs_theta as V

    E = lambda th: V.h2_energy(th[0])  # noqa: E731
    G = lambda th: [V.h2_energy_grad(th[0])]  # noqa: E731
    out = {}
    for lr in lrs:
        for sp in SPACES:
            es = descend(E, G, [0.0], sp, lr, steps)
            out[(lr, sp)] = (es[-1] - V.EXACT_FCI_ENERGY, sustained_within(es, V.EXACT_FCI_ENERGY, V.CHEMICAL_ACCURACY))
    return out


def landscape_study(lrs=(0.1, 0.5, 1.0, 2.0, 3.0), n=300, seed=7, iters=200, tol=1e-3):
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n):
        hz, hx = rng.uniform(-2, 2, size=2)
        pole = rng.choice([0.0, math.pi])
        off = rng.uniform(1e-4, 0.05) * rng.choice([-1, 1])
        trials.append((hz, hx, -math.hypot(hz, hx), pole + off))
    out = {}
    for lr in lrs:
        for sp in SPACES:
            ok = 0
            for hz, hx, tmin, t0 in trials:
                es = descend(lambda th: toy_vqe_energy(th[0], hz, hx), lambda th: [toy_vqe_grad_theta(th[0], hz, hx)],
                             [t0], sp, lr, iters)
                ok += abs(es[-1] - tmin) < tol
            out[(lr, sp)] = ok / n
    return out


def lih_study(lrs=(0.3, 1.0, 2.0), steps=400):
    import lih_vqe_ry_rx_ansatz as L

    out = {}
    t0 = [math.pi, math.pi, 1e-6, 1e-6]
    for lr in lrs:
        for sp in SPACES:
            es = descend(L.lih_energy, L.lih_energy_grad, t0, sp, lr, steps)
            out[(lr, sp)] = (es[-1] - L.ANSATZ_OPTIMUM, sustained_within(es, L.ANSATZ_OPTIMUM, L.CONVERGENCE_TOL))
    return out


def main():
    print("A. H2 from Hartree-Fock (theta = 0): final error vs FCI (Ha) and first step after which")
    print("   the error stays below chemical accuracy")
    h = h2_study()
    for lr in sorted({k[0] for k in h}):
        print(f"  lr = {lr:<5}" + "".join(f" | {sp}: {h[(lr, sp)][0]:+.1e} ({h[(lr, sp)][1]})" for sp in SPACES))
    print("\nB. 300 random near-pole landscapes E = hz cos + hx sin: success rate")
    b = landscape_study()
    for lr in sorted({k[0] for k in b}):
        print(f"  lr = {lr:<4}" + "".join(f" | {sp}: {b[(lr, sp)]:.2f}" for sp in SPACES))
    print("\nC. LiH 4-parameter Ry/Rx ansatz from Hartree-Fock: final error vs ansatz optimum and")
    print("   sustained-convergence step (tol 1e-5 Ha)")
    c = lih_study()
    for lr in sorted({k[0] for k in c}):
        print(f"  lr = {lr:<4}" + "".join(f" | {sp}: {c[(lr, sp)][0]:+.1e} ({c[(lr, sp)][1]})" for sp in SPACES))
    return h, b, c


if __name__ == "__main__":
    main()
