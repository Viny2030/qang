"""
quang.gradients — the coordinate-singularity limitation from Section 4.1 of
the paper, a regularized alternative, and a toy VQE benchmark comparing
optimizer convergence in theta-space vs raw qg-space vs regularized qg-space.

This addresses Future Research Direction #3 ("Characterization of qang
gradients: study and, where possible, regularize the coordinate singularity
at theta = 0, pi ... and compare the convergence behavior of qg-space
optimization against standard theta-space parameterization").

Background (from the paper)
----------------------------
    d(qg_Z)/d(theta)  =  -sin(theta)                    (exact)
    d(theta)/d(qg_Z)  =  -1/sin(theta)                   (exact, diverges at
                                                           theta = 0, pi)
    dE/d(qg)          =  (dE/d(theta)) * (d(theta)/d(qg))

Two regularizations of the inverse Jacobian are provided:

  * ``inverse_jacobian_clipped``   floors |sin(theta)| at ``eps`` before
    inverting. This bounds the conversion factor at +/-1/eps near a pole
    while leaving it essentially exact away from the poles -- a direct,
    interpretable "safety clamp" on step size.
  * ``inverse_jacobian_tikhonov``  a smooth Tikhonov-style regularization,
    sin(theta) / (sin(theta)**2 + eps**2), which suppresses (rather than
    clamps) the qg-space gradient exactly at a pole. Bounded and smooth
    everywhere, including at theta = 0, pi.

Both coincide with the exact -1/sin(theta) away from the poles as eps -> 0,
and both are finite everywhere, which is the property the paper's Section
4.1 "Limitation" paragraph asks for.

A fifth optimization space: ``theta_pole_damped``
--------------------------------------------------
The three qg-space variants above all share a second, independent problem,
demonstrated on a real molecule in examples/vqe_h2_qg_vs_theta.py: every
qg-space step round-trips through ``qg = cos(theta)`` and back via
``theta = acos(qg)``, and ``acos`` only ever returns a value in [0, pi].
Since cos(theta) = cos(-theta) = cos(2*pi - theta), that round trip is not
just numerically delicate near the poles, it is *structurally* unable to
reach a minimum that lies in the other half of the circle -- for H2,
started at the physical Hartree-Fock point, all three variants converge
to exactly the Hartree-Fock energy and recover zero correlation energy,
not because of slow convergence but because that energy is a hard ceiling
on what arccos's range can reach.

``theta_pole_damped`` keeps every update in theta-space -- there is no
qg-space round trip, so this second failure mode cannot occur -- and reuses
only the same pole-proximity floor already defined for the regularized
Jacobians, applied directly as a per-step damping factor:

    pole_damping_factor(theta, eps) = max(|sin(theta)|, eps)
    theta_new = theta - lr * pole_damping_factor(theta, eps) * dE/d(theta)

This is a position-dependent trust-region-style damping of the raw
theta-space step, in the spirit of Levenberg-Marquardt damping (Levenberg,
1944; Marquardt, 1963) -- it is *not* a new class of optimizer, and the
underlying idea (shrink the step where the local problem is
ill-conditioned) is a long-established one. What is specific to this
framework is where the damping schedule comes from: it is not a free
hyperparameter tuned by hand, but falls directly out of the same Bloch
-sphere pole geometry already used for the regularized Jacobians above
(and empirically, the resulting robustness is insensitive to the exact
choice of eps -- see tests/test_gradients.py).

The concrete, measured effect (see examples/pole_damped_gradient_descent_robustness.py
for the full study, and examples/vqe_h2_qg_vs_theta.py for a real-molecule
demonstration): at a *safe*, well-tuned learning rate, ``theta_pole_damped``
matches plain theta-space gradient descent, at the cost of a modest number
of extra iterations spent near the pole. At a *badly-tuned, too-aggressive*
learning rate -- exactly the kind of hyperparameter mistake that is easy to
make when trying a new ansatz on real NISQ hardware -- plain theta-space
gradient descent frequently overshoots and diverges, while
``theta_pole_damped`` continues to converge reliably, across a 300-trial
statistical study over random one-qubit cost landscapes and, deterministically,
on the real H2 molecular Hamiltonian. This comes with two honestly-measured
costs, not hidden: the extra iterations already mentioned at safe learning
rates, and a small (roughly 2-5% in the aggressive-learning-rate regime of
the statistical study) failure mode where the optimizer gets "trapped"
oscillating near the pole it started at instead of escaping toward the
true optimum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List


# --------------------------------------------------------------------- #
# Jacobian and its regularized inverses
# --------------------------------------------------------------------- #
def jacobian(theta: float) -> float:
    """d(qg_Z)/d(theta) = -sin(theta). Exact; vanishes at theta = 0, pi."""
    return -math.sin(theta)


def inverse_jacobian_raw(theta: float) -> float:
    """d(theta)/d(qg_Z) = -1/sin(theta). Exact; diverges at theta = 0, pi."""
    s = math.sin(theta)
    if abs(s) < 1e-12:
        return float("-inf")
    return -1.0 / s


def inverse_jacobian_clipped(theta: float, eps: float = 0.05) -> float:
    """
    Floor |sin(theta)| at eps before inverting. Bounded by 1/eps in
    magnitude; exact away from the poles.
    """
    s = math.sin(theta)
    s_floored = math.copysign(max(abs(s), eps), s if s != 0.0 else 1.0)
    return -1.0 / s_floored


def inverse_jacobian_tikhonov(theta: float, eps: float = 0.05) -> float:
    """
    Smooth Tikhonov-style regularization: -sin(theta) / (sin(theta)**2 + eps**2).
    Bounded and smooth everywhere, including exactly at theta = 0, pi (-> 0).
    """
    s = math.sin(theta)
    return -s / (s * s + eps * eps)


def pole_damping_factor(theta: float, eps: float = 0.05) -> float:
    """
    |sin(theta)|, floored at eps: a position-dependent, Levenberg-Marquardt
    -style trust-region damping factor for a *theta-space* gradient step
    (see this module's docstring, "A fifth optimization space"). Always in
    (0, 1], so it only ever shrinks (never enlarges) the raw theta-space
    step: 1.0 away from the poles (no effect), down to eps exactly at a
    pole (maximal damping).
    """
    return max(abs(math.sin(theta)), eps)


# --------------------------------------------------------------------- #
# Toy VQE loss: E(theta) = <psi(theta)| h_z*sigma_z + h_x*sigma_x |psi(theta)>
# for the real (phi=0) state |psi(theta)> = cos(theta/2)|0> + sin(theta/2)|1>,
# for which <sigma_z> = cos(theta) = qg_Z(theta) and <sigma_x> = sin(theta).
#
# Default h_z=0.0, h_x=-1.0 places the (unique, interior) minimum at
# theta = pi/2 -- deliberately far from both poles -- so that starting the
# optimizer AT a pole (theta0 near 0 or pi) is a genuine stress test of
# having to escape the coordinate singularity discussed in Section 4.1,
# rather than of a minimum that happens to sit at the domain boundary.
# --------------------------------------------------------------------- #
def toy_vqe_energy(theta: float, h_z: float = 0.0, h_x: float = -1.0) -> float:
    return h_z * math.cos(theta) + h_x * math.sin(theta)


def toy_vqe_grad_theta(theta: float, h_z: float = 0.0, h_x: float = -1.0) -> float:
    """dE/d(theta) for the toy Hamiltonian above."""
    return -h_z * math.sin(theta) + h_x * math.cos(theta)


# --------------------------------------------------------------------- #
# Gradient descent runner
# --------------------------------------------------------------------- #
@dataclass
class GDHistory:
    space: str
    theta: List[float] = field(default_factory=list)
    qg: List[float] = field(default_factory=list)
    energy: List[float] = field(default_factory=list)
    diverged: bool = False


def run_gradient_descent(
    space: str,
    theta0: float,
    lr: float = 0.05,
    steps: int = 200,
    eps: float = 0.05,
    h_z: float = 0.0,
    h_x: float = -1.0,
    grad_fn: Callable[[float, float, float], float] = toy_vqe_grad_theta,
    energy_fn: Callable[[float, float, float], float] = toy_vqe_energy,
) -> GDHistory:
    """
    Minimize energy_fn(theta) by gradient descent, parameterized either
    directly in theta-space, in qg-space (raw or regularized), or in
    theta-space with pole-proximity damping applied to the raw step.

    space : {"theta", "qg_raw", "qg_clipped", "qg_tikhonov", "theta_pole_damped"}
    """
    theta = float(theta0)
    hist = GDHistory(space=space)

    for _ in range(steps):
        if space != "theta_pole_damped":
            # "theta" and every qg-space variant are kept inside [0, pi]:
            # the qg-space ones because acos(qg) never returns anything
            # else, and plain "theta" for consistency with the rest of
            # this benchmark. "theta_pole_damped" is deliberately exempt
            # from this clamp -- it is a *theta-space* update with no
            # qg round trip at all, and letting theta range freely is
            # exactly what lets it recover from an overshoot that would
            # otherwise look like divergence (see this module's
            # docstring and tests/test_gradients.py).
            theta = min(max(theta, 1e-9), math.pi - 1e-9)  # stay in the physical domain
        qg = math.cos(theta)
        e = energy_fn(theta, h_z, h_x)
        hist.theta.append(theta)
        hist.qg.append(qg)
        hist.energy.append(e)

        if not math.isfinite(e) or abs(theta) > 1e6:
            hist.diverged = True
            break

        dE_dtheta = grad_fn(theta, h_z, h_x)

        if space == "theta":
            theta = theta - lr * dE_dtheta
        elif space == "theta_pole_damped":
            theta = theta - lr * pole_damping_factor(theta, eps=eps) * dE_dtheta
        else:
            if space == "qg_raw":
                dtheta_dqg = inverse_jacobian_raw(theta)
            elif space == "qg_clipped":
                dtheta_dqg = inverse_jacobian_clipped(theta, eps=eps)
            elif space == "qg_tikhonov":
                dtheta_dqg = inverse_jacobian_tikhonov(theta, eps=eps)
            else:
                raise ValueError(
                    "space must be one of 'theta', 'qg_raw', 'qg_clipped', "
                    "'qg_tikhonov', 'theta_pole_damped'."
                )
            dE_dqg = dE_dtheta * dtheta_dqg
            new_qg = max(-1.0, min(1.0, qg - lr * dE_dqg))
            theta = math.acos(new_qg)

        if not math.isfinite(theta):
            hist.diverged = True
            break

    return hist


def benchmark(
    theta0: float = 0.05,
    lr: float = 0.05,
    steps: int = 200,
    eps: float = 0.05,
    h_z: float = 0.0,
    h_x: float = -1.0,
) -> Dict[str, GDHistory]:
    """Run all five optimization spaces from the same near-pole start point."""
    spaces = ["theta", "qg_raw", "qg_clipped", "qg_tikhonov", "theta_pole_damped"]
    return {
        s: run_gradient_descent(s, theta0, lr=lr, steps=steps, eps=eps, h_z=h_z, h_x=h_x)
        for s in spaces
    }
