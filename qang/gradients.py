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
    directly in theta-space, or in qg-space (raw or regularized).

    space : {"theta", "qg_raw", "qg_clipped", "qg_tikhonov"}
    """
    theta = float(theta0)
    hist = GDHistory(space=space)

    for _ in range(steps):
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
        else:
            if space == "qg_raw":
                dtheta_dqg = inverse_jacobian_raw(theta)
            elif space == "qg_clipped":
                dtheta_dqg = inverse_jacobian_clipped(theta, eps=eps)
            elif space == "qg_tikhonov":
                dtheta_dqg = inverse_jacobian_tikhonov(theta, eps=eps)
            else:
                raise ValueError(
                    "space must be one of 'theta', 'qg_raw', 'qg_clipped', 'qg_tikhonov'."
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
    """Run all four optimization spaces from the same near-pole start point."""
    spaces = ["theta", "qg_raw", "qg_clipped", "qg_tikhonov"]
    return {
        s: run_gradient_descent(s, theta0, lr=lr, steps=steps, eps=eps, h_z=h_z, h_x=h_x)
        for s in spaces
    }
