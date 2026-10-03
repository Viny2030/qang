"""
qang.geometric -- geometric (Berry / Pancharatnam) phase in qg units
(RESEARCH_NOTES §84).

A qubit carried around a closed loop on the Bloch sphere picks up a geometric
phase equal to minus half the solid angle the loop encloses:

    gamma = -Omega / 2      (mod 2 pi)

This is Stokes' theorem on the sphere: the line integral of the Berry
connection around the loop equals the flux of the Berry curvature, a monopole
field of strength 1/2, through the enclosed cap. In qg units the loop is a
path of qg vectors (qg_X, qg_Y, qg_Z), and the phase is a value of the qg_Phi
unit (qang.phase), a fraction of a full turn. For a loop at constant qg_Z
around the Z axis (a cone), Omega = 2 pi (1 - qg_Z), so

    phi = gamma / (2 pi) = (qg_Z - 1) / 2    (mod 1):

the geometric phase of a cone loop is fixed by qg_Z alone. For polarized
light the same phase is Pancharatnam's phase on the Poincare sphere
(qang.polarization). The physics is standard; the module states it in qang's
units and checks it three ways (overlaps, solid angle, curvature flux).
NumPy only.
"""

from __future__ import annotations

import math

import numpy as np

from .phase import QangPhi

__all__ = ["state_from_bloch", "bloch_from_state", "pancharatnam_phase", "solid_angle",
           "berry_phase_from_solid_angle", "cone_phase_turns", "curvature_flux", "geometric_phase_qg"]


def state_from_bloch(r) -> np.ndarray:
    """Pure state with Bloch vector r = (qg_X, qg_Y, qg_Z), |r| = 1
    (gauge: real non-negative first component when possible)."""
    x, y, z = (float(v) for v in r)
    theta = math.acos(max(-1.0, min(1.0, z)))
    phi = math.atan2(y, x)
    return np.array([math.cos(theta / 2), np.exp(1j * phi) * math.sin(theta / 2)])


def bloch_from_state(psi) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex)
    a, b = psi / np.linalg.norm(psi)
    return np.array([2 * np.real(np.conj(a) * b), 2 * np.imag(np.conj(a) * b), abs(a) ** 2 - abs(b) ** 2])


def pancharatnam_phase(states) -> float:
    """Gauge-invariant geometric phase of a closed discrete loop of states,
    gamma = -arg prod_k <psi_k | psi_{k+1}> (the last state connects to the
    first). Returned in (-pi, pi]."""
    prod = 1.0 + 0.0j
    n = len(states)
    for k in range(n):
        prod *= np.vdot(states[k], states[(k + 1) % n])
    return float(-np.angle(prod))


def _triangle(a, b, c) -> float:
    # signed solid angle of the spherical triangle (Van Oosterom-Strackee)
    num = np.dot(a, np.cross(b, c))
    den = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    return 2.0 * math.atan2(num, den)


def solid_angle(path, reference=None) -> float:
    """Signed solid angle enclosed by a closed loop of unit Bloch vectors
    (counterclockwise seen from outside = positive), as a fan of spherical
    triangles from ``reference`` (default: the normalized mean of the path)."""
    P = np.asarray(path, float)
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    ref = P.mean(axis=0) if reference is None else np.asarray(reference, float)
    ref = ref / np.linalg.norm(ref)
    n = len(P)
    return float(sum(_triangle(ref, P[k], P[(k + 1) % n]) for k in range(n)))


def berry_phase_from_solid_angle(omega: float) -> float:
    """gamma = -Omega / 2, wrapped to (-pi, pi]."""
    g = -0.5 * omega
    return float(math.atan2(math.sin(g), math.cos(g)))


def cone_phase_turns(qg_z: float) -> float:
    """Geometric phase of a loop at constant qg_Z around the Z axis, in turns
    (the qg_Phi unit): (qg_Z - 1) / 2 mod 1."""
    return float(((qg_z - 1.0) / 2.0) % 1.0)


def curvature_flux(qg_z_cap: float, n_theta: int = 400, n_phi: int = 400) -> float:
    """Numerical flux of the Berry curvature F = r / 2 (unit monopole of
    strength 1/2) through the polar cap qg_Z >= qg_z_cap. Stokes' theorem says
    it equals Omega / 2 = pi (1 - qg_z_cap), the magnitude of the phase."""
    theta_max = math.acos(qg_z_cap)
    th = (np.arange(n_theta) + 0.5) * theta_max / n_theta
    dA = np.sin(th) * (theta_max / n_theta) * (2 * math.pi / n_phi)
    return float(0.5 * n_phi * dA.sum())  # F . n = 1/2 on the unit sphere


def geometric_phase_qg(path) -> QangPhi:
    """Geometric phase of a closed loop of qg vectors, as a qg_Phi value
    (fraction of a turn), computed from the state overlaps."""
    states = [state_from_bloch(r) for r in path]
    return QangPhi(pancharatnam_phase(states) / (2 * math.pi))
