"""
qang.clock_comparison — differential phase between two clocks (or two atom
interferometers) read in qg units, when their common phase is random
(RESEARCH_NOTES §126).

Two qubit clocks interrogated by the same laser accumulate phases phi + delta
and phi - delta. After a Ramsey sequence each one gives

    qg_Z1 = cos(phi + delta),   qg_Z2 = cos(phi - delta).

When the laser cannot hold phi fixed over the interrogation (phi varies at
random from run to run), each clock alone sees noise, but every pair lies on
the ellipse

    x^2 + y^2 - 2 x y c = 1 - c^2,   c = cos(2 delta),

whose opening gives the differential phase 2 delta: the gravitational redshift
between two heights, for instance. This is correlation (or synchronous)
spectroscopy, and fitting the ellipse is the standard way to extract the
differential phase of coupled atom interferometers (Foster et al. 2002;
Chwalla et al. 2007). This module estimates c (hence delta in [0, pi/2]) from
measured qg pairs:

  * ``ellipse_estimate``: one-parameter least squares on the ellipse
    equation, with the finite-shot bias of the squares removed with
    qang.statistics' unbiased square (N x^2 - 1)/(N - 1);
  * ``moment_estimate``: c = <x y> / sqrt(<x^2> <y^2>), valid only for a
    common phase uniform on the circle; kept as the simple baseline.

Ideal contrast (|qg| reaches 1) and no offset are assumed; a real clock needs
the contrast and offset fitted or calibrated first.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = [
    "differential_phase_from_c",
    "ellipse_estimate",
    "moment_estimate",
    "redshift_phase",
    "simulate_pairs",
]

G_STANDARD = 9.80665          # m s^-2
C_LIGHT = 299_792_458.0       # m s^-1


def redshift_phase(nu_hz: float, height_m: float, t_s: float, g: float = G_STANDARD) -> float:
    """Differential phase 2*delta = 2 pi nu (g h / c^2) t accumulated in a Ramsey time t by two clocks
    of frequency nu separated by a height h (weak-field gravitational redshift)."""
    return 2.0 * math.pi * nu_hz * (g * height_m / C_LIGHT ** 2) * t_s


def differential_phase_from_c(c: float) -> float:
    """delta = arccos(c) / 2 in [0, pi/2] (the sign of delta is not identifiable from the ellipse)."""
    return 0.5 * math.acos(min(1.0, max(-1.0, c)))


def _squares(q: np.ndarray, n_shots: int | None) -> np.ndarray:
    q = np.asarray(q, float)
    if n_shots is None:
        return q * q
    if n_shots < 2:
        raise ValueError("n_shots must be >= 2 for the unbiased square.")
    return (n_shots * q * q - 1.0) / (n_shots - 1.0)


def ellipse_estimate(x: Sequence[float], y: Sequence[float], n_shots: int | None = None) -> Dict[str, float]:
    """
    Least-squares c for the ellipse x^2 + y^2 - 2 x y c = 1 - c^2 from paired
    qg values (x_i, y_i). With ``n_shots`` the squares are replaced by their
    unbiased estimates (shot noise inflates <x^2> by (1 - x^2)/N); the product
    x*y needs no correction because the two clocks' shot noise is independent.
    Returns {"c": c_hat, "delta": delta_hat, "residual": rms of the equation}.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape != y.shape or x.size < 3:
        raise ValueError("x and y must have the same length, at least 3.")
    A = _squares(x, n_shots) + _squares(y, n_shots) - 1.0
    B = x * y
    n = x.size
    # minimize sum (A - 2 B c + c^2)^2 :  n c^3 - 3 S_B c^2 + (2 S_BB + S_A) c - S_AB = 0
    roots = np.roots([n, -3.0 * B.sum(), 2.0 * (B * B).sum() + A.sum(), -(A * B).sum()])
    cands = [float(r.real) for r in roots if abs(r.imag) < 1e-9] + [-1.0, 1.0]
    cands = [min(1.0, max(-1.0, c)) for c in cands]
    cost = lambda c: float(np.sum((A - 2.0 * B * c + c * c) ** 2))  # noqa: E731
    c_hat = min(cands, key=cost)
    return {"c": c_hat, "delta": differential_phase_from_c(c_hat), "residual": math.sqrt(cost(c_hat) / n)}


def moment_estimate(x: Sequence[float], y: Sequence[float], n_shots: int | None = None) -> Dict[str, float]:
    """Baseline c = <x y> / sqrt(<x^2> <y^2>) (common phase uniform on the circle), with unbiased squares when
    ``n_shots`` is given."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mx, my = float(np.mean(_squares(x, n_shots))), float(np.mean(_squares(y, n_shots)))
    if mx <= 0 or my <= 0:
        return {"c": float("nan"), "delta": float("nan")}
    c = float(np.mean(x * y)) / math.sqrt(mx * my)
    return {"c": c, "delta": differential_phase_from_c(c)}


def simulate_pairs(delta: float, n_runs: int, n_shots: int, rng: np.random.Generator,
                   phase_spread: float = math.pi) -> Tuple[np.ndarray, np.ndarray]:
    """Measured (qg_Z1, qg_Z2) for ``n_runs`` runs: common phase uniform in [-spread, spread], each clock read
    with ``n_shots`` projective shots (binomial projection noise), ideal contrast."""
    phi = rng.uniform(-phase_spread, phase_spread, n_runs)
    p1 = (1.0 + np.cos(phi + delta)) / 2.0
    p2 = (1.0 + np.cos(phi - delta)) / 2.0
    k1 = rng.binomial(n_shots, p1)
    k2 = rng.binomial(n_shots, p2)
    return 2.0 * k1 / n_shots - 1.0, 2.0 * k2 / n_shots - 1.0
