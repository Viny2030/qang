"""
qang.probability -- the qang probability equation and the milli-qang (RESEARCH_NOTES §167).

Every qang reading of a qubit measured in the Z basis is an exact function of one probability, the
population p0 of the outcome 0 (Born's rule):

    p0 = cos^2(theta/2)  <=>  theta = 2 * arccos(sqrt(p0))
    qg_Z = 2*p0 - 1 = cos(theta)
    qg_S = H(p0)
    m    = 1000 * qg_Z = 2000*p0 - 1000   (milli-qang, m-qg)

The milli-qang scale runs from -1000 m-qg (pole |1>, p0 = 0) to +1000 m-qg (pole |0>, p0 = 1) and is linear
in p0: one m-qg is 0.0005 of probability. It is not linear in the angle: dm/dtheta = -1000 sin(theta), so one
m-qg is one milliradian on the equator and grows towards the poles, where a distance of delta m-qg from the pole
is the angle arccos(1 - delta/1000) ~= sqrt(2 delta / 1000). Shot noise is sigma = 1000 sin(theta) / sqrt(N)
m-qg, so resolving delta m-qg needs N = (1000 sin(theta) / delta)^2 shots. A random relative pulse-amplitude
error of standard deviation sigma moves the expected qg_Z of R_x(theta) from cos(theta) to
cos(theta) exp(-theta^2 sigma^2 / 2) (§160): ``pulse_error_milliqang`` gives that shift in m-qg.

For more than two outcomes the qang angle generalizes to Wootters' statistical distance (Bhattacharyya angle)
arccos(sum_k sqrt(p_k q_k)); for two outcomes, the distance from (p0, 1 - p0) to the pole (1, 0) is theta/2.

``analyze(k0, n)`` reads counts (k0 zeros in n shots) and returns every reading with its interval, the
intervals built once for qg_Z (Wilson) and mapped through the exact monotone relations above. Standard library
only; the estimators come from ``qang.statistics``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .core import MILLIQANG_PER_QANG, _binary_entropy
from .statistics import qg_s_from_qg_z_interval, wilson_qg_estimate

__all__ = [
    "readings_from_p0",
    "milliqang_from_p0",
    "p0_from_milliqang",
    "theta_from_milliqang",
    "milliqang_from_theta",
    "angle_per_milliqang",
    "pole_angle",
    "shot_noise_milliqang",
    "shots_for_resolution",
    "pulse_error_milliqang",
    "statistical_angle",
    "Reading",
    "analyze",
]


def _check_p(p0: float) -> float:
    if not 0.0 <= p0 <= 1.0:
        raise ValueError(f"p0 must lie in [0, 1], got {p0}.")
    return float(p0)


def _check_m(m: float) -> float:
    if not -MILLIQANG_PER_QANG <= m <= MILLIQANG_PER_QANG:
        raise ValueError(f"m must lie in [-1000, 1000] m-qg, got {m}.")
    return float(m)


def readings_from_p0(p0: float) -> dict:
    """The four exact readings of one probability: theta (rad), qg_Z, qg_S (bits) and m-qg."""
    p0 = _check_p(p0)
    return {
        "p0": p0,
        "theta": 2.0 * math.acos(math.sqrt(p0)),
        "qg_z": 2.0 * p0 - 1.0,
        "qg_s": _binary_entropy(p0),
        "milliqang": milliqang_from_p0(p0),
    }


def milliqang_from_p0(p0: float) -> float:
    """m = 2000 p0 - 1000."""
    return 2.0 * MILLIQANG_PER_QANG * _check_p(p0) - MILLIQANG_PER_QANG


def p0_from_milliqang(m: float) -> float:
    """p0 = (1000 + m) / 2000."""
    return (MILLIQANG_PER_QANG + _check_m(m)) / (2.0 * MILLIQANG_PER_QANG)


def theta_from_milliqang(m: float) -> float:
    """theta = arccos(m / 1000), in [0, pi]."""
    return math.acos(_check_m(m) / MILLIQANG_PER_QANG)


def milliqang_from_theta(theta: float) -> float:
    """m = 1000 cos(theta)."""
    return MILLIQANG_PER_QANG * math.cos(theta)


def angle_per_milliqang(theta: float) -> float:
    """Local size of one m-qg in radians, 1 / (1000 sin(theta)): 1 mrad on the equator, infinite at a pole."""
    s = abs(math.sin(theta))
    return math.inf if s == 0.0 else 1.0 / (MILLIQANG_PER_QANG * s)


def pole_angle(delta: float) -> float:
    """Exact angle from the pole |0> of a state delta m-qg below it: arccos(1 - delta/1000) ~= sqrt(2 delta/1000)."""
    if not 0.0 <= delta <= 2.0 * MILLIQANG_PER_QANG:
        raise ValueError(f"delta must lie in [0, 2000] m-qg, got {delta}.")
    return math.acos(1.0 - delta / MILLIQANG_PER_QANG)


def shot_noise_milliqang(theta: float, n_shots: int) -> float:
    """Standard deviation of the m-qg reading from n_shots shots of a pure state: 1000 |sin(theta)| / sqrt(N)."""
    if n_shots < 1:
        raise ValueError("n_shots must be >= 1.")
    return MILLIQANG_PER_QANG * abs(math.sin(theta)) / math.sqrt(n_shots)


def shots_for_resolution(delta: float, theta: float) -> int:
    """Shots needed for a one-sigma resolution of delta m-qg at angle theta: ceil((1000 sin(theta) / delta)^2)."""
    if delta <= 0.0:
        raise ValueError("delta must be > 0.")
    return max(1, math.ceil((MILLIQANG_PER_QANG * abs(math.sin(theta)) / delta) ** 2 - 1e-9))


def pulse_error_milliqang(theta: float, sigma: float) -> float:
    """Shift E[qg_Z] - cos(theta) in m-qg of R_x(theta) under a relative amplitude error of std sigma (§160)."""
    return MILLIQANG_PER_QANG * math.cos(theta) * (math.exp(-(theta * sigma) ** 2 / 2.0) - 1.0)


def statistical_angle(p: Sequence[float], q: Sequence[float]) -> float:
    """Wootters' statistical distance (Bhattacharyya angle) arccos(sum_k sqrt(p_k q_k)) between two distributions."""
    if len(p) != len(q):
        raise ValueError("p and q must have the same number of outcomes.")
    for d in (p, q):
        if any(x < 0.0 for x in d) or not math.isclose(sum(d), 1.0, abs_tol=1e-9):
            raise ValueError("p and q must be probability distributions.")
    bc = sum(math.sqrt(a * b) for a, b in zip(p, q))
    return math.acos(min(1.0, bc))


@dataclass(frozen=True)
class Reading:
    """Every qang reading of k0 zeros in n shots, each as (estimate, low, high)."""

    n_shots: int
    p0: Tuple[float, float, float]
    qg_z: Tuple[float, float, float]
    milliqang: Tuple[float, float, float]
    theta: Tuple[float, float, float]
    qg_s: Tuple[float, float, float]
    deviation_milliqang: Optional[float] = None


def analyze(k0: int, n_shots: int, target_qg_z: Optional[float] = None, z: float = 1.96) -> Reading:
    """Read counts in every qang unit. The Wilson interval of qg_Z is mapped through the exact relations; theta
    decreases with qg_Z, so its bounds swap. With target_qg_z, also the deviation of the estimate in m-qg."""
    w = wilson_qg_estimate(k0, n_shots, z)
    lo, hi = max(-1.0, w.low), min(1.0, w.high)
    s_lo, s_hi = qg_s_from_qg_z_interval(lo, hi)
    dev = None
    if target_qg_z is not None:
        if not -1.0 <= target_qg_z <= 1.0:
            raise ValueError("target_qg_z must lie in [-1, 1].")
        dev = MILLIQANG_PER_QANG * (w.qg_z - target_qg_z)
    return Reading(
        n_shots=n_shots,
        p0=((1.0 + w.qg_z) / 2.0, (1.0 + lo) / 2.0, (1.0 + hi) / 2.0),
        qg_z=(w.qg_z, lo, hi),
        milliqang=(MILLIQANG_PER_QANG * w.qg_z, MILLIQANG_PER_QANG * lo, MILLIQANG_PER_QANG * hi),
        theta=(math.acos(w.qg_z), math.acos(hi), math.acos(lo)),
        qg_s=(_binary_entropy((1.0 + w.qg_z) / 2.0), s_lo, s_hi),
        deviation_milliqang=dev,
    )
