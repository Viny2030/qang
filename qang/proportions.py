"""
qang.proportions — the qang angle as the angular (arcsine) transformation of a
binomial proportion (RESEARCH_NOTES §125).

The qang polar angle and the outcome probability are linked by Born's rule,

    p0 = cos^2(theta/2)        theta = 2 * arccos(sqrt(p0)).

Up to the reflection theta -> pi - theta this is the classical *angular* or
*arcsine* transformation of a proportion used in statistics since Fisher and
Bartlett, phi = 2 * arcsin(sqrt(p)) (Bartlett 1947; Anscombe 1948; Freeman &
Tukey 1950):

    theta(p) = pi - phi(p).

That is why qang.statistics finds a first-order angular uncertainty of
1/sqrt(N) that does not depend on theta: the arcsine transformation is the
variance-stabilizing transformation of the binomial, Var(phi_hat) ~= 1/N.
The difference of two transformed proportions is Cohen's effect size h
(Cohen 1988), h = phi(p1) - phi(p2) = theta(p2) - theta(p1).

Nothing here is new statistics. What this module adds is the same toolkit, in
qang's own angle, for ANY binomial proportion (a measured qubit, or the share
of direct contracts of a public agency, or the share of highly cited papers of
a department): point estimates with Anscombe's small-count correction,
intervals obtained on the angle and mapped back, and two-sample comparisons on
the angle. NumPy-free; only the standard library.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

__all__ = [
    "theta_from_proportion",
    "proportion_from_theta",
    "arcsine",
    "anscombe_theta",
    "theta_variance",
    "proportion_interval",
    "cohens_h",
    "compare_proportions",
    "z_value",
]


def _check_p(p: float) -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}.")
    return float(p)


def _check_kn(k: int, n: int) -> None:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, n], got k={k}, n={n}.")


def theta_from_proportion(p: float) -> float:
    """qang polar angle of a proportion p (= P(outcome 0)): 2*arccos(sqrt(p)) in [0, pi]."""
    return 2.0 * math.acos(math.sqrt(_check_p(p)))


def proportion_from_theta(theta: float) -> float:
    """Inverse of theta_from_proportion: cos^2(theta/2)."""
    if not 0.0 <= theta <= math.pi:
        raise ValueError(f"theta must be in [0, pi], got {theta}.")
    return math.cos(theta / 2.0) ** 2


def arcsine(p: float) -> float:
    """The classical angular transformation phi = 2*arcsin(sqrt(p)) = pi - theta(p)."""
    return 2.0 * math.asin(math.sqrt(_check_p(p)))


def anscombe_theta(k: int, n: int) -> float:
    """
    qang angle of k successes in n trials with Anscombe's correction,
    2*arccos(sqrt((k + 3/8) / (n + 3/4))). The correction keeps the variance
    close to 1/(n + 1/2) down to small counts and avoids the poles at k = 0, n.
    """
    _check_kn(k, n)
    return 2.0 * math.acos(math.sqrt((k + 0.375) / (n + 0.75)))


def theta_variance(n: int, anscombe: bool = True) -> float:
    """Approximate variance of the angle: 1/(n + 1/2) with Anscombe's correction, 1/n without."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    return 1.0 / (n + 0.5) if anscombe else 1.0 / n


def z_value(confidence: float) -> float:
    """Two-sided normal quantile for a confidence level (bisection on erf, standard library only)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if math.erf(mid / math.sqrt(2.0)) < confidence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def proportion_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    (p_hat, low, high): k/n and an interval built on the qang angle,
    theta_A +- z / sqrt(n + 1/2) (clipped to [0, pi]) and mapped back with
    cos^2(theta/2). Because the angle has constant variance, the interval is
    well behaved near 0 and 1, where the textbook Wald interval collapses.
    """
    _check_kn(k, n)
    z = z_value(confidence)
    th = anscombe_theta(k, n)
    half = z * math.sqrt(theta_variance(n))
    t_lo, t_hi = max(0.0, th - half), min(math.pi, th + half)
    # a larger angle means a smaller p0; the poles map exactly to 0 and 1
    low = 0.0 if t_hi >= math.pi else proportion_from_theta(t_hi)
    high = 1.0 if t_lo <= 0.0 else proportion_from_theta(t_lo)
    return k / n, low, high


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's effect size h = phi(p1) - phi(p2) = theta(p2) - theta(p1). |h| 0.2 small, 0.5 medium, 0.8 large."""
    return arcsine(p1) - arcsine(p2)


def compare_proportions(k1: int, n1: int, k2: int, n2: int, confidence: float = 0.95) -> Dict[str, float]:
    """
    Two-sample comparison on the angle: h (from the raw proportions), its
    standard error sqrt(1/n1 + 1/n2), the z statistic, the two-sided p-value
    and a confidence interval for h.
    """
    _check_kn(k1, n1)
    _check_kn(k2, n2)
    h = cohens_h(k1 / n1, k2 / n2)
    se = math.sqrt(1.0 / n1 + 1.0 / n2)
    z = h / se
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    zc = z_value(confidence)
    return {"p1": k1 / n1, "p2": k2 / n2, "h": h, "se": se, "z": z, "p_value": p_value,
            "h_low": h - zc * se, "h_high": h + zc * se}
