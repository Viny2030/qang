"""
qang.complementarity — wave-particle duality and entropic uncertainty in qg
units (RESEARCH_NOTES §127).

Duality. In a two-path interferometer whose path is the qubit, the
predictability of the path is P = |p0 - p1| = |qg_Z| and the fringe
visibility is V = sqrt(qg_X^2 + qg_Y^2). The duality relation
P^2 + V^2 <= 1 (Greenberger & Yasin 1988; Jaeger, Shimony & Vaidman 1995;
Englert 1996) is therefore the Bloch radius: P^2 + V^2 = r^2, with equality
exactly for pure states.

Entropic uncertainty. For the complementary bases Z and X the
Maassen-Uffink relation (1988) reads H(Z) + H(X) >= 1 bit. In qg units the
Shannon entropy of a basis is the entropic qang of that axis,
qg_S = H((1 + qg)/2), so the relation is

    qg_S(Z) + qg_S(X) >= 1 + S(rho),

where the mixed-state term S(rho) = H((1 + r)/2) (the von Neumann entropy) is
the memory-free case of Berta et al. (2010).

Both are classical results; this module computes them from qg values and from
measurement counts (unbiased radius from qang.statistics, Miller-Madow
entropies from qang.multiqubit).
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

from .core import qg_s_from_qg_z
from .multiqubit import shannon_entropy_miller_madow_bits
from .statistics import radius2_estimate

__all__ = [
    "predictability",
    "visibility",
    "duality_sum",
    "von_neumann_entropy_bits",
    "entropic_sum",
    "entropic_bound",
    "duality_from_counts",
    "entropic_sum_from_counts",
]


def predictability(qg_z: float) -> float:
    """Path predictability P = |p0 - p1| = |qg_Z|."""
    return abs(qg_z)


def visibility(qg_x: float, qg_y: float = 0.0) -> float:
    """Fringe visibility V = sqrt(qg_X^2 + qg_Y^2) (transverse Bloch length)."""
    return math.hypot(qg_x, qg_y)


def duality_sum(qg_x: float, qg_y: float, qg_z: float) -> float:
    """P^2 + V^2 = qg_X^2 + qg_Y^2 + qg_Z^2 = r^2 (<= 1, = 1 for pure states)."""
    return predictability(qg_z) ** 2 + visibility(qg_x, qg_y) ** 2


def von_neumann_entropy_bits(r: float) -> float:
    """S(rho) = H((1 + r)/2) for a qubit with Bloch radius r."""
    if not 0.0 <= r <= 1.0 + 1e-12:
        raise ValueError(f"r must be in [0, 1], got {r}.")
    return qg_s_from_qg_z(min(r, 1.0))


def entropic_sum(qg_z: float, qg_x: float) -> float:
    """qg_S(Z) + qg_S(X): Shannon entropies (bits) of the Z and X measurement outcomes."""
    return qg_s_from_qg_z(qg_z) + qg_s_from_qg_z(qg_x)


def entropic_bound(qg_x: float, qg_y: float, qg_z: float) -> float:
    """Lower bound 1 + S(rho) for qg_S(Z) + qg_S(X) (Maassen-Uffink, with the mixed-state term)."""
    r = math.sqrt(duality_sum(qg_x, qg_y, qg_z))
    return 1.0 + von_neumann_entropy_bits(min(r, 1.0))


def duality_from_counts(counts_x: Tuple[int, int], counts_y: Tuple[int, int], counts_z: Tuple[int, int],
                        unbiased: bool = True) -> Tuple[float, float]:
    """P^2 + V^2 (= r^2) and its standard error from (k0, n_shots) counts in the X, Y, Z bases."""
    return radius2_estimate([counts_x, counts_y, counts_z], unbiased=unbiased)


def entropic_sum_from_counts(counts_z: Tuple[int, int], counts_x: Tuple[int, int],
                             correction: str = "miller_madow") -> float:
    """qg_S(Z) + qg_S(X) from (k0, n_shots) counts, plug-in ("none") or Miller-Madow corrected."""
    total = 0.0
    for k0, n in (counts_z, counts_x):
        if correction == "miller_madow":
            total += shannon_entropy_miller_madow_bits([k0, n - k0])
        elif correction == "none":
            p = k0 / n
            total += qg_s_from_qg_z(2.0 * p - 1.0)
        else:
            raise ValueError("correction must be 'miller_madow' or 'none'.")
    return total
