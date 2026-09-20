"""
Closed-form transformations between qang-parameterized states.

This module backs Section 13 of the full reference notebook
(``qang_full_reference.ipynb``, "Reformulating the results: where qang
actually accelerates hybrid/quantum code"), which measures a ~1.7x speedup
from using ``transition_probability_qang`` in place of an explicit
statevector overlap when evaluating many transition probabilities (e.g.
building a kernel matrix for a quantum-kernel-method prototype).

Note on scope: the closed form below assumes both states have the same
azimuthal phase phi (i.e. real amplitudes on the great circle qang already
parameterizes via theta alone -- see Section 2 of the paper). It is not a
general two-qubit-state fidelity formula.
"""

import math

__all__ = ["transition_probability_qang"]


def transition_probability_qang(q1: float, q2: float) -> float:
    """Born-rule transition probability / fidelity between two qang states
    that share the same azimuthal phase phi:

        P(q1 -> q2) = |<psi1|psi2>|^2
                     = [1 + q1*q2 + sqrt((1 - q1^2)(1 - q2^2))] / 2

    where q1 = qg_Z(theta1) = cos(theta1) and q2 = qg_Z(theta2) = cos(theta2).

    This is the standard single-qubit fidelity cos^2((theta1 - theta2)/2)
    rewritten directly in terms of qg_Z, avoiding an explicit trip through
    theta or a statevector construction. Raises ValueError if either qg_Z
    value lies outside [-1, 1].
    """
    if not (-1.0 <= q1 <= 1.0 and -1.0 <= q2 <= 1.0):
        raise ValueError("q1 and q2 must lie in [-1.0, 1.0].")
    sin1 = math.sqrt(max(0.0, 1.0 - q1**2))
    sin2 = math.sqrt(max(0.0, 1.0 - q2**2))
    return float((1.0 + q1 * q2 + sin1 * sin2) / 2.0)
