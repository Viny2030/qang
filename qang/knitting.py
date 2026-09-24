"""
qang.knitting — the sampling cost of circuit cutting (gate cutting /
circuit knitting), written in qg units.

Cutting a two-qubit gate replaces it with a quasi-probability
decomposition (QPD) into local operations. The estimator is still
unbiased, but its variance grows by gamma^2, so the number of shots
needed for a fixed precision is multiplied by gamma^2 (Mitarai & Fujii
2021; Piveteau & Sutter 2024). For the gate families below the optimal
gamma is known in closed form, and it becomes a simple algebraic function
of the gate angle's qg value, qg = cos(theta):

  * Two-qubit Pauli rotations R_PP(theta) = exp(-i theta/2 P(x)P)
    (RXX, RYY, RZZ, RZX, ...):

        gamma = 1 + 2|sin(theta)| = 1 + 2*sqrt(1 - qg^2)

    The cost depends only on the "transverse part" sqrt(1 - qg^2) of the
    gate angle: it is free at qg = +-1 (the identity, or a local Pauli
    product) and maximal, gamma^2 = 9, at qg = 0 (a CNOT-equivalent
    rotation).

  * Controlled rotations CR_P(theta) and CPhase(theta):

        gamma = 1 + 2|sin(theta/2)| = 1 + 2*sqrt((1 - qg)/2)

    Here (1 - qg)/2 is exactly the Born probability P(|1>) of the state
    Ry(theta)|0>, so the cutting cost is 1 + 2*sqrt(P(1)).

Both formulas are pinned against the reference implementation in
qiskit-addon-cutting (QPDBasis.from_instruction(gate).overhead) in
tests/test_knitting.py. When k gates are cut one by one, the total
shot multiplier is the product of their gamma^2 (joint decompositions of
several gates can be cheaper; total_sampling_overhead is the one-by-one
upper bound).

A practical corollary: for R_ZZ(theta) acting on |+>|+>, the local
expectation <X_0> equals cos(theta) = qg exactly, so the qg of an
unknown ZZ-type coupling -- and therefore its cutting cost -- can be read
off a single-qubit X measurement.
"""

from __future__ import annotations

import math
from typing import Iterable

__all__ = [
    "pauli_rotation_cut_gamma",
    "controlled_rotation_cut_gamma",
    "cut_sampling_overhead",
    "total_sampling_overhead",
]


def _check_qg(qg: float) -> float:
    qg = float(qg)
    if not -1.0 <= qg <= 1.0:
        raise ValueError(f"qg must lie in [-1, 1], got {qg}.")
    return qg


def pauli_rotation_cut_gamma(qg: float) -> float:
    """Optimal QPD 1-norm gamma for cutting a two-qubit Pauli rotation
    R_PP(theta) (RXX, RYY, RZZ, RZX, ...) whose angle has qg = cos(theta):
    gamma = 1 + 2*sqrt(1 - qg^2)."""
    qg = _check_qg(qg)
    return 1.0 + 2.0 * math.sqrt(max(0.0, 1.0 - qg * qg))


def controlled_rotation_cut_gamma(qg: float) -> float:
    """Optimal QPD 1-norm gamma for cutting a controlled rotation CR_P(theta)
    or CPhase(theta) whose angle has qg = cos(theta):
    gamma = 1 + 2*sqrt((1 - qg)/2), i.e. 1 + 2*sqrt(P(1))."""
    qg = _check_qg(qg)
    return 1.0 + 2.0 * math.sqrt(max(0.0, (1.0 - qg) / 2.0))


def cut_sampling_overhead(qg: float, gate: str = "pauli_rotation") -> float:
    """Shot multiplier gamma^2 for one cut. gate is "pauli_rotation"
    (RXX/RYY/RZZ/RZX) or "controlled_rotation" (CRX/CRY/CRZ/CPhase)."""
    if gate == "pauli_rotation":
        g = pauli_rotation_cut_gamma(qg)
    elif gate == "controlled_rotation":
        g = controlled_rotation_cut_gamma(qg)
    else:
        raise ValueError(f"gate must be 'pauli_rotation' or 'controlled_rotation', got {gate!r}.")
    return g * g


def total_sampling_overhead(qgs: Iterable[float], gate: str = "pauli_rotation") -> float:
    """Shot multiplier when several gates are cut one by one: the product
    of each cut's gamma^2 (an upper bound; joint cuts can be cheaper)."""
    total = 1.0
    for q in qgs:
        total *= cut_sampling_overhead(q, gate=gate)
    return total
