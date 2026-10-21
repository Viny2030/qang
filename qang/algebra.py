"""
Closed-form qg algebra: the universal single-qubit gate, reflections and transition probabilities written in qg_Z,
and the qudit generalization of qg_Z (RESEARCH_NOTES §166).

With q = qg_Z = cos(theta), cos(theta/2) = sqrt((1 + q)/2) and sin(theta/2) = sqrt((1 - q)/2), so every
single-qubit gate is an algebraic function of q and two phases:

    U3_qang(q, phi, lam) = [[ sqrt((1+q)/2),               -e^{i lam} sqrt((1-q)/2)        ],
                            [ e^{i phi} sqrt((1-q)/2),      e^{i(phi+lam)} sqrt((1+q)/2)    ]],

equal to Qiskit's U(theta, phi, lam) with theta = arccos q. Its transition matrix |U_ij|^2 is the doubly
stochastic [[(1+q)/2, (1-q)/2], [(1-q)/2, (1+q)/2]], rational in q and free of the phases (unistochastic by
construction).

The Householder reflector through |psi(q)> = sqrt((1+q)/2)|0> + sqrt((1-q)/2)|1> is
I - 2|psi><psi| = -q Z - sqrt(1 - q^2) X, and the Grover diffuser about the same state is its negative,
2|psi><psi| - I = q Z + sqrt(1 - q^2) X. The transition probability between two pure states with qg vectors
r1, r2 is (1 + r1 . r2)/2; on a common meridian it is [1 + q1 q2 + sqrt((1 - q1^2)(1 - q2^2))]/2.

Qudits. The diagonal generalized Gell-Mann matrices (the Cartan subalgebra of su(d)),
lambda_k = sqrt(2/(k(k+1))) diag(1, ..., 1, -k, 0, ..., 0) with k ones, k = 1 .. d-1, are the Z-basis-measurable
generalization of sigma_z (d = 2 gives lambda_1 = sigma_z, so the single value is qg_Z). Their expectation values are
linear in the populations and determine them: p = 1/d + (1/2) sum_k <lambda_k> diag(lambda_k).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def u3_qang(q: float, phi: float = 0.0, lam: float = 0.0) -> np.ndarray:
    """Universal single-qubit gate in qg_Z: Qiskit's U(arccos q, phi, lam) without trigonometry in q."""
    if not -1.0 <= q <= 1.0:
        raise ValueError("q = qg_Z must be in [-1, 1].")
    c, s = math.sqrt((1 + q) / 2), math.sqrt((1 - q) / 2)
    return np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def transition_matrix(q: float) -> np.ndarray:
    """|U3_qang_ij|^2: doubly stochastic, rational in q, independent of the phases."""
    return np.array([[(1 + q) / 2, (1 - q) / 2], [(1 - q) / 2, (1 + q) / 2]])


def householder_qg(q: float) -> np.ndarray:
    """Householder reflector I - 2|psi(q)><psi(q)| = -q Z - sqrt(1 - q^2) X."""
    return -q * _Z - math.sqrt(max(0.0, 1 - q * q)) * _X


def grover_diffuser_qg(q: float) -> np.ndarray:
    """Grover diffuser 2|psi(q)><psi(q)| - I = -householder_qg(q)."""
    return -householder_qg(q)


def qg_vector(q: float, phi: float = 0.0) -> np.ndarray:
    """(qg_X, qg_Y, qg_Z) of the pure state with polar bias q and azimuth phi."""
    t = math.sqrt(max(0.0, 1 - q * q))
    return np.array([t * math.cos(phi), t * math.sin(phi), q])


def transition_probability(q1: float, q2: float, phi1: float = 0.0, phi2: float = 0.0) -> float:
    """|<psi1|psi2>|^2 = (1 + r1 . r2)/2; with phi1 = phi2 it is [1 + q1 q2 + sqrt((1-q1^2)(1-q2^2))]/2."""
    return float((1 + qg_vector(q1, phi1) @ qg_vector(q2, phi2)) / 2)


# ---------------------------------------------------------------- qudits

def cartan_generators(d: int) -> np.ndarray:
    """Diagonals of the d - 1 diagonal generalized Gell-Mann matrices, shape (d - 1, d); Tr(l_j l_k) = 2 delta_jk."""
    if d < 2:
        raise ValueError("d >= 2")
    out = np.zeros((d - 1, d))
    for k in range(1, d):
        out[k - 1, :k] = 1.0
        out[k - 1, k] = -k
        out[k - 1] *= math.sqrt(2 / (k * (k + 1)))
    return out


def cartan_values(probs: Sequence[float]) -> np.ndarray:
    """<lambda_k> for k = 1 .. d-1 from the Z-basis populations (d = 2: the single value is qg_Z)."""
    p = np.asarray(probs, float)
    return cartan_generators(p.size) @ p


def populations_from_cartan(values: Sequence[float]) -> np.ndarray:
    """Inverse of cartan_values: p = 1/d + (1/2) sum_k <lambda_k> diag(lambda_k)."""
    v = np.asarray(values, float)
    d = v.size + 1
    return 1.0 / d + 0.5 * (cartan_generators(d).T @ v)
