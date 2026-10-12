"""
qang.qsp — quantum signal processing (QSP) and weighted QSP in qg terms (§134).

QSP builds a polynomial P(x) as the top-left entry of a product of signal
rotations U(x) = exp(i arccos(x) X) and phase rotations Rz(phi) = exp(i phi Z).
The signal rotation turns the Bloch vector by 2 arccos(x): that is the qang
angle theta = 2 arccos(sqrt(p0)) of a state with sqrt(p0) = x, so the signal is
the amplitude x = cos(theta / 2) and the state U(x)|0> has qg_Z = 2x^2 - 1.

Weighted QSP (Sarkar et al., arXiv:2609.21567) raises each signal rotation to a
weight w_j:

    U_w(x) = Rz(phi_k) U(x)^{w_k} ... Rz(phi_1) U(x)^{w_1} Rz(phi_0),

with standard QSP as w = (1, ..., 1). Integer weights give the same polynomials
with fewer phases.

The read-out is qg as well: a Hadamard test on U_w(x) leaves the ancilla with
p0 = (1 + Re P) / 2, so Re P(x) is the ancilla's qg_Z. Estimated from N shots,
its variance (1 - Re P^2) / N changes with x; in the arcsine (qang angle) scale
the variance is ~1/N everywhere (qang.proportions), which is the point of
``fit_phases(..., scale="arcsine")``.

NumPy only; ``fit_phases`` needs SciPy, ``wqsp_ops`` needs PennyLane (both
imported on use).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "signal_from_qg",
    "qg_of_signal",
    "wqsp_unitary",
    "wqsp_poly",
    "hadamard_test_p0",
    "sample_hadamard_test",
    "fit_phases",
    "wqsp_ops",
]


def signal_from_qg(qg_z):
    """QSP signal x = sqrt((1 + qg_Z) / 2) = cos(theta / 2) of a state with the given qg_Z (x >= 0)."""
    q = np.asarray(qg_z, float)
    if np.any(np.abs(q) > 1):
        raise ValueError("qg_Z must lie in [-1, 1].")
    return np.sqrt((1 + q) / 2)


def qg_of_signal(x):
    """qg_Z = 2x^2 - 1 of the state U(x)|0> (the inverse of signal_from_qg for x >= 0)."""
    x = np.asarray(x, float)
    return 2 * x * x - 1


def _weights(phis, weights):
    phis = np.asarray(phis, float)
    k = len(phis) - 1
    w = np.ones(k) if weights is None else np.asarray(weights, float)
    if len(w) != k:
        raise ValueError(f"{len(phis)} phases need {len(phis) - 1} weights, got {len(w)}.")
    return phis, w


def wqsp_unitary(x, phis: Sequence[float], weights: Optional[Sequence[float]] = None):
    """The 2x2 WQSP unitary for each x (shape x.shape + (2, 2)); weights None = standard QSP."""
    phis, w = _weights(phis, weights)
    x = np.asarray(x, float)
    if np.any(np.abs(x) > 1):
        raise ValueError("the signal x must lie in [-1, 1].")
    a = np.arccos(x)[..., None, None]
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])

    def rz(p):
        return np.diag([np.exp(1j * p), np.exp(-1j * p)])

    U = np.broadcast_to(rz(phis[0]), x.shape + (2, 2)).astype(complex)
    for j in range(1, len(phis)):
        sig = np.cos(w[j - 1] * a) * I + 1j * np.sin(w[j - 1] * a) * X
        U = rz(phis[j]) @ (sig @ U)
    return U


def wqsp_poly(x, phis, weights=None):
    """P(x) = <0| U_w(x) |0> (complex)."""
    return wqsp_unitary(x, phis, weights)[..., 0, 0]


def hadamard_test_p0(x, phis, weights=None):
    """Ancilla P(0) of the Hadamard test on U_w(x): (1 + Re P(x)) / 2; its qg_Z is Re P(x)."""
    return (1 + np.real(wqsp_poly(x, phis, weights))) / 2


def sample_hadamard_test(x, phis, weights=None, shots: int = 100, rng=None):
    """Simulated Hadamard-test counts of outcome 0 at each x (binomial with the exact p0)."""
    rng = np.random.default_rng(rng)
    p0 = np.clip(hadamard_test_p0(x, phis, weights), 0.0, 1.0)
    return rng.binomial(shots, p0)


def _theta(p):
    return 2 * np.arccos(np.sqrt(np.clip(p, 0.0, 1.0)))


def fit_phases(x, k0, shots: int, n_phases: int, weights=None, scale: str = "arcsine",
               starts: int = 8, seed: int = 0):
    """Least-squares phases of a WQSP whose Hadamard test gave ``k0`` zeros out of ``shots`` at each x.

    scale="linear" fits Re P = 2 k0/shots - 1 directly (unequal variance (1 - Re P^2)/shots);
    scale="arcsine" fits the qang angle theta = 2 arccos(sqrt(k0/shots)), variance ~1/shots everywhere.
    Returns (phases, cost) of the best of ``starts`` seeded random starts.
    """
    from scipy.optimize import least_squares

    x = np.asarray(x, float)
    p_hat = np.asarray(k0, float) / shots
    if scale == "linear":
        target = 2 * p_hat - 1

        def resid(ph):
            return np.real(wqsp_poly(x, ph, weights)) - target
    elif scale == "arcsine":
        target = _theta(p_hat)

        def resid(ph):
            return _theta(hadamard_test_p0(x, ph, weights)) - target
    else:
        raise ValueError("scale must be 'linear' or 'arcsine'.")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        r = least_squares(resid, rng.uniform(-np.pi, np.pi, n_phases))
        if best is None or r.cost < best.cost:
            best = r
    return best.x, float(best.cost)


def wqsp_ops(x, phis, weights=None, wires=0):
    """Queue the WQSP sequence on ``wires`` inside a PennyLane QNode (differentiable in the phases).

    exp(i phi Z) = RZ(-2 phi) and exp(i w arccos(x) X) = RX(-2 w arccos(x)); the first phase acts first.
    """
    import pennylane as qml

    k = len(phis) - 1
    w = [1.0] * k if weights is None else list(weights)
    a = qml.math.arccos(x)
    qml.RZ(-2 * phis[0], wires=wires)
    for j in range(1, k + 1):
        qml.RX(-2 * w[j - 1] * a, wires=wires)
        qml.RZ(-2 * phis[j], wires=wires)
