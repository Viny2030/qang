"""
Variational real-time evolution (McLachlan) with a weight-preserving ansatz (RESEARCH_NOTES §162).

The ansatz conserves the Hamming weight: initial X gates, then L layers of four RBS (XX+YY) gates on the pairs
(0,1), (2,3), (1,2), (3,4) of a 5-qubit chain and one RZ per qubit (9 parameters per layer). McLachlan's
principle gives A theta_dot = C with A_ij = Re<d_i psi|d_j psi> and C_i = Im<d_i psi|H|psi>, the derivatives
projected orthogonally to psi (global-phase gauge). The circuit depth is fixed: evolving to a later time
changes the angles, not the circuit, whereas a Trotter circuit grows with t. The integration is classical (here
by finite differences on the state vector); the device only prepares and reads the state.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

PAIRS = ((0, 1), (2, 3), (1, 2), (3, 4))


def n_params(layers: int, n: int = 5) -> int:
    return layers * (len(PAIRS) + n)


def ansatz(theta: Sequence[float], layers: int, initial: Sequence[int] = (0, 1), n: int = 5, measure: bool = False):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate

    qc = QuantumCircuit(n)
    for q in initial:
        qc.x(q)
    i = 0
    for _ in range(layers):
        for a, b in PAIRS:
            qc.append(XXPlusYYGate(float(theta[i]), math.pi / 2), [a, b])
            i += 1
        for q in range(n):
            qc.rz(float(theta[i]), q)
            i += 1
    if measure:
        qc.measure_all()
    return qc


def _state(theta, layers, initial, n):
    from qiskit.quantum_info import Statevector

    return Statevector(ansatz(theta, layers, initial, n)).data


def mclachlan_rhs(theta, H: np.ndarray, layers: int, initial=(0, 1), n: int = 5, eps: float = 1e-6) -> np.ndarray:
    th = np.asarray(theta, float)
    psi = _state(th, layers, initial, n)
    D = []
    for i in range(th.size):
        e = np.zeros_like(th)
        e[i] = eps
        d = (_state(th + e, layers, initial, n) - _state(th - e, layers, initial, n)) / (2 * eps)
        D.append(d - psi * np.vdot(psi, d))
    D = np.array(D)
    A = np.real(D.conj() @ D.T)
    C = np.imag(D.conj() @ (H @ psi))
    return np.linalg.lstsq(A, C, rcond=1e-8)[0]


def evolve(H: np.ndarray, layers: int, times: Sequence[float], dt: float = 0.01, initial=(0, 1), n: int = 5,
           seed: int = 0) -> Dict[float, np.ndarray]:
    """RK4 integration of the McLachlan equations from angles ~ 0 (a 1e-3 random offset leaves the singular
    start); returns {t: theta(t)} at the requested times (multiples of dt)."""
    th = np.random.default_rng(seed).normal(0, 1e-3, n_params(layers, n))
    out, t = {}, 0.0
    targets = sorted(times)
    steps = int(round(targets[-1] / dt))
    for s in range(1, steps + 1):
        k1 = mclachlan_rhs(th, H, layers, initial, n)
        k2 = mclachlan_rhs(th + dt / 2 * k1, H, layers, initial, n)
        k3 = mclachlan_rhs(th + dt / 2 * k2, H, layers, initial, n)
        k4 = mclachlan_rhs(th + dt * k3, H, layers, initial, n)
        th = th + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = s * dt
        for T in targets:
            if abs(t - T) < dt / 2:
                out[T] = th.copy()
    return out


def xxz_matrix(n: int, J: float, V: float, h) -> np.ndarray:
    """Dense XXZ Hamiltonian with the conventions of qang.ibm.xxz_exact_occupations."""
    from qiskit.quantum_info import SparsePauliOp

    terms = []
    for k in range(n - 1):
        for P in ("X", "Y"):
            lab = ["I"] * n
            lab[n - 1 - k] = lab[n - 2 - k] = P
            terms.append(("".join(lab), J / 2))
        lab = ["I"] * n
        lab[n - 1 - k] = lab[n - 2 - k] = "Z"
        terms.append(("".join(lab), V / 4))
    for k in range(n):
        lab = ["I"] * n
        lab[n - 1 - k] = "Z"
        terms.append(("".join(lab), h[k] / 2))
    return SparsePauliOp.from_list(terms).to_matrix()


def occupations_from_counts(counts, n: int, weight=None) -> np.ndarray:
    """P(qubit k excited); with ``weight``, among the shots of that Hamming weight (the qg filter)."""
    occ, tot = np.zeros(n), 0
    for k, c in counts.items():
        b = k.replace(" ", "")
        bits = np.array([int(b[-1 - q]) for q in range(n)])
        if weight is not None and bits.sum() != weight:
            continue
        occ += c * bits
        tot += c
    return occ / tot if tot else np.full(n, np.nan)
