"""
Single-qubit channels as affine maps of the qg vector (RESEARCH_NOTES §162).

A qubit state is rho = (I + r . sigma) / 2 with r = (qg_X, qg_Y, qg_Z). Every channel with Kraus operators K_a
acts on r as r' = M r + c, with M_ij = Tr(sigma_i E(sigma_j)) / 2 and c_i = Tr(sigma_i E(I)) / 2
(``bloch_affine``). So every channel is an affine map of the qg values, and qg_Z after it is affine in the qg
values before it. In particular:

* amplitude damping (gamma): qg_Z -> (1 - gamma) qg_Z + gamma, qg_X,Y -> sqrt(1 - gamma) qg_X,Y;
* phase damping (lambda): qg_Z unchanged, qg_X,Y -> sqrt(1 - lambda) qg_X,Y;
* depolarizing (p, Kraus form sqrt(1 - 3p/4) I, sqrt(p/4) X, Y, Z): r -> (1 - p) r, so the qang angle
  arccos(qg_Z / |r|) is unchanged (§102);
* thermal relaxation for a time t (T1, T2 <= 2 T1, excited fraction p1):
  qg_Z -> e^(-t/T1) qg_Z + (1 - e^(-t/T1)) (1 - 2 p1), qg_X,Y -> e^(-t/T2) qg_X,Y.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

_P = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]


def apply_kraus(kraus: Sequence[np.ndarray], rho: np.ndarray) -> np.ndarray:
    return sum(K @ rho @ K.conj().T for K in kraus)


def bloch_affine(kraus: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """(M, c) with r' = M r + c for the qg vector r = (qg_X, qg_Y, qg_Z)."""
    M = np.array([[np.real(np.trace(_P[i] @ apply_kraus(kraus, _P[j]))) / 2 for j in range(3)] for i in range(3)])
    c = np.array([np.real(np.trace(_P[i] @ apply_kraus(kraus, np.eye(2)))) / 2 for i in range(3)])
    return M, c


def qg_after(kraus: Sequence[np.ndarray], r: Sequence[float]) -> np.ndarray:
    M, c = bloch_affine(kraus)
    return M @ np.asarray(r, float) + c


def amplitude_damping(gamma: float) -> List[np.ndarray]:
    return [np.array([[1, 0], [0, math.sqrt(1 - gamma)]]), np.array([[0, math.sqrt(gamma)], [0, 0]])]


def phase_damping(lam: float) -> List[np.ndarray]:
    return [np.array([[1, 0], [0, math.sqrt(1 - lam)]]), np.array([[0, 0], [0, math.sqrt(lam)]])]


def depolarizing(p: float) -> List[np.ndarray]:
    return [math.sqrt(1 - 3 * p / 4) * np.eye(2)] + [math.sqrt(p / 4) * P for P in _P]


def thermal_relaxation(t: float, T1: float, T2: float, p1: float = 0.0) -> List[np.ndarray]:
    """Kraus operators of T1/T2 relaxation for a time t (T2 <= 2 T1) toward the excited fraction p1."""
    if T2 > 2 * T1:
        raise ValueError("T2 <= 2 T1 is required.")
    g = 1 - math.exp(-t / T1)
    # amplitude damping toward p1 (generalized), then the extra pure dephasing that gives e^(-t/T2) coherences
    K = [math.sqrt(1 - p1) * np.array([[1, 0], [0, math.sqrt(1 - g)]]),
         math.sqrt(1 - p1) * np.array([[0, math.sqrt(g)], [0, 0]]),
         math.sqrt(p1) * np.array([[math.sqrt(1 - g), 0], [0, 1]]),
         math.sqrt(p1) * np.array([[0, 0], [math.sqrt(g), 0]])]
    coh = math.exp(-t / T2) / math.sqrt(1 - g) if g < 1 else 0.0
    lam = 1 - min(1.0, coh) ** 2
    D = phase_damping(lam)
    return [d @ k for d in D for k in K]
