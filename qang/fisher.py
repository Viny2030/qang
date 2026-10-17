"""
Fisher information, shift rules and the natural gradient in qg terms (RESEARCH_NOTES §159).

One qubit (already in §15.1): the Z-basis Fisher information per shot is 1 in the angle theta and
1 / (1 - qg_Z^2) in qg_Z, so the qang angle is the Fisher-flat coordinate and qg_Z the curved one; the sample
mean qg_hat attains the Cramer-Rao bound (1 - qg_Z^2) / N, and arccos(qg_hat) has the theta-independent variance
1/N of §125.

Shift rules. A gate exp(-i theta G) with G^2 = I/4 makes E(theta) = a + b cos theta + c sin theta. The
derivative b'(theta) = -b sin theta + c cos theta depends on (b, c), and a single evaluation E(theta + s) cannot
separate it from the unknown a: ``indistinguishable_pair`` builds two such functions with equal E(theta + s)
and different E'(theta). Two evaluations suffice (the parameter-shift rule), so 2 per parameter is the minimum,
in any coordinate: dE/dqg = E'(theta) dtheta/dqg uses the same two numbers and adds the factor -1/sin theta.
A coordinate change cannot lower the count; it can only add a pole.

Several qubits, real ansatz. For L layers of RY on every qubit followed by a CZ chain, every state along the
circuit is real, so <Y_k> = 0 before every RY, and the Fubini-Study metric g_ij = Re<d_i psi|d_j psi> -
<d_i psi|psi><psi|d_j psi> has g_ii = Var(Y_k / 2) = 1/4 exactly for every parameter: the angles are
Fisher-flat one by one, and all curvature sits in the off-diagonal terms. Inside one layer
g_ij = <Y_i Y_j> / 4, a covariance of qg_Y values read from one Y-basis circuit (the block-diagonal metric of
Stokes et al. 2020, here as qg correlators). ``block_metric_from_counts`` estimates it from shots.

Qubit k of a state vector is bit k of its index (Qiskit order).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- one qubit


def fisher_theta(theta: float) -> float:
    """Z-basis Fisher information per shot of RY(theta)|0> in theta: 1 (for 0 < theta < pi)."""
    s, c = math.sin(theta), math.cos(theta)
    p = (1 - c) / 2
    return (s / 2) ** 2 / (p * (1 - p)) if 0 < p < 1 else float("nan")


def fisher_qg(q: float) -> float:
    """Fisher information per shot in qg_Z = q: 1 / (1 - q^2)."""
    return 1.0 / (1.0 - q * q) if abs(q) < 1 else float("inf")


def cramer_rao_qg(q: float, n_shots: int) -> float:
    """Cramer-Rao bound on Var(qg_hat): (1 - q^2) / N, attained by the sample mean."""
    return (1.0 - q * q) / n_shots


# ---------------------------------------------------------------- shift rules


def shift_gradient(f, theta: float, s: float = math.pi / 2) -> float:
    """Parameter-shift derivative of a function a + b cos + c sin: (f(theta + s) - f(theta - s)) / (2 sin s)."""
    return (f(theta + s) - f(theta - s)) / (2 * math.sin(s))


def indistinguishable_pair(theta: float, s: float):
    """Two functions a + b cos t + c sin t with the same value at theta + s but different derivatives at theta.

    Returns (f1, f2, d1, d2): one evaluation at any single shift cannot give the derivative."""
    f1 = lambda t: 0.0 + math.cos(t)  # noqa: E731
    # f2 = a + b cos t + c sin t with b, c = (0, 1) shifted by a constant that matches f1 at theta + s
    a2 = math.cos(theta + s) - math.sin(theta + s)
    f2 = lambda t: a2 + math.sin(t)  # noqa: E731
    return f1, f2, -math.sin(theta), math.cos(theta)


# ---------------------------------------------------------------- real hardware-efficient ansatz (statevector)

_Y = np.array([[0, -1j], [1j, 0]])


def _apply_1q(psi: np.ndarray, U: np.ndarray, k: int, n: int) -> np.ndarray:
    t = psi.reshape([2] * n)
    ax = n - 1 - k
    t = np.moveaxis(np.tensordot(U, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def _ry(t: float) -> np.ndarray:
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def _cz_chain_phases(n: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    ph = np.ones(2 ** n)
    for k in range(n - 1):
        ph *= np.where(((idx >> k) & 1) & ((idx >> (k + 1)) & 1), -1.0, 1.0)
    return ph


def ansatz_states(theta: Sequence[float], n: int, layers: int) -> List[np.ndarray]:
    """[psi_0, psi_1, ..., psi_L]: psi_l is the state before the RY layer l (psi_L the output). Each layer is RY on
    every qubit, then a CZ chain. theta has n * layers entries, layer-major."""
    th = np.asarray(theta, float).reshape(layers, n)
    psi = np.zeros(2 ** n)
    psi[0] = 1.0
    cz = _cz_chain_phases(n)
    out = [psi]
    for l in range(layers):
        for k in range(n):
            psi = _apply_1q(psi, _ry(th[l, k]), k, n)
        psi = cz * psi
        out.append(psi)
    return out


def ansatz_state(theta, n: int, layers: int) -> np.ndarray:
    return ansatz_states(theta, n, layers)[-1]


def tfim_hamiltonian(n: int, h: float = 1.0) -> np.ndarray:
    """Open transverse-field Ising chain H = -sum Z_k Z_{k+1} - h sum X_k (dense, real)."""
    idx = np.arange(2 ** n)
    z = [1 - 2 * ((idx >> k) & 1) for k in range(n)]
    H = np.diag(-sum(z[k] * z[k + 1] for k in range(n - 1)).astype(float))
    for k in range(n):
        H[idx ^ (1 << k), idx] += -h
    return H


def energy(theta, H: np.ndarray, n: int, layers: int) -> float:
    psi = ansatz_state(theta, n, layers)
    return float(psi @ H @ psi)


def gradient(theta, H: np.ndarray, n: int, layers: int) -> np.ndarray:
    """Exact gradient by the parameter-shift rule (2 evaluations per parameter)."""
    th = np.asarray(theta, float)
    g = np.zeros_like(th)
    for i in range(th.size):
        e = np.zeros_like(th)
        e[i] = math.pi / 2
        g[i] = (energy(th + e, H, n, layers) - energy(th - e, H, n, layers)) / 2
    return g


def full_metric(theta, n: int, layers: int) -> np.ndarray:
    """Exact Fubini-Study metric: d_i psi = psi(theta + pi e_i) / 2 for RY parameters."""
    th = np.asarray(theta, float)
    psi = ansatz_state(th, n, layers)
    d = []
    for i in range(th.size):
        e = np.zeros_like(th)
        e[i] = math.pi
        d.append(ansatz_state(th + e, n, layers) / 2)
    D = np.array(d)
    G = D @ D.T
    v = D @ psi
    return G - np.outer(v, v)


def _qg_y_correlators(psi: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    ys = [_apply_1q(psi.astype(complex), _Y, k, n) for k in range(n)]
    y1 = np.array([np.real(np.vdot(psi, yk)) for yk in ys])
    yy = np.array([[np.real(np.vdot(ys[i], ys[j])) for j in range(n)] for i in range(n)])
    return y1, yy


def block_metric(theta, n: int, layers: int) -> np.ndarray:
    """Block-diagonal metric: for layer l, g_ij = (<Y_i Y_j> - <Y_i><Y_j>) / 4 on psi_l. Diagonal 1/4 exactly."""
    states = ansatz_states(theta, n, layers)
    g = np.zeros((n * layers, n * layers))
    for l in range(layers):
        y1, yy = _qg_y_correlators(states[l], n)
        g[l * n:(l + 1) * n, l * n:(l + 1) * n] = (yy - np.outer(y1, y1)) / 4
    return g


def y_basis_circuit(theta, n: int, layers: int, layer: int):
    """Qiskit circuit: the ansatz up to (not including) the RY layer ``layer``, then every qubit rotated to the Y
    basis (Sdg, H) and measured."""
    from qiskit import QuantumCircuit

    th = np.asarray(theta, float).reshape(layers, n)
    qc = QuantumCircuit(n)
    for l in range(layer):
        for k in range(n):
            qc.ry(float(th[l, k]), k)
        for k in range(n - 1):
            qc.cz(k, k + 1)
    for k in range(n):
        qc.sdg(k)
        qc.h(k)
    qc.measure_all()
    return qc


def block_metric_from_counts(counts: Dict[str, int], n: int) -> np.ndarray:
    """One layer's metric block (qg_YiYj - qg_Yi qg_Yj) / 4 from Y-basis counts (bitstrings, qubit 0 rightmost)."""
    tot = sum(counts.values())
    y1 = np.zeros(n)
    yy = np.zeros((n, n))
    for k, c in counts.items():
        b = k.replace(" ", "")
        s = np.array([1 - 2 * int(b[-1 - q]) for q in range(n)], float)
        y1 += c * s
        yy += c * np.outer(s, s)
    y1 /= tot
    yy /= tot
    return (yy - np.outer(y1, y1)) / 4
