"""qang.qscramble — Hayden–Preskill / Yoshida–Kitaev (decodificador probabilístico), ley P·F = 1/d_A²,
OTOC y curva de Page: Rényi-2 frente a qg_S. Backend JAX (jax<0.9). Alineado con §164.
Orden de qubits: [R, A, B_1..B_nB, B'_1..B'_nB, A', R']  (EPR en (R,A), (B_i,B'_i), (A',R')).
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
EPR = jnp.array([1, 0, 0, 1], dtype=jnp.complex128) / jnp.sqrt(2.0)


def haar_unitary(key, d: int) -> jnp.ndarray:
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (d, d)) + 1j * jax.random.normal(k2, (d, d))
    Q, R = jnp.linalg.qr(A)
    return Q * (jnp.diag(R) / jnp.abs(jnp.diag(R)))


def _apply(psi: jnp.ndarray, U: jnp.ndarray, qubits: Tuple[int, ...], n: int) -> jnp.ndarray:
    k = len(qubits)
    T = psi.reshape([2] * n)
    T = jnp.moveaxis(T, qubits, tuple(range(k)))
    T = (U @ T.reshape(2 ** k, -1)).reshape([2] * n)
    T = jnp.moveaxis(T, tuple(range(k)), qubits)
    return T.reshape(-1)


def initial_state(nB: int) -> jnp.ndarray:
    n = 4 + 2 * nB
    psi = EPR                                     # (R, A)
    for _ in range(nB):
        psi = jnp.kron(psi, jnp.array([1.0, 0.0], dtype=jnp.complex128))  # B_i placeholder
    for _ in range(nB):
        psi = jnp.kron(psi, jnp.array([1.0, 0.0], dtype=jnp.complex128))  # B'_i placeholder
    psi = jnp.kron(psi, EPR)                      # (A', R')
    # EPR (B_i, B'_i): aplicar H⊗CNOT sobre |00>
    H = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / jnp.sqrt(2.0)
    CNOT = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=jnp.complex128)
    for i in range(nB):
        b, bp = 2 + i, 2 + nB + i
        psi = _apply(psi, H, (b,), n)
        psi = _apply(psi, CNOT, (b, bp), n)
    return psi


def yoshida_kitaev(U: jnp.ndarray, nB: int, nD: int) -> Tuple[float, float, float]:
    """Decodificador probabilístico: U en (A,B), U* en (A',B'), proyección EPR en (D,D') con D = primeros nD
    qubits de (A,B). Devuelve (P_EPR, F_EPR, F vía correladores qg de (R,R'))."""
    n = 4 + 2 * nB
    psi = initial_state(nB)
    AB = tuple([1] + [2 + i for i in range(nB)])
    ABp = tuple([2 + 2 * nB] + [2 + nB + i for i in range(nB)])
    psi = _apply(psi, U, AB, n)
    psi = _apply(psi, jnp.conj(U), ABp, n)
    P_epr = jnp.outer(EPR, jnp.conj(EPR))
    for j in range(nD):
        psi = _apply(psi, P_epr, (AB[j], ABp[j]), n)
    P = float(jnp.real(jnp.vdot(psi, psi)))
    psi = psi / jnp.sqrt(P)
    # ρ_{RR'}: qubits 0 y n-1
    T = psi.reshape([2] * n)
    T = jnp.moveaxis(T, (0, n - 1), (0, 1)).reshape(4, -1)
    rho = T @ jnp.conj(T).T
    F = float(jnp.real(jnp.conj(EPR) @ rho @ EPR))
    corr = lambda Pm: float(jnp.real(jnp.trace(rho @ jnp.kron(Pm, Pm))))
    F_qg = (1 + corr(X) - corr(Y) + corr(Z)) / 4
    return P, F, F_qg


def otoc(U: jnp.ndarray, W: jnp.ndarray, V: jnp.ndarray) -> float:
    """OTOC = (1/d) Re Tr(W† V(t)† W V(t)),  V(t) = U† V U."""
    d = U.shape[0]
    Vt = jnp.conj(U).T @ V @ U
    return float(jnp.real(jnp.trace(jnp.conj(W).T @ jnp.conj(Vt).T @ W @ Vt)) / d)


def renyi2_subsystem(psi: jnp.ndarray, n: int, k: int) -> float:
    T = psi.reshape(2 ** k, -1)
    rho = T @ jnp.conj(T).T
    return float(-jnp.log2(jnp.real(jnp.trace(rho @ rho))))


def qg_S_mean(psi: jnp.ndarray, n: int, k: int) -> float:
    """Media de qg_S = H((1+qg_Z)/2) sobre los k primeros qubits (sólo base Z; NO mide entrelazamiento)."""
    p = jnp.abs(psi) ** 2
    idx = jnp.arange(2 ** n)
    out = []
    for q in range(k):
        z = 1.0 - 2.0 * ((idx >> (n - 1 - q)) & 1)
        p0 = (1 + jnp.sum(z * p)) / 2
        p0 = jnp.clip(p0, 1e-300, 1 - 1e-16)
        out.append(float(-(p0 * jnp.log2(p0) + (1 - p0) * jnp.log2(1 - p0))))
    return sum(out) / k
