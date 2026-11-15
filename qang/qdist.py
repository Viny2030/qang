"""qang.qdist — Acoplamiento QAN ↔ distribuciones de probabilidad en espacio qang (JAX, jax<0.9).

Expansión de momentos Z (Walsh–Hadamard):  p(x) = 2^-n Σ_S qg_{Z_S} χ_S(x),  χ_S(x) = Π_{i∈S} z_i(x).
Truncación k-local, divergencias en qg, previa de Haar y presupuesto de disparos.
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------- Walsh / momentos qg
def _chi(n: int) -> jnp.ndarray:
    """Matriz (2^n, 2^n): chi[S, x] = Π_{i∈S} z_i(x); S y x en binario, qubit q ↔ bit n−1−q (big-endian)."""
    idx = jnp.arange(2 ** n)
    bits = ((idx[:, None] >> (n - 1 - jnp.arange(n))[None, :]) & 1)  # (2^n, n)
    par = (bits[:, None, :] * bits[None, :, :]).sum(-1) % 2          # S·x mod 2
    return 1.0 - 2.0 * par


@partial(jax.jit, static_argnames=("n",))
def qg_moments(p: jnp.ndarray, n: int) -> jnp.ndarray:
    """qg_{Z_S} = Σ_x p(x) χ_S(x) para todos los 2^n subconjuntos S (exact; q_∅ = 1)."""
    return _chi(n) @ p


@partial(jax.jit, static_argnames=("n",))
def probs_from_moments(q: jnp.ndarray, n: int) -> jnp.ndarray:
    """Inversa exacta: p(x) = 2^-n Σ_S q_S χ_S(x)."""
    return (_chi(n).T @ q) / (2 ** n)


def moment_order(n: int) -> jnp.ndarray:
    """|S| para cada índice de momento."""
    idx = jnp.arange(2 ** n)
    return ((idx[:, None] >> jnp.arange(n)[None, :]) & 1).sum(-1)


@partial(jax.jit, static_argnames=("n", "k"))
def truncate_k_local(p: jnp.ndarray, n: int, k: int) -> jnp.ndarray:
    """Reconstrucción con momentos de orden ≤ k (proyección L2; puede dar p<0 si k<n)."""
    q = qg_moments(p, n)
    q = jnp.where(moment_order(n) <= k, q, 0.0)
    return probs_from_moments(q, n)


# ---------------------------------------------------------------- divergencias en qg
@jax.jit
def tv_distance(p: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * jnp.sum(jnp.abs(p - r))


@jax.jit
def kl_bits(p: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    p = jnp.clip(p, 1e-300, 1.0); r = jnp.clip(r, 1e-300, 1.0)
    return jnp.sum(p * (jnp.log2(p) - jnp.log2(r)))


@jax.jit
def hellinger2_qg(qz: jnp.ndarray, qz2: jnp.ndarray) -> jnp.ndarray:
    """H² = 1 − √(p0 p0') − √(p1 p1') = 1 − cos(qg_L)  (qg_L = distancia geodésica, Cap. 1)."""
    p0, p0b = (1 + qz) / 2, (1 + qz2) / 2
    return 1.0 - jnp.sqrt(p0 * p0b) - jnp.sqrt((1 - p0) * (1 - p0b))


@jax.jit
def kl_qg_bits(qz: jnp.ndarray, qz2: jnp.ndarray) -> jnp.ndarray:
    """D(qz || qz') binaria en bits (lectura m-qg: dividir argumentos por 1000)."""
    p, r = (1 + qz) / 2, (1 + qz2) / 2
    return kl_bits(jnp.array([p, 1 - p]), jnp.array([r, 1 - r]))


# ---------------------------------------------------------------- previa de Haar y disparos
@jax.jit
def haar_posterior_qz(k0: jnp.ndarray, N: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Posterior Beta(1+k0, 1+N−k0) de p0 (previa Haar = Beta(1,1)); devuelve (E[qg_Z], Var[qg_Z])."""
    a, b = 1.0 + k0, 1.0 + N - k0
    m = a / (a + b)
    v = a * b / ((a + b) ** 2 * (a + b + 1.0))
    return 2.0 * m - 1.0, 4.0 * v


def shots_tv_bound(n: int, M: int, tv: float) -> float:
    """N tal que E[TV] ≤ ½√(M/N) ≤ tv.  M = 2^n − 1 (histograma) o nº de momentos retenidos (exact)."""
    return M / (4.0 * tv ** 2)


def n_moments_k_local(n: int, k: int) -> int:
    from math import comb
    return sum(comb(n, j) for j in range(1, k + 1))
