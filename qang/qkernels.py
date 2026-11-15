"""qang.qkernels — Núcleos cuánticos en espacio qg: forma cerrada para mapas producto, test de inversión
sin ancilla, concentración exponencial (análogo de meseta) y núcleo unario de sector (x·x')² sin concentración.
Backend JAX (jax<0.9). Alineado con `qang.kernels` (Cap. 23 antiguo).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qang.qweightqnn import unary_loader

jax.config.update("jax_enable_x64", True)
RAD_PER_QGA = 2.0 * jnp.pi / 1000.0


@jax.jit
def bloch_from_qga(alpha_qga, phi_qga):
    th, ph = alpha_qga * RAD_PER_QGA, phi_qga * RAD_PER_QGA
    return jnp.stack([jnp.sin(th) * jnp.cos(ph), jnp.sin(th) * jnp.sin(ph), jnp.cos(th)], axis=-1)


@jax.jit
def product_fidelity_kernel(r1: jnp.ndarray, r2: jnp.ndarray) -> jnp.ndarray:
    """k(x,x') = |<ψ(x)|ψ(x')>|² = Π_q (1 + r_q·r'_q)/2 para mapas producto de 1 qubit (exact)."""
    return jnp.prod((1.0 + jnp.sum(r1 * r2, axis=-1)) / 2.0)


@jax.jit
def z_only_kernel(qz1: jnp.ndarray, qz2: jnp.ndarray) -> jnp.ndarray:
    """Núcleo clásico de lectura: Π_q (1 + qg_Z qg_Z')/2 (1 base; no es la fidelidad)."""
    return jnp.prod((1.0 + qz1 * qz2) / 2.0)


def unary_kernel(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """|<x|y>|² = (x·y)² para estados unarios Σ x_k |e_k> (Cap. 23): polinomial de grado 2, sin concentración."""
    return jnp.dot(x, y) ** 2


def unary_inversion_test(x: jnp.ndarray, y: jnp.ndarray, n: int) -> jnp.ndarray:
    """P(|e_0>) tras cargar x y 'descargar' con el cargador de y (U(y)† U(x)|e_0>): = (x·y)² (0 ancillas)."""
    psi_x = unary_loader(x, n)
    # U(y)† como matriz: columnas = U(y)|basis>; usamos el solapamiento directo (equivalente al test de inversión)
    psi_y = unary_loader(y, n)
    return jnp.abs(jnp.vdot(psi_y, psi_x)) ** 2


def kernel_concentration(n: int, samples: int, key) -> float:
    """Var_{x,x'}[k] para características aleatorias de producto (α, φ uniformes en qga): decae ~ (3/4)^n ... exponencial."""
    k1, k2 = jax.random.split(key)
    a = jax.random.uniform(k1, (samples, 2, n), minval=0.0, maxval=500.0)
    p = jax.random.uniform(k2, (samples, 2, n), minval=0.0, maxval=1000.0)
    r = bloch_from_qga(a, p)                                 # (samples, 2, n, 3)
    ks = jax.vmap(lambda rr: product_fidelity_kernel(rr[0], rr[1]))(r)
    return float(jnp.var(ks)), float(jnp.mean(ks))


def shots_for_kernel(k: float, sigma: float) -> float:
    return k * (1 - k) / sigma ** 2
