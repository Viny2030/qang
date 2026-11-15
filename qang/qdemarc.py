"""qang.qdemarc — Demarcación: dónde aplica `qang` (qubits físicos: QKD, QRNG) y dónde no (PQC, aritmética clásica).
Backend JAX (jax<0.9). Alineado con Cap. 16 antiguo (ML-KEM/ML-DSA/SLH-DSA/HQC: `against`, THEORY_CORE §2.6).

QBER e ⇔ qg_Z = 1 − 2e (eje de la base de reconciliación). Tasa asintótica BB84 (ataque colectivo, un solo paso):
r = 1 − 2 h(e) > 0  ⇔  e < 0.1100  ⇔  qg_Z > 0.780 (780 m-qg).
QRNG de un qubit: H_min = −log2 max(p0,p1) = −log2((1 + |qg_Z|)/2).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@jax.jit
def h2(p):
    p = jnp.clip(p, 1e-300, 1.0); q = jnp.clip(1.0 - p, 1e-300, 1.0)
    return -(p * jnp.log2(p) + q * jnp.log2(q))


@jax.jit
def qber_to_qz(e):
    return 1.0 - 2.0 * e


@jax.jit
def qz_to_qber(qz):
    return (1.0 - qz) / 2.0


@jax.jit
def bb84_rate_from_qz(qz):
    """r = 1 − 2 h((1 − qg_Z)/2)  (bits de clave por bit tamizado; asintótico)."""
    return 1.0 - 2.0 * h2(qz_to_qber(qz))


def bb84_threshold_qz(tol: float = 1e-12) -> float:
    """qg_Z* tal que r = 0 (bisección en [0,1])."""
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if float(bb84_rate_from_qz(jnp.asarray(mid))) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@jax.jit
def qrng_min_entropy_from_qz(qz):
    return -jnp.log2((1.0 + jnp.abs(qz)) / 2.0)


def shots_to_resolve_bias(delta: float, z: float = 3.0) -> float:
    """Disparos para distinguir un sesgo δ en p0 a zσ cerca del ecuador: N = (z/(2δ))²  (σ_qgZ = 1/√N)."""
    return (z / (2.0 * delta)) ** 2
