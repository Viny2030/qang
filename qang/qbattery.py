"""qang.qbattery — Baterías cuánticas y redes: ergotropía en qg, carga, y cadenas de pares de Werner
(swapping multiplica visibilidades). Backend JAX (jax<0.9). (§56, §57)

Qubit con H = ω|1><1| (fundamental |0>): energía E/ω = (1 − qg_Z)/2; ergotropía W/ω = (r − qg_Z)/2,
r = radio local de Bloch (Cap. 5). Carga completa: qg_Z = −1 (−1000 m-qg), W = ω.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from qang.qtele import werner, teleport, bloch, state_from_bloch, resource_correlators

jax.config.update("jax_enable_x64", True)


@jax.jit
def energy_qg(qz):
    return (1.0 - qz) / 2.0


@jax.jit
def ergotropy_qg(r, qz):
    """W/ω = (r − qg_Z)/2; r² = qg_X² + qg_Y² + qg_Z² (3 bases) o r = |qg_Z| si el estado es diagonal."""
    return (r - qz) / 2.0


@jax.jit
def passive_fraction(r, qz):
    """Energía no extraíble / energía almacenada = (1 − r)/(1 − qg_Z)."""
    return (1.0 - r) / (1.0 - qz)


def swap_visibility(V: float, k: int) -> float:
    """k enlaces de Werner V con medidas de Bell perfectas ⇒ V_end = V^k (exact, canal depolarizante compuesto)."""
    return V ** k


def max_links(V: float) -> int:
    """Mayor k con V^k > 1/3 (Cap. 15: utilidad del par)."""
    import math
    return int(math.floor(math.log(1 / 3) / math.log(V) - 1e-12))


def chain_two_links_numeric(V1: float, V2: float):
    """Teleporta el qubit B del primer par a través del segundo par ⇒ qg_ZZ del par extremo = V1·V2."""
    rho1 = werner(V1)
    # estado reducido del qubit B de ρ1 condicionado... usamos linealidad: canal depolarizante sobre B
    # Equivalente exacto: ρ_end = (id ⊗ Λ_{V2})(ρ1) con Λ_V(r) = V r.
    import numpy as np
    I = jnp.eye(2); X = jnp.array([[0, 1], [1, 0]], dtype=complex); Y = jnp.array([[0, -1j], [1j, 0]]); Z = jnp.diag(jnp.array([1., -1.]))
    paulis = [I, X, Y, Z]
    coeffs = [[jnp.real(jnp.trace(rho1 @ jnp.kron(P, Q))) / 4 for Q in paulis] for P in paulis]
    rho_end = sum(coeffs[i][j] * (V2 if j > 0 else 1.0) * jnp.kron(paulis[i], paulis[j]) for i in range(4) for j in range(4))
    return resource_correlators(rho_end)
