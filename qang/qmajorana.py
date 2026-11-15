"""qang.qmajorana — Fases de Majorana, masa efectiva de 0νββ, neutrinos estériles (3+1) en 2 qubits y
proyecciones angulares (`qang.angular`) en qga. Backend JAX (jax<0.9).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from qang.qneutrino import pmns, p3_vacuum, KAPPA

jax.config.update("jax_enable_x64", True)
RAD_PER_QGA = 2.0 * jnp.pi / 1000.0


def pmns_majorana(t12, t13, t23, delta, a21, a31) -> jnp.ndarray:
    """U · diag(1, e^{iα21/2}, e^{iα31/2}): las fases de Majorana son R_Z en la base de masa."""
    return pmns(t12, t13, t23, delta) @ jnp.diag(jnp.exp(0.5j * jnp.array([0.0, a21, a31])))


@jax.jit
def m_betabeta(U: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    """|m_ββ| = |Σ_i U_ei² m_i|: suma de tres fasores (aritmética clásica)."""
    return jnp.abs(jnp.sum(U[0, :] ** 2 * m))


def masses_from_lightest(m_light: float, dm21: float, dm31: float, ordering: str = "NO") -> jnp.ndarray:
    if ordering == "NO":
        return jnp.array([m_light, jnp.sqrt(m_light ** 2 + dm21), jnp.sqrt(m_light ** 2 + dm31)])
    m3 = m_light
    m2 = jnp.sqrt(m3 ** 2 + abs(dm31))
    return jnp.array([jnp.sqrt(m2 ** 2 - dm21), m2, m3])


# ---------------------------------------------------------------- 3+1 en 2 qubits
def p4_vacuum(U4: jnp.ndarray, m2: jnp.ndarray, L: float, E: float, alpha: int) -> jnp.ndarray:
    """P(ν_α→ν_β), β = e,μ,τ,s; el sector |11> deja de ser no físico: su peso es P(ν_α→ν_s)."""
    ph = jnp.exp(-1j * 2.0 * KAPPA * m2 * L / E)
    return jnp.abs(U4 @ (ph * jnp.conj(U4)[alpha, :])) ** 2


def u4_minimal(U3: jnp.ndarray, t14: float, t24: float = 0.0) -> jnp.ndarray:
    """Mezcla 3+1 mínima: rotaciones R14(θ14) R24(θ24) sobre U3 ⊕ 1."""
    U = jnp.zeros((4, 4), dtype=jnp.complex128).at[:3, :3].set(U3).at[3, 3].set(1.0)
    def rot(i, j, t):
        R = jnp.eye(4, dtype=jnp.complex128)
        R = R.at[i, i].set(jnp.cos(t)).at[j, j].set(jnp.cos(t)).at[i, j].set(jnp.sin(t)).at[j, i].set(-jnp.sin(t))
        return R
    return rot(0, 3, t14) @ rot(1, 3, t24) @ U


def sterile_sensitivity_floor(eps: float, z: float = 3.0) -> float:
    """Con despolarización ε el peso espurio en |11> es ε/4: sólo P(ν→ν_s) > z·ε/4 es distinguible
    sin calibrar el fondo (cota conservadora, exact para el modelo)."""
    return z * eps / 4.0


# ---------------------------------------------------------------- proyecciones angulares (qang.angular)
@jax.jit
def qz_spin_projection(m: jnp.ndarray, j: jnp.ndarray) -> jnp.ndarray:
    """qg_Z^{(j)} = m/j ∈ {−1, …, +1} con paso 1/j (1000/j m-qg); j = 1/2 recupera el qubit."""
    return m / j


@jax.jit
def spin_coherent_qfi_theta(j: jnp.ndarray) -> jnp.ndarray:
    """QFI del ángulo polar de un estado coherente de espín j: 4 Var(J_y) = 2j rad⁻²  (j = ½ → 1, Cap. 9)."""
    return 2.0 * j
