"""qang.qtele — Teleportación con recurso de Werner: fidelidad, umbral clásico y lectura del canal en m-qg.
Backend JAX (jax<0.9). Alineado con `qang.qpon` (core NumPy).

Recurso: ρ_V = V|Φ+><Φ+| + (1−V) I/4  ⇒  qg_XX = V, qg_YY = −V, qg_ZZ = V.
Teleportación estándar (medida de Bell + corrección de Pauli): canal depolarizante r → V r,
F̄ = (1+V)/2; umbral clásico F̄ = 2/3 ⇔ V = 1/3 ⇔ qg_ZZ = 333 m-qg.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

I2 = jnp.eye(2, dtype=jnp.complex128)
X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
PHI_PLUS = jnp.array([1, 0, 0, 1], dtype=jnp.complex128) / jnp.sqrt(2.0)


def kron(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = jnp.kron(out, m)
    return out


@jax.jit
def werner(V: jnp.ndarray) -> jnp.ndarray:
    return V * jnp.outer(PHI_PLUS, jnp.conj(PHI_PLUS)) + (1.0 - V) * jnp.eye(4) / 4.0


@jax.jit
def resource_correlators(rho2: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """(qg_XX, qg_YY, qg_ZZ) del recurso de 2 qubits."""
    f = lambda P: jnp.real(jnp.trace(rho2 @ kron(P, P)))
    return f(X), f(Y), f(Z)


@jax.jit
def singlet_fraction_from_qg(qxx: jnp.ndarray, qyy: jnp.ndarray, qzz: jnp.ndarray) -> jnp.ndarray:
    """F_s = <Φ+|ρ|Φ+> = (1 + qg_XX − qg_YY + qg_ZZ)/4  (3 bases, 0 ancillas; exact)."""
    return (1.0 + qxx - qyy + qzz) / 4.0


@jax.jit
def teleport_fidelity_from_singlet(Fs: jnp.ndarray) -> jnp.ndarray:
    """F̄ = (2 F_s + 1)/3  (Horodecki); umbral clásico 2/3 ⇔ F_s = 1/2."""
    return (2.0 * Fs + 1.0) / 3.0


@jax.jit
def teleport_fidelity_werner_qzz(qzz: jnp.ndarray) -> jnp.ndarray:
    """Bajo modelo de Werner V = qg_ZZ (1 base): F̄ = (1 + qg_ZZ)/2."""
    return (1.0 + qzz) / 2.0


def bell_basis() -> jnp.ndarray:
    s = 1 / jnp.sqrt(2.0)
    return jnp.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1, -1, 0]], dtype=jnp.complex128) * s


def teleport(rho_in: jnp.ndarray, rho_res: jnp.ndarray) -> jnp.ndarray:
    """Teleportación completa (qubits: entrada ⊗ recurso_A ⊗ recurso_B), medida de Bell en (in, A),
    corrección de Pauli en B. Devuelve ρ_out (2×2). Exact."""
    rho = kron(rho_in, rho_res)                    # 8×8
    corr = [I2, Z, X, X @ Z]                        # Φ+, Φ−, Ψ+, Ψ−
    out = jnp.zeros((2, 2), dtype=jnp.complex128)
    for k, b in enumerate(bell_basis()):
        Pk = jnp.outer(b, jnp.conj(b))
        M = kron(Pk, I2)
        rb = (M @ rho @ M).reshape(4, 2, 4, 2)
        rb = jnp.einsum("iaib->ab", rb)             # traza parcial sobre (in, A)
        out = out + corr[k] @ rb @ jnp.conj(corr[k]).T
    return out


@jax.jit
def bloch(rho: jnp.ndarray) -> jnp.ndarray:
    return jnp.real(jnp.array([jnp.trace(rho @ X), jnp.trace(rho @ Y), jnp.trace(rho @ Z)]))


def state_from_bloch(r: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (I2 + r[0] * X + r[1] * Y + r[2] * Z)


def shots_for_fidelity(V: float, sigma_F: float) -> float:
    """F̄ = (1+qg_ZZ)/2 ⇒ σ_F = ½√((1−V²)/N) ⇒ N = (1−V²)/(4σ_F²)."""
    return (1.0 - V ** 2) / (4.0 * sigma_F ** 2)
