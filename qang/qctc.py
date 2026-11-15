"""qang.qctc — Curvas temporales cerradas por postselección (P-CTC, Lloyd) frente a D-CTC (Deutsch),
simuladas como mapas sobre el qubit cronológico (CR) y leídas en m-qg. Backend JAX (jax<0.9).
Alineado con `qang.postselection` (§164). Simulación de los MAPAS; la existencia física de CTC es `unverified`.

P-CTC:  ρ_CR → C ρ C† / Tr(C ρ C†),  C = Tr_CTC(U)      (requiere par de Bell + postselección, prob. p_post)
D-CTC:  ρ_CTC = Tr_CR[U(ρ⊗ρ_CTC)U†] (punto fijo),  ρ_out = Tr_CTC[U(ρ⊗ρ_CTC)U†]
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

I2 = jnp.eye(2, dtype=jnp.complex128)
X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
CNOT = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=jnp.complex128)  # CR controla CTC
SWAP = jnp.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=jnp.complex128)


def ptrace_second(rho4: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("iaib->ab", rho4.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2))  # traza sobre el 2º qubit


def ptrace_first(rho4: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("iaib->ab", rho4.reshape(2, 2, 2, 2))


def pctc_operator(U: jnp.ndarray) -> jnp.ndarray:
    """C = Tr_CTC(U) (segundo qubit)."""
    return U.reshape(2, 2, 2, 2).trace(axis1=1, axis2=3)


def pctc_map(rho: jnp.ndarray, U: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """(ρ_out, p_post); p_post = Tr(CρC†)/2² es la probabilidad de postselección sobre el par de Bell."""
    C = pctc_operator(U)
    M = C @ rho @ jnp.conj(C).T
    p = jnp.real(jnp.trace(M)) / 4.0
    return M / jnp.trace(M), p


def dctc_map(rho: jnp.ndarray, U: jnp.ndarray, iters: int = 200) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """Punto fijo de Deutsch por iteración desde I/2 (selección de máxima entropía). Devuelve (ρ_out, ρ_CTC, iters)."""
    rho_ctc = I2 / 2
    for k in range(iters):
        joint = U @ jnp.kron(rho, rho_ctc) @ jnp.conj(U).T
        new = ptrace_first(joint)
        if float(jnp.max(jnp.abs(new - rho_ctc))) < 1e-14:
            rho_ctc = new
            break
        rho_ctc = new
    joint = U @ jnp.kron(rho, rho_ctc) @ jnp.conj(U).T
    return ptrace_second(joint), rho_ctc, k + 1


@jax.jit
def bloch(rho: jnp.ndarray) -> jnp.ndarray:
    Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
    return jnp.real(jnp.array([jnp.trace(rho @ X), jnp.trace(rho @ Y), jnp.trace(rho @ Z)]))


def state_from_bloch(r) -> jnp.ndarray:
    Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
    return 0.5 * (I2 + r[0] * X + r[1] * Y + r[2] * Z)


def shots_multiplier_pctc(p_post: float) -> float:
    return 1.0 / p_post
