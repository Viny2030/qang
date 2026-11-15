"""qang.qdualrail — Qubit fotónico de doble carril, pérdida como borrado anunciado y filtro de sector
(peso de Hamming 1). Backend JAX (jax<0.9). Alineado con `qang.qpon` / `qang.sectors` (§158).

Codificación: |0_L> = |10>, |1_L> = |01> (modos a, b). Pérdida con transmisión η por modo: amplitud
damping independiente ⇒ el peso sale del sector {|10>,|01>} hacia |00> (vacío).
Sin filtro (vacío leído como "0"): qg_Z' = η qg_Z + (1 − η)  (sesgo hacia +1000 m-qg).
Con filtro de sector: qg_Z recuperado sin sesgo con tasa η (exact).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def transmission(L_km: jnp.ndarray, alpha_db_km: float = 0.2) -> jnp.ndarray:
    """η = 10^(−α L / 10)."""
    return 10.0 ** (-alpha_db_km * L_km / 10.0)


def _kraus_loss(eta):
    """Amortiguamiento de amplitud en un modo {|0>,|1>} (a lo sumo un fotón)."""
    K0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(eta)]], dtype=jnp.complex128)
    K1 = jnp.array([[0.0, jnp.sqrt(1.0 - eta)], [0.0, 0.0]], dtype=jnp.complex128)
    return K0, K1


def dual_rail_state(qz: float, qx: float = 0.0, qy: float = 0.0) -> jnp.ndarray:
    """ρ_L (2×2) con Bloch (qx,qy,qz) incrustada en los 4 estados de Fock {|00>,|01>,|10>,|11>}: |0_L>=|10>, |1_L>=|01>."""
    rho_l = 0.5 * jnp.array([[1 + qz, qx - 1j * qy], [qx + 1j * qy, 1 - qz]], dtype=jnp.complex128)
    E = jnp.zeros((4, 2), dtype=jnp.complex128).at[2, 0].set(1.0).at[1, 1].set(1.0)   # |10>←|0_L>, |01>←|1_L>
    return E @ rho_l @ jnp.conj(E).T


def apply_loss(rho4: jnp.ndarray, eta: float) -> jnp.ndarray:
    K0, K1 = _kraus_loss(eta)
    out = jnp.zeros_like(rho4)
    for Ka in (K0, K1):
        for Kb in (K0, K1):
            K = jnp.kron(Ka, Kb)
            out = out + K @ rho4 @ jnp.conj(K).T
    return out


@jax.jit
def readout_unfiltered(rho4: jnp.ndarray) -> jnp.ndarray:
    """Detector sin anuncio: 'click en a' → 0_L, cualquier otra cosa (incl. vacío) → se cuenta como 0_L si no hay click en b.
    qg_Z' = P(no click en b) − P(click en b)."""
    p = jnp.real(jnp.diag(rho4))            # |00>,|01>,|10>,|11>
    p_click_b = p[1] + p[3]
    return 1.0 - 2.0 * p_click_b


@jax.jit
def readout_sector_filtered(rho4: jnp.ndarray):
    """Postselección en peso de Hamming 1: devuelve (qg_Z filtrado, tasa de aceptación)."""
    p = jnp.real(jnp.diag(rho4))
    acc = p[1] + p[2]
    return (p[2] - p[1]) / acc, acc


def shots_multiplier_two_photons(L_km: float, alpha_db_km: float = 0.2) -> float:
    """Par de Bell con un fotón por brazo de longitud L/2: tasa de anuncio η(L/2)² = η(L) ⇒ disparos × 1/η(L)."""
    return float(1.0 / transmission(L_km, alpha_db_km))
