"""qang.qneutrino — Oscilación de sabor en qga: 2 sabores = 1 qubit, 3 sabores = 2 qubits (sector físico),
efectos de materia (MSW) y lectura adiabática. Backend JAX (jax<0.9). Alineado con `qang.neutrino` (core NumPy).

Fase de oscilación: φ = 2·1.267·Δm²[eV²]·L[km]/E[GeV] rad  ⇒  φ_qga = 1000 φ / 2π;
una longitud de oscilación L_osc = 4πE/Δm² equivale a 1000 qga (una vuelta).
Lectura de 2 sabores: qg_Z = 1 − 2 P(ν_α→ν_β) = 1 − 2 sin²2θ sin²(φ/2);  m-qg = 1000 qg_Z.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
RAD_PER_QGA = 2.0 * jnp.pi / 1000.0
KAPPA = 1.267  # (Δm² L / 4E) en unidades eV², km, GeV


# ---------------------------------------------------------------- 2 sabores = 1 qubit
@jax.jit
def osc_phase_rad(dm2: jnp.ndarray, L: jnp.ndarray, E: jnp.ndarray) -> jnp.ndarray:
    return 2.0 * KAPPA * dm2 * L / E


@jax.jit
def osc_phase_qga(dm2: jnp.ndarray, L: jnp.ndarray, E: jnp.ndarray) -> jnp.ndarray:
    return osc_phase_rad(dm2, L, E) / RAD_PER_QGA


@jax.jit
def p_transition(theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """P(ν_α→ν_β) = sin²2θ sin²(φ/2)."""
    return jnp.sin(2 * theta) ** 2 * jnp.sin(phi / 2) ** 2


@jax.jit
def survival_qz(theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """qg_Z = 1 − 2P (eje de sabor); circuito: RY(2θ)·RZ(φ)·RY(−2θ)|0>, RZ virtual (Cap. 2)."""
    return 1.0 - 2.0 * p_transition(theta, phi)


@jax.jit
def fisher_phi(theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """I_F(φ) por evento = (∂φ p)²/(p(1−p)): cero en los máximos de oscilación."""
    p = p_transition(theta, phi)
    dp = jax.grad(p_transition, argnums=1)(theta, phi)
    return dp ** 2 / jnp.clip(p * (1 - p), 1e-300)


@jax.jit
def fisher_theta(theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """I_F(θ) por evento: máximo en φ = π (máximo de oscilación)."""
    p = p_transition(theta, phi)
    dp = jax.grad(p_transition, argnums=0)(theta, phi)
    return dp ** 2 / jnp.clip(p * (1 - p), 1e-300)


def optimal_phase_for(fisher_fn, theta: float, lo: float = 0.05, hi: float = 2 * jnp.pi - 0.05, n: int = 4001):
    phis = jnp.linspace(lo, hi, n)
    vals = jax.vmap(lambda f: fisher_fn(jnp.asarray(theta), f))(phis)
    k = int(jnp.argmax(vals))
    return float(phis[k]), float(vals[k])


# ---------------------------------------------------------------- 3 sabores = 2 qubits (sector físico)
def pmns(t12: float, t13: float, t23: float, delta: float) -> jnp.ndarray:
    c12, s12, c13, s13, c23, s23 = (jnp.cos(t12), jnp.sin(t12), jnp.cos(t13), jnp.sin(t13), jnp.cos(t23), jnp.sin(t23))
    e = jnp.exp(1j * delta)
    return jnp.array([
        [c12 * c13, s12 * c13, s13 / e],
        [-s12 * c23 - c12 * s23 * s13 * e, c12 * c23 - s12 * s23 * s13 * e, s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * e, -c12 * s23 - s12 * c23 * s13 * e, c23 * c13],
    ])


def p3_vacuum(U: jnp.ndarray, m2: jnp.ndarray, L: float, E: float, alpha: int) -> jnp.ndarray:
    """Probabilidades P(ν_α→ν_β), β = e,μ,τ, en el vacío (exact)."""
    ph = jnp.exp(-1j * 2.0 * KAPPA * m2 * L / E)
    amp = U @ (ph * jnp.conj(U)[alpha, :])
    return jnp.abs(amp) ** 2


@jax.jit
def embed_two_qubits(p3: jnp.ndarray) -> jnp.ndarray:
    """e,μ,τ → |00>,|01>,|10>; |11> es el sector no físico (peso de Hamming 2)."""
    return jnp.array([p3[0], p3[1], p3[2], 0.0])


@jax.jit
def depolarize(p4: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    return (1.0 - eps) * p4 + eps / 4.0


@jax.jit
def sector_filter(p4: jnp.ndarray) -> jnp.ndarray:
    """Descarta |11> y renormaliza el sector físico (filtro de sector, Cap. 2 / `qang.sectors`)."""
    q = p4.at[3].set(0.0)
    return q / jnp.sum(q)


@jax.jit
def flavor_error(p_est: jnp.ndarray, p_true: jnp.ndarray) -> jnp.ndarray:
    """Error medio absoluto sobre los 3 sabores (métrica de §165)."""
    return jnp.mean(jnp.abs(p_est[:3] - p_true[:3]))


# ---------------------------------------------------------------- materia (MSW)
@jax.jit
def matter_potential_A(Ne_cm3: jnp.ndarray, E_GeV: jnp.ndarray) -> jnp.ndarray:
    """A = 2√2 G_F N_e E  [eV²];  2√2 G_F = 1.52e-4 eV² cm³ mol⁻¹ GeV⁻¹ · (N_e/N_A)."""
    return 1.52e-4 * (Ne_cm3 / 6.02214076e23) * E_GeV


@jax.jit
def sin2_2theta_m(theta: jnp.ndarray, dm2: jnp.ndarray, A: jnp.ndarray) -> jnp.ndarray:
    s, c = jnp.sin(2 * theta), jnp.cos(2 * theta)
    return s ** 2 / ((c - A / dm2) ** 2 + s ** 2)


@jax.jit
def cos2theta_m(theta: jnp.ndarray, dm2: jnp.ndarray, A: jnp.ndarray) -> jnp.ndarray:
    """qg_Z del eje efectivo de masa en materia: cos2θ_m = (cos2θ − A/Δm²)/√((cos2θ − A/Δm²)² + sin²2θ)."""
    s, c = jnp.sin(2 * theta), jnp.cos(2 * theta)
    x = c - A / dm2
    return x / jnp.sqrt(x ** 2 + s ** 2)


@jax.jit
def resonance_A(theta: jnp.ndarray, dm2: jnp.ndarray) -> jnp.ndarray:
    """A_res = Δm² cos2θ  ⇔  θ_m = π/4  ⇔  eje efectivo en el ecuador (250 qga, qg_Z = 0)."""
    return dm2 * jnp.cos(2 * theta)


@jax.jit
def adiabatic_survival_qz(theta: jnp.ndarray, dm2: jnp.ndarray, A_prod: jnp.ndarray) -> jnp.ndarray:
    """Adiabático (solar): P_ee = ½(1 + cos2θ_m cos2θ) ⇒ qg_Z^{ee} = cos2θ_m(A_prod)·cos2θ  (producto de dos qg)."""
    return cos2theta_m(theta, dm2, A_prod) * jnp.cos(2 * theta)


def trotter_survival_qz(theta: float, dm2: float, E: float, A_profile: jnp.ndarray, dL_km: float) -> jnp.ndarray:
    """Evolución numérica de 1 qubit con H(L) = (Δm²/4E)[−cos2θ + A/Δm², sin2θ; sin2θ, cos2θ − A/Δm²] por tramos.
    Estado inicial |ν_e> = |0>; devuelve qg_Z = 2P_ee − 1."""
    s, c = jnp.sin(2 * theta), jnp.cos(2 * theta)
    psi = jnp.array([1.0 + 0j, 0.0 + 0j])
    def step(psi, A):
        hz, hx = (-c + A / dm2), s
        ang = 2.0 * KAPPA * dm2 * dL_km / E * 0.5 * jnp.sqrt(hz ** 2 + hx ** 2)   # ½·(2κ Δm² dL/E)·|h|
        nz, nx = hz / jnp.sqrt(hz ** 2 + hx ** 2), hx / jnp.sqrt(hz ** 2 + hx ** 2)
        U = jnp.cos(ang) * jnp.eye(2) - 1j * jnp.sin(ang) * jnp.array([[nz, nx], [nx, -nz]])
        return U @ psi, None
    psi, _ = jax.lax.scan(step, psi, A_profile)
    return 2.0 * jnp.abs(psi[0]) ** 2 - 1.0
