"""qang.qgeom — Fases geométricas (Berry / Pancharatnam) y teorema de Stokes en la esfera, en qga.
Backend JAX (jax<0.9). Alineado con `qang.geometric` / `qang.phase` (core NumPy).

Convenciones: Bloch (qg_X, qg_Y, qg_Z); curvatura de Berry F = ½ sinθ dθ∧dφ (monopolo de carga ½);
fase de Berry γ = −Ω/2 con Ω el ángulo sólido orientado del lazo.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
RAD_PER_QGA = 2.0 * jnp.pi / 1000.0


@jax.jit
def bloch_from_state(psi: jnp.ndarray) -> jnp.ndarray:
    """(qg_X, qg_Y, qg_Z) de un estado puro de 1 qubit."""
    a, b = psi[0], psi[1]
    return jnp.real(jnp.array([2 * jnp.conj(a) * b, 2j * jnp.conj(a) * b, jnp.abs(a) ** 2 - jnp.abs(b) ** 2]))


@jax.jit
def state_from_qga(alpha_qga: jnp.ndarray, phi_qga: jnp.ndarray) -> jnp.ndarray:
    """|ψ> = cos(θ/2)|0> + e^{iφ} sin(θ/2)|1>, θ = 2π α/1000, φ = 2π φ_qga/1000."""
    th, ph = alpha_qga * RAD_PER_QGA, phi_qga * RAD_PER_QGA
    return jnp.array([jnp.cos(th / 2), jnp.exp(1j * ph) * jnp.sin(th / 2)])


@jax.jit
def berry_phase_latitude(qz: jnp.ndarray) -> jnp.ndarray:
    """Lazo a qg_Z constante (una vuelta en φ): Ω = 2π(1 − qg_Z), γ = −π(1 − qg_Z)  [rad] (exact, Stokes)."""
    return -jnp.pi * (1.0 - qz)


@jax.jit
def solid_angle_triangle(r1: jnp.ndarray, r2: jnp.ndarray, r3: jnp.ndarray) -> jnp.ndarray:
    """Ángulo sólido orientado del triángulo geodésico (Van Oosterom–Strackee), vectores unitarios."""
    num = jnp.dot(r1, jnp.cross(r3, r2))   # orientación: lazo con φ creciente ⇒ Ω > 0
    den = 1.0 + jnp.dot(r1, r2) + jnp.dot(r2, r3) + jnp.dot(r3, r1)
    return 2.0 * jnp.arctan2(num, den)


@jax.jit
def bargmann_phase(p1: jnp.ndarray, p2: jnp.ndarray, p3: jnp.ndarray) -> jnp.ndarray:
    """Fase de Pancharatnam/Berry discreta γ = arg(<ψ2|ψ1><ψ3|ψ2><ψ1|ψ3>) = −Ω/2 (invariante de gauge, exact).

    Convención: γ = i∮<ψ|dψ>; con |ψ> = cos(θ/2)|0> + e^{iφ} sin(θ/2)|1> y φ creciente, γ = −π(1−cosθ).
    """
    z = jnp.vdot(p2, p1) * jnp.vdot(p3, p2) * jnp.vdot(p1, p3)
    return jnp.angle(z)


def solid_angle_polygon(rs: jnp.ndarray) -> jnp.ndarray:
    """Ω de un polígono geodésico cerrado (abanico de triángulos desde rs[0])."""
    tot = 0.0
    for k in range(1, rs.shape[0] - 1):
        tot = tot + solid_angle_triangle(rs[0], rs[k], rs[k + 1])
    return tot


@jax.jit
def ramsey_phase_fisher_qg(qz_readout: jnp.ndarray) -> jnp.ndarray:
    """Lectura de fase por interferometría: qg_Z = cos γ ⇒ I_F(γ) = 1 por disparo (uniforme);
    varianza de γ estimada desde qg_Z: Var(γ) = (1 − qg_Z²)/(N sin²γ) = 1/N."""
    return jnp.ones_like(qz_readout)


def shots_for_berry_latitude(qz: float, sigma_gamma_rad: float) -> float:
    """γ = −π(1−qg_Z): σ_γ = π σ_{qg_Z} = π√((1−qg_Z²)/N) ⇒ N = π²(1−qg_Z²)/σ_γ²  (1 base Z, 0 ancillas)."""
    return float(jnp.pi ** 2 * (1.0 - qz ** 2) / sigma_gamma_rad ** 2)
