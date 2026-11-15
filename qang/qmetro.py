"""qang.qmetro — Metrología en qga: punto de operación Ramsey, GHZ bajo desfase markoviano,
termometría tanh y comparación de relojes (corrimiento gravitacional). Backend JAX (jax<0.9).
Alineado con `qang.fisher` / `qang.clock_comparison` (core NumPy).

Lectura Ramsey: qg_Z = C cos φ  (C = contraste, φ = fase acumulada). Fisher por disparo en φ:
I_F(φ) = C² sin²φ / (1 − C² cos²φ); máximo C² en φ = π/2 (qg_Z = 0, ecuador = 250 qga).
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
RAD_PER_QGA = 2.0 * jnp.pi / 1000.0


@jax.jit
def ramsey_fisher_phi(phi: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
    """I_F(φ) por disparo con lectura qg_Z = C cos φ (exact)."""
    return C ** 2 * jnp.sin(phi) ** 2 / (1.0 - C ** 2 * jnp.cos(phi) ** 2)


@jax.jit
def ramsey_fisher_omega(T: jnp.ndarray, T2: jnp.ndarray, phi: jnp.ndarray = jnp.pi / 2) -> jnp.ndarray:
    """I_F(ω) por disparo = T² I_F(φ; C = e^{−T/T2}); por unidad de tiempo: I_F(ω)/T."""
    C = jnp.exp(-T / T2)
    return T ** 2 * ramsey_fisher_phi(phi, C)


@jax.jit
def ghz_fisher_omega(n: jnp.ndarray, T: jnp.ndarray, T2: jnp.ndarray) -> jnp.ndarray:
    """GHZ de n qubits, lectura de paridad <X^{⊗n}> = e^{−nT/T2} cos(nφ): I_F(ω) = n² T² e^{−2nT/T2}
    en el punto óptimo (nφ = π/2). Desfase markoviano independiente por qubit (exact)."""
    return n ** 2 * T ** 2 * jnp.exp(-2.0 * n * T / T2)


@jax.jit
def product_fisher_omega(n: jnp.ndarray, T: jnp.ndarray, T2: jnp.ndarray) -> jnp.ndarray:
    """n qubits en producto, Ramsey individual: I_F(ω) = n T² e^{−2T/T2}."""
    return n * T ** 2 * jnp.exp(-2.0 * T / T2)


def optimal_T(fisher_fn, T2: float, n: float) -> Tuple[float, float]:
    """Maximiza I_F(ω)/T (Fisher por unidad de tiempo) en T por descenso en log T con jax.grad."""
    f = lambda logT: -(fisher_fn(jnp.asarray(n), jnp.exp(logT), jnp.asarray(T2)) / jnp.exp(logT))
    g = jax.jit(jax.grad(f))
    logT = jnp.log(jnp.asarray(T2) / 2.0)
    for _ in range(2000):
        logT = logT - 0.1 * g(logT)
    T = float(jnp.exp(logT))
    return T, float(-f(jnp.log(T)))


@jax.jit
def thermal_qz(beta_omega: jnp.ndarray) -> jnp.ndarray:
    """Estado de Gibbs de un qubit: qg_Z = tanh(ħω/2kT) = tanh(x/2), x = ħω/kT (§25, exact)."""
    return jnp.tanh(beta_omega / 2.0)


@jax.jit
def thermal_fisher_x(x: jnp.ndarray) -> jnp.ndarray:
    """Fisher por disparo del parámetro x = ħω/kT desde qg_Z: I_F(x) = (∂x qg_Z)²/(1 − qg_Z²) = ¼ sech²(x/2)."""
    q = thermal_qz(x)
    dq = jax.grad(thermal_qz)(x)
    return dq ** 2 / (1.0 - q ** 2)


@jax.jit
def thermal_fisher_T(x: jnp.ndarray) -> jnp.ndarray:
    """Fisher relativa de temperatura: I_F(ln T) = x² I_F(x) (óptimo en x* ≈ 2.40, qg_Z* ≈ 0.83)."""
    return x ** 2 * thermal_fisher_x(x)


def clock_redshift_phase(omega: float, T: float, dh: float, g: float = 9.80665, c: float = 299792458.0) -> float:
    """Δφ = ω T g Δh / c² [rad] entre dos relojes separados Δh (exact, aproximación de campo débil)."""
    return omega * T * g * dh / c ** 2


def shots_to_resolve_height(omega: float, T: float, dh: float, z: float = 3.0, C: float = 1.0) -> float:
    """N para resolver Δh a zσ con lectura en el ecuador: σ_φ = 1/(C√N) ⇒ N = (z/(C Δφ))²."""
    return (z / (C * clock_redshift_phase(omega, T, dh))) ** 2
