"""qang.qfisher — Información de Fisher clásica (lectura Z) y cuántica (Fubini–Study) en qga,
gradiente natural regularizado y diagnóstico de mesetas estériles. Backend JAX (jax<0.9).

Unidades: parámetros en qga; métricas en qga^-2 (factor (2π/1000)^2 respecto a rad^-2).
"""
from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

RAD_PER_QGA = 2.0 * jnp.pi / 1000.0
QFI_1Q_QGA = float(RAD_PER_QGA ** 2)        # 4π²·10⁻⁶ qga⁻² (THEORY_CORE §1)
SIGMA_1Q_QGA = float(1.0 / RAD_PER_QGA)     # 159.15 qga / √N


def fubini_study_metric(state_fn: Callable[[jnp.ndarray], jnp.ndarray], params: jnp.ndarray) -> jnp.ndarray:
    """g_ij = Re[<∂iψ|∂jψ> − <∂iψ|ψ><ψ|∂jψ>]  (QFI = 4 g para estados puros). Unidades qga⁻²."""
    psi = state_fn(params)
    J = jax.jacfwd(state_fn)(params)                      # (2^n, P) complejo
    a = jnp.conj(J).T @ J
    b = jnp.conj(J).T @ psi
    return jnp.real(a - jnp.outer(b, jnp.conj(b)))


def classical_fisher_z(state_fn: Callable[[jnp.ndarray], jnp.ndarray], params: jnp.ndarray,
                       eps: float = 1e-12) -> jnp.ndarray:
    """F_ij = Σ_x ∂i p(x) ∂j p(x) / p(x) para la lectura proyectiva Z (sólo histograma). qga⁻²."""
    p_fn = lambda t: jnp.abs(state_fn(t)) ** 2
    p = p_fn(params)
    Jp = jax.jacfwd(p_fn)(params)                          # (2^n, P)
    w = jnp.where(p > eps, 1.0 / jnp.maximum(p, eps), 0.0)
    return Jp.T @ (w[:, None] * Jp)


def natural_step(params: jnp.ndarray, grad: jnp.ndarray, metric: jnp.ndarray,
                 lr: float, lam: float) -> jnp.ndarray:
    """params − lr (metric + lam·I)⁻¹ grad. Regularización lam obligatoria (metric singular en polos)."""
    P = params.shape[0]
    return params - lr * jnp.linalg.solve(metric + lam * jnp.eye(P), grad)


def psd_gap(A: jnp.ndarray, B: jnp.ndarray) -> float:
    """min eig(A − B): ≥ 0 ⇔ A ⪰ B."""
    return float(jnp.linalg.eigvalsh(0.5 * ((A - B) + (A - B).T))[0])


def gradient_variance(loss_fn: Callable[[jnp.ndarray], jnp.ndarray], P: int, key: jax.Array,
                      samples: int = 200) -> float:
    """Var[∂L/∂θ_0] sobre parámetros uniformes en [0, 1000) qga (diagnóstico de meseta estéril)."""
    thetas = jax.random.uniform(key, (samples, P), minval=0.0, maxval=1000.0)
    g = jax.vmap(jax.grad(loss_fn))(thetas)[:, 0]
    return float(jnp.var(g))


def circuits_for_metric(P: int) -> Tuple[int, int]:
    """(circuitos QFI vía test de Hadamard con 1 ancilla, circuitos CFI_Z vía desplazamiento de parámetro).

    QFI: P(P+1)/2 solapamientos + P términos de Berry; CFI_Z: 2P circuitos, 0 ancillas (exact).
    """
    return P * (P + 1) // 2 + P, 2 * P
