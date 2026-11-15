"""qang.qan — QAN / MI / IQAN en coordenadas qga con backend JAX (jax<0.9).

Escalas: ``qga`` (1 vuelta = 1000 qga, theta_rad = 2*pi*qga/1000);
lectura ``qg_Z = <Z> in [-1,1]``; ``m-qg = 1000*qg_Z``.
"""
from __future__ import annotations

from functools import partial
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

QGA_PER_TURN = 1000.0
RAD_PER_QGA = 2.0 * jnp.pi / QGA_PER_TURN


# ---------------------------------------------------------------- escalas
def qga_to_rad(qga: jnp.ndarray) -> jnp.ndarray:
    """theta [rad] = 2*pi*qga/1000 (exact)."""
    return qga * RAD_PER_QGA


def _h2(p: jnp.ndarray) -> jnp.ndarray:
    """Entropía binaria en bits con 0*log0 = 0 (qg_S)."""
    p = jnp.clip(p, 1e-300, 1.0)
    q = jnp.clip(1.0 - p, 1e-300, 1.0)
    return -(p * jnp.log2(p) + q * jnp.log2(q))


# ---------------------------------------------------------------- MI (exact)
@jax.jit
def joint_probs_qg(qz_a: jnp.ndarray, qz_b: jnp.ndarray, qzz: jnp.ndarray) -> jnp.ndarray:
    """p(s,t) = (1 + s*qz_a + t*qz_b + s*t*qzz)/4, s,t in {+1,-1} (orden 00,01,10,11).

    Exact: inversión de la expansión de Pauli-Z de la distribución marginal en Z.
    """
    s = jnp.array([1.0, 1.0, -1.0, -1.0])
    t = jnp.array([1.0, -1.0, 1.0, -1.0])
    return (1.0 + s * qz_a + t * qz_b + s * t * qzz) / 4.0


@jax.jit
def mutual_information_qg(qz_a: jnp.ndarray, qz_b: jnp.ndarray, qzz: jnp.ndarray) -> jnp.ndarray:
    """I(Z_a;Z_b) [bits] = qg_S(a) + qg_S(b) - H(p_ab).

    Depende sólo de la base Z: NO es medida de entrelazamiento (THEORY_CORE §2.4).
    """
    p = joint_probs_qg(qz_a, qz_b, qzz)
    p = jnp.clip(p, 1e-300, 1.0)
    h_ab = -jnp.sum(p * jnp.log2(p))
    return _h2((1.0 + qz_a) / 2.0) + _h2((1.0 + qz_b) / 2.0) - h_ab


@jax.jit
def mi_shot_variance_qg(qz_a: jnp.ndarray, qz_b: jnp.ndarray, qzz: jnp.ndarray) -> jnp.ndarray:
    """Varianza asintótica (método delta) de Î por disparo: N*Var(Î).

    Σ = (diag(p) - p pᵀ) es la covarianza multinomial de un disparo; g = ∂I/∂p.
    Devuelve gᵀ Σ g  (multiplicar por 1/N para N disparos).
    """
    def mi_of_p(p):
        p = jnp.clip(p, 1e-300, 1.0)
        pa = p[0] + p[1]
        pb = p[0] + p[2]
        return _h2(pa) + _h2(pb) + jnp.sum(p * jnp.log2(p))

    p = joint_probs_qg(qz_a, qz_b, qzz)
    g = jax.grad(mi_of_p)(p)
    sigma = jnp.diag(p) - jnp.outer(p, p)
    return g @ sigma @ g


# ---------------------------------------------------------------- QAN (statevector)
def _ry(theta):
    c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.complex128)


def _apply_1q(psi, u, q, n):
    psi = psi.reshape([2] * n)
    psi = jnp.moveaxis(jnp.tensordot(u, psi, axes=([1], [q])), 0, q)
    return psi.reshape(-1)


def _apply_rzz(psi, theta, q1, q2, n):
    idx = jnp.arange(2 ** n)
    z1 = 1.0 - 2.0 * ((idx >> (n - 1 - q1)) & 1)
    z2 = 1.0 - 2.0 * ((idx >> (n - 1 - q2)) & 1)
    return psi * jnp.exp(-0.5j * theta * z1 * z2)


@partial(jax.jit, static_argnames=("n", "pairs"))
def qan_state(alpha_qga: jnp.ndarray, beta_qga: jnp.ndarray, gamma_qga: jnp.ndarray,
              n: int, pairs: Tuple[Tuple[int, int], ...]) -> jnp.ndarray:
    """Capa QAN: RY(alpha_q) ⊗ ... → ∏ RZZ(beta_pq) → RY(gamma_q). Ángulos en qga.

    RZZ tiene período 500 qga (THEORY_CORE §1): beta y beta+500 producen la misma lectura.
    """
    psi = jnp.zeros(2 ** n, dtype=jnp.complex128).at[0].set(1.0)
    for q in range(n):
        psi = _apply_1q(psi, _ry(qga_to_rad(alpha_qga[q])), q, n)
    for k, (a, b) in enumerate(pairs):
        psi = _apply_rzz(psi, qga_to_rad(beta_qga[k]), a, b, n)
    for q in range(n):
        psi = _apply_1q(psi, _ry(qga_to_rad(gamma_qga[q])), q, n)
    return psi


@partial(jax.jit, static_argnames=("n",))
def qan_readout(psi: jnp.ndarray, n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Lectura proyectiva Z: qg_Z[q] y qg_ZZ[q,r] a partir de |psi|^2 (una sola base)."""
    p = jnp.abs(psi) ** 2
    idx = jnp.arange(2 ** n)
    z = jnp.stack([1.0 - 2.0 * ((idx >> (n - 1 - q)) & 1) for q in range(n)])  # (n, 2^n)
    qz = z @ p
    qzz = (z * p) @ z.T
    return qz, qzz


@partial(jax.jit, static_argnames=("n",))
def mi_matrix_qg(qz: jnp.ndarray, qzz: jnp.ndarray, n: int) -> jnp.ndarray:
    """Matriz I(Z_q;Z_r) [bits] para todos los pares, desde una lectura Z única."""
    f = jax.vmap(jax.vmap(mutual_information_qg, in_axes=(None, 0, 0)), in_axes=(0, None, 0))
    return f(qz, qz, qzz)


# ---------------------------------------------------------------- IQAN (inverse, regularizado)
def iqan_loss(params: jnp.ndarray, target_qz: jnp.ndarray, target_mi: jnp.ndarray,
              n: int, pairs: Tuple[Tuple[int, int], ...], lam: float = 1.0, mu: float = 1e-3) -> jnp.ndarray:
    """L(α,β,γ) = ||qg_Z − qg_Z*||² + λ Σ_pairs (I − I*)² + μ ||β||²/500².

    Gradiente euclidiano en ángulo (qga): equivale al paso amortiguado por qg_κ en qg_Z
    (THEORY_CORE §2.1); μ penaliza área de pulso RZZ (compilación de mínima área).
    """
    alpha, beta, gamma = params[:n], params[n:n + len(pairs)], params[n + len(pairs):]
    psi = qan_state(alpha, beta, gamma, n, pairs)
    qz, qzz = qan_readout(psi, n)
    loss = jnp.sum((qz - target_qz) ** 2)
    for k, (a, b) in enumerate(pairs):
        loss += lam * (mutual_information_qg(qz[a], qz[b], qzz[a, b]) - target_mi[k]) ** 2
    loss += mu * jnp.sum((beta / 500.0) ** 2)
    return loss


def iqan_fit(target_qz, target_mi, n, pairs, steps=300, lr=40.0, seed=0, lam=1.0, mu=1e-3):
    """Descenso por gradiente en qga (lr en qga²/unidad de pérdida). Devuelve (params, loss)."""
    pairs = tuple(pairs)
    key = jax.random.PRNGKey(seed)
    params = jax.random.uniform(key, (2 * n + len(pairs),), minval=0.0, maxval=500.0)
    vg = jax.jit(jax.value_and_grad(iqan_loss), static_argnames=("n", "pairs"))
    for _ in range(steps):
        loss, g = vg(params, target_qz, target_mi, n, pairs, lam, mu)
        params = params - lr * g
    return params, float(loss)
