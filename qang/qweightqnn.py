"""qang.qweightqnn — QNN que conserva el peso de Hamming (compuertas RBS/Givens en qga), cargador unario,
filtro de simetría bajo relajación T1 y QSP de un qubit (polinomios de Chebyshev en qg). Backend JAX (jax<0.9).
Alineado con `qang.qml` (WeightQNN, MultiClassQNN) y `qang.qsp`.
"""
from __future__ import annotations

from functools import partial
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
RAD_PER_QGA = 2.0 * jnp.pi / 1000.0


# ---------------------------------------------------------------- RBS (conserva el peso)
def _rbs(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]], dtype=jnp.complex128)


def _apply2(psi, U4, i, j, n):
    T = psi.reshape([2] * n)
    T = jnp.moveaxis(T, (i, j), (0, 1))
    T = (U4 @ T.reshape(4, -1)).reshape([2] * n)
    return jnp.moveaxis(T, (0, 1), (i, j)).reshape(-1)


def unary_loader(x: jnp.ndarray, n: int) -> jnp.ndarray:
    """|x> = Σ_k x_k |e_k> (x unitario, peso 1) con n−1 compuertas RBS desde |e_0> (exact)."""
    psi = jnp.zeros(2 ** n, dtype=jnp.complex128).at[1 << (n - 1)].set(1.0)   # |e_0> = |100…0>
    rem = 1.0
    for k in range(n - 1):
        # ángulo tal que amplitud en e_k sea x_k del resto
        if k < n - 2:
            ang = jnp.arctan2(jnp.sqrt(jnp.clip(rem - x[k] ** 2, 0.0)), x[k])
        else:                                   # último par: conserva el signo de x_{n−1}
            ang = jnp.arctan2(x[k + 1], x[k])
        psi = _apply2(psi, _rbs(ang), k, k + 1, n)
        rem = rem - x[k] ** 2
    return psi


@partial(jax.jit, static_argnames=("n", "pairs"))
def weight_qnn(theta_qga: jnp.ndarray, psi_in: jnp.ndarray, n: int, pairs: Tuple[Tuple[int, int], ...]) -> jnp.ndarray:
    """Capas de RBS(θ_k) sobre `pairs` (θ en qga). Conserva el peso de Hamming."""
    psi = psi_in
    for k, (i, j) in enumerate(pairs):
        psi = _apply2(psi, _rbs(theta_qga[k] * RAD_PER_QGA), i, j, n)
    return psi


def weight_mask(n: int, w: int) -> jnp.ndarray:
    idx = jnp.arange(2 ** n)
    return (jnp.array([bin(int(i)).count("1") for i in range(2 ** n)]) == w)


def sector_readout(p: jnp.ndarray, n: int, group_a: Sequence[int]):
    """y = Σ_{k∈A} p(e_k) − Σ_{k∉A} p(e_k) renormalizado al sector de peso 1 (filtro de simetría)."""
    e = jnp.array([1 << (n - 1 - k) for k in range(n)])
    pe = p[e]
    acc = jnp.sum(pe)
    sign = jnp.array([1.0 if k in group_a else -1.0 for k in range(n)])
    return jnp.sum(sign * pe) / acc, acc


def readout_unfiltered(p: jnp.ndarray, n: int, group_a: Sequence[int]):
    e = jnp.array([1 << (n - 1 - k) for k in range(n)])
    sign = jnp.array([1.0 if k in group_a else -1.0 for k in range(n)])
    return jnp.sum(sign * p[e])


def amplitude_damping_probs(p: jnp.ndarray, n: int, gammas: Sequence[float]) -> jnp.ndarray:
    """Relajación T1 clásica sobre el histograma: cada bit 1 decae a 0 con prob. γ_k (modelo diagonal)."""
    out = p
    for k, g in enumerate(gammas):
        bit = 1 << (n - 1 - k)
        idx = jnp.arange(2 ** n)
        has = ((idx & bit) > 0)
        moved = jnp.where(has, out * g, 0.0)
        out = out - moved
        out = out.at[idx & ~bit].add(moved)      # suma la masa decaída al string con el bit apagado
    return out


# ---------------------------------------------------------------- QSP de un qubit
@jax.jit
def qsp_readout(phis_qga: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """QSP: U = R_Z(φ_0) Π_k [W(a) R_Z(φ_k)], W(a) = R_X con cos(θ/2) = a. Devuelve qg_Z = <0|U†ZU|0>.
    Con φ ≡ 0: qg_Z = cos(dθ) = T_d(2a²−1)  (Chebyshev; exact)."""
    th = 2.0 * jnp.arccos(jnp.clip(a, -1.0, 1.0))
    c, s = jnp.cos(th / 2), jnp.sin(th / 2)
    W = jnp.array([[c, -1j * s], [-1j * s, c]], dtype=jnp.complex128)
    def rz(phi):
        return jnp.diag(jnp.exp(jnp.array([-0.5j, 0.5j]) * phi * RAD_PER_QGA))
    U = rz(phis_qga[0])
    for k in range(1, phis_qga.shape[0]):
        U = U @ W @ rz(phis_qga[k])
    psi = U[:, 0]
    return jnp.real(jnp.abs(psi[0]) ** 2 - jnp.abs(psi[1]) ** 2)
