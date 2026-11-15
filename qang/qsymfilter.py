"""qang.qsymfilter — Filtro de simetría (sector de peso de Hamming) con matrices densidad: exactitud bajo
relajación T1 uniforme (F5, cocientes T1), fallo bajo desfase y rescate parcial por ZNE (F6).
Backend JAX (jax<0.9). Alineado con `qang.qml` / `qang.sectors` (Cap. 21 antiguo).
"""
from __future__ import annotations

from functools import partial
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from qang.qweightqnn import _rbs, RAD_PER_QGA

jax.config.update("jax_enable_x64", True)


def _op_on(U2: jnp.ndarray, q: int, n: int) -> jnp.ndarray:
    ops = [jnp.eye(2, dtype=jnp.complex128)] * n
    ops[q] = U2
    out = ops[0]
    for o in ops[1:]:
        out = jnp.kron(out, o)
    return out


def _op2_on(U4: jnp.ndarray, i: int, j: int, n: int) -> jnp.ndarray:
    """Matriz 2^n×2^n de U4 sobre (i,j) vía permutación de ejes."""
    d = 2 ** n
    U = U4.reshape(2, 2, 2, 2)
    full = jnp.eye(d, dtype=jnp.complex128).reshape([2] * (2 * n))
    # construir aplicando a una base: más simple vía acción sobre vectores
    def act(v):
        T = v.reshape([2] * n)
        T = jnp.moveaxis(T, (i, j), (0, 1))
        T = (U4 @ T.reshape(4, -1)).reshape([2] * n)
        return jnp.moveaxis(T, (0, 1), (i, j)).reshape(-1)
    return jax.vmap(act, in_axes=1, out_axes=1)(jnp.eye(d, dtype=jnp.complex128))


def kraus_amplitude_damping(g):
    return [jnp.array([[1, 0], [0, jnp.sqrt(1 - g)]], dtype=jnp.complex128),
            jnp.array([[0, jnp.sqrt(g)], [0, 0]], dtype=jnp.complex128)]


def kraus_dephasing(p):
    Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
    return [jnp.sqrt(1 - p) * jnp.eye(2, dtype=jnp.complex128), jnp.sqrt(p) * Z]


def apply_local_channel(rho, kraus, n):
    for q in range(n):
        new = jnp.zeros_like(rho)
        for K in kraus:
            Kf = _op_on(K, q, n)
            new = new + Kf @ rho @ jnp.conj(Kf).T
        rho = new
    return rho


def apply_local_channel_per_qubit(rho, kraus_list, n):
    for q, kraus in enumerate(kraus_list):
        new = jnp.zeros_like(rho)
        for K in kraus:
            Kf = _op_on(K, q, n)
            new = new + Kf @ rho @ jnp.conj(Kf).T
        rho = new
    return rho


def run_weight_circuit(rho0, theta_qga: Sequence[float], pairs, n, noise_after_layer=None):
    """Aplica RBS(θ_k) en orden; tras cada compuerta aplica `noise_after_layer(rho)` si se da."""
    rho = rho0
    for k, (i, j) in enumerate(pairs):
        U = _op2_on(_rbs(theta_qga[k] * RAD_PER_QGA), i, j, n)
        rho = U @ rho @ jnp.conj(U).T
        if noise_after_layer is not None:
            rho = noise_after_layer(rho)
    return rho


def sector_indices(n, w=1):
    return np.array([i for i in range(2 ** n) if bin(i).count("1") == w])


def filtered_sector_state(rho, n, w=1):
    """ρ restringida al sector de peso w y renormalizada; devuelve (ρ_S, tasa de aceptación)."""
    idx = sector_indices(n, w)
    sub = rho[np.ix_(idx, idx)]
    acc = jnp.real(jnp.trace(sub))
    return sub / acc, acc


def sector_observable(rho_S, signs):
    """<Σ_k s_k |e_k><e_k|> + coherencias no entran: observable diagonal en el sector."""
    return jnp.real(jnp.sum(jnp.diag(rho_S) * jnp.asarray(signs)))


def richardson_zne(lams, vals):
    """Extrapolación de Richardson a λ=0 (polinomio de grado len−1)."""
    lams = np.asarray(lams, float); vals = np.asarray(vals, float)
    coef = np.polyfit(lams, vals, len(lams) - 1)
    return float(np.polyval(coef, 0.0))
