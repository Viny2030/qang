"""
qang.integrations.jax — qang readings and the qg filter as JAX functions (book, chapters 20 and 23).

Every function here is written with ``jax.numpy``, so it can be ``jit``-compiled, differentiated with ``grad`` and
batched with ``vmap``:

  * ``qg_from_p0``, ``theta_from_qg``, ``milliqang``   the Chapter 1 readings;
  * ``qg_z_from_probs``                               qg_Z of each qubit from a probability vector;
  * ``filtered_qg_z``                                 the filtered readout <P_w Z_q> / <P_w> and the kept fraction;
  * ``pole_damped_update``                            the pole-damped step of chapter 23, step x max(|sin theta|, eps).

Qubit 0 is the most significant bit of the outcome index, as in ``qml.probs()`` and qang.qml. These functions take
the probabilities a PennyLane QNode returns with ``interface="jax"``, so a filtered loss can be trained end to end
with JAX (the open item of chapter 20).
"""

from __future__ import annotations

from . import require

jax = require("jax", "jax")
jnp = jax.numpy

__all__ = ["qg_from_p0", "theta_from_qg", "milliqang", "qg_z_from_probs", "filtered_qg_z", "pole_damped_update"]


def qg_from_p0(p0):
    """qg_Z = 2 p0 - 1."""
    return 2.0 * p0 - 1.0


def theta_from_qg(qg):
    """qang angle theta = arccos(qg_Z) (singular derivative at the poles |qg| = 1)."""
    return jnp.arccos(qg)


def milliqang(qg):
    """m = 1000 qg."""
    return 1000.0 * qg


def _bits(n):
    idx = jnp.arange(2**n)
    return ((idx[:, None] >> jnp.arange(n)[::-1]) & 1).astype(jnp.float32)


def qg_z_from_probs(probs, n: int):
    """qg_Z of each qubit: 1 - 2 P(qubit q = 1). ``probs`` may carry a leading batch axis."""
    return 1.0 - 2.0 * probs @ _bits(n).astype(probs.dtype)


def filtered_qg_z(probs, n: int, weight: int):
    """(filtered qg_Z per qubit, kept fraction) after post-selection on Hamming weight ``weight``."""
    b = _bits(n).astype(probs.dtype)
    mask = (b.sum(axis=1) == weight).astype(probs.dtype)
    kept = probs @ mask
    ones = (probs * mask) @ b
    return 1.0 - 2.0 * ones / kept[..., None], kept


def pole_damped_update(theta, grad, lr, eps=0.05):
    """theta - lr * max(|sin theta|, eps) * grad: the pole-damped step of chapter 23 (a trust region near the poles)."""
    return theta - lr * jnp.maximum(jnp.abs(jnp.sin(theta)), eps) * grad
