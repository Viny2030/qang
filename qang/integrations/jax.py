"""
qang.integrations.jax — qang readings and the qg filter as JAX functions (book, chapters 20 and 23).

Every function here is written with ``jax.numpy``, so it can be ``jit``-compiled, differentiated with ``grad`` and
batched with ``vmap``:

  * ``qg_from_p0``, ``theta_from_qg``, ``milliqang``   the Chapter 1 readings;
  * ``qg_z_from_probs``                               qg_Z of each qubit from a probability vector;
  * ``filtered_qg_z``                                 the filtered readout <P_w Z_q> / <P_w> and the kept fraction;
  * ``pole_damped_update``                            the pole-damped step of chapter 23, step x max(|sin theta|, eps).

The angular qga scale of chapter 1 (1000 qga per turn): ``radians_from_qga``, ``qga_from_radians``, ``p0_from_qga``,
``milliqang_from_qga`` and ``qga_from_milliqang``.

The derived quantities of chapter 1 ("Magnitudes derivadas"): ``quantum_snr`` (qg_SNR), ``information_curvature``
(qg_kappa), ``dissipative_deficit`` (qg_tau), ``geodesic_distance`` (qg_L) and ``statistical_angle`` (Wootters'
distance for any number of outcomes).

The qg algebra of chapters 2 and 3, as JAX arrays: ``u3_qang``, ``transition_matrix``, ``householder_qg``,
``grover_diffuser_qg``, ``transition_probability``, ``chebyshev_trajectory`` and ``grover_fisher_information``.

``curvature_damped_update`` is the step the treatise proposes with qg_kappa: on the coordinate qg_Z,
qg <- qg - lr * grad / (ln 2 * qg_kappa) = qg - lr * (1 - qg^2) * grad, a natural-gradient step with the Fisher
information of the Z measurement. To first order it is plain gradient descent on the qang angle theta = arccos qg_Z,
because (dqg/dtheta)^2 = 1 - qg^2 (book, chapter 1; ``tools/bench_jax_curvature_step.py`` measures it).

Every function matches its NumPy counterpart in qang.probability, qang.algebra and qang.qga (tests in
``tests/test_integrations.py``). Qubit 0 is the most significant bit of the outcome index, as in ``qml.probs()`` and
qang.qml. These functions take the probabilities a PennyLane QNode returns with ``interface="jax"``, so a filtered loss
can be trained end to end with JAX (the open item of chapter 20).
"""

from __future__ import annotations

import math

from . import require

jax = require("jax", "jax")
jnp = jax.numpy

__all__ = [
    "qg_from_p0", "theta_from_qg", "milliqang", "qg_z_from_probs", "filtered_qg_z", "pole_damped_update",
    "radians_from_qga", "qga_from_radians", "p0_from_qga", "milliqang_from_qga", "qga_from_milliqang",
    "quantum_snr", "information_curvature", "dissipative_deficit", "geodesic_distance", "statistical_angle",
    "u3_qang", "transition_matrix", "householder_qg", "grover_diffuser_qg", "transition_probability",
    "chebyshev_trajectory", "grover_fisher_information", "curvature_damped_update",
]

_QGA_PER_RAD = 1000.0 / (2.0 * math.pi)
_LN2 = math.log(2.0)


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


# ---------------------------------------------------------------- the angular qga scale (chapter 1)

def radians_from_qga(theta_qga):
    """theta [rad] = 2 pi theta_qga / 1000."""
    return theta_qga / _QGA_PER_RAD


def qga_from_radians(theta):
    """theta_qga = 1000 theta / (2 pi)."""
    return theta * _QGA_PER_RAD


def p0_from_qga(theta_qga):
    """p0 = cos^2(pi theta_qga / 1000), the Bloch polar angle given in qga."""
    return jnp.cos(radians_from_qga(theta_qga) / 2.0) ** 2


def milliqang_from_qga(theta_qga):
    """m = 1000 cos(2 pi theta_qga / 1000) m-qg."""
    return 1000.0 * jnp.cos(radians_from_qga(theta_qga))


def qga_from_milliqang(m):
    """Polar angle in qga, in [0, 500], of the reading m (m-qg)."""
    return qga_from_radians(jnp.arccos(m / 1000.0))


# ---------------------------------------------------------------- derived quantities (chapter 1)

def quantum_snr(qg):
    """qg_SNR = |qg_Z| / sqrt(1 - qg_Z^2) = |cot theta|; inf at a pole."""
    return jnp.abs(qg) / jnp.sqrt(1.0 - qg * qg)


def information_curvature(qg):
    """qg_kappa = -d^2 qg_S / d qg_Z^2 = 1 / (ln 2 (1 - qg_Z^2)): the Fisher information about qg_Z, in bits."""
    return 1.0 / (_LN2 * (1.0 - qg * qg))


def dissipative_deficit(r):
    """qg_tau = sqrt(1 - |r|^2) of the qg vector r = (qg_X, qg_Y, qg_Z) on the last axis."""
    return jnp.sqrt(jnp.clip(1.0 - jnp.sum(r * r, axis=-1), 0.0, None))


def geodesic_distance(q1, q2):
    """qg_L = |arccos q1 - arccos q2| / 2: Fubini-Study distance on a common meridian, Bhattacharyya angle in Z."""
    return jnp.abs(jnp.arccos(q1) - jnp.arccos(q2)) / 2.0


def statistical_angle(p, q):
    """Wootters' distance arccos(sum_k sqrt(p_k q_k)) between distributions on the last axis."""
    return jnp.arccos(jnp.clip(jnp.sum(jnp.sqrt(p * q), axis=-1), -1.0, 1.0))


# ---------------------------------------------------------------- qg algebra (chapters 2 and 3)

def u3_qang(q, phi=0.0, lam=0.0):
    """Universal single-qubit gate in qg_Z: Qiskit's U(arccos q, phi, lam) without trigonometry in q."""
    c, s = jnp.sqrt((1.0 + q) / 2.0), jnp.sqrt((1.0 - q) / 2.0)
    ep, el = jnp.exp(1j * phi), jnp.exp(1j * lam)
    return jnp.array([[c, -el * s], [ep * s, ep * el * c]])


def transition_matrix(q):
    """|U_ij|^2 = [[(1+q)/2, (1-q)/2], [(1-q)/2, (1+q)/2]]."""
    a, b = (1.0 + q) / 2.0, (1.0 - q) / 2.0
    return jnp.array([[a, b], [b, a]])


_X = jnp.array([[0.0, 1.0], [1.0, 0.0]])
_Z = jnp.array([[1.0, 0.0], [0.0, -1.0]])


def householder_qg(q):
    """I - 2|psi(q)><psi(q)| = -q Z - sqrt(1 - q^2) X."""
    return -q * _Z - jnp.sqrt(jnp.clip(1.0 - q * q, 0.0, None)) * _X


def grover_diffuser_qg(q):
    """2|psi(q)><psi(q)| - I = q Z + sqrt(1 - q^2) X."""
    return -householder_qg(q)


def transition_probability(q1, q2, phi1=0.0, phi2=0.0):
    """|<psi1|psi2>|^2 = (1 + r1 . r2) / 2 for pure states with polar biases q1, q2 and azimuths phi1, phi2."""
    t1, t2 = jnp.sqrt(jnp.clip(1.0 - q1 * q1, 0.0, None)), jnp.sqrt(jnp.clip(1.0 - q2 * q2, 0.0, None))
    return (1.0 + t1 * t2 * jnp.cos(phi1 - phi2) + q1 * q2) / 2.0


def chebyshev_trajectory(qg0, k):
    """Grover flag bias after k iterations: T_{2k+1}(qg0) = cos((2k+1) arccos qg0); k may be an array."""
    return jnp.cos((2 * k + 1) * jnp.arccos(qg0))


def grover_fisher_information(qg0, depth_m):
    """Fisher information about qg0 of one shot at depth m = 2k + 1: m^2 / (1 - qg0^2)."""
    return depth_m * depth_m / (1.0 - qg0 * qg0)


def curvature_damped_update(qg, grad, lr, bound=1.0 - 1e-9):
    """qg - lr * grad / (ln 2 qg_kappa) = qg - lr (1 - qg^2) grad, clipped to [-bound, bound]: the natural-gradient
    step on qg_Z with the Fisher information of the Z measurement (the treatise's use of qg_kappa)."""
    step = lr * grad / (_LN2 * information_curvature(qg))
    return jnp.clip(qg - step, -bound, bound)
