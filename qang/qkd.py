"""qang.qkd — BB84 en espacio qg: estados en qga, intercepta-y-reenvía, QBER como lectura m-qg,
tasa de clave y presupuesto de bits tamizados. Backend JAX (jax<0.9). (§40, §43, §44, §49)

Estados BB84 en qga: Z: {0, 500} qga polares; X: 250 qga con φ ∈ {0, 500} qga.
QBER e ⇔ qg_Z = 1 − 2e (base reconciliada). Intercepta-y-reenvía con fracción f: e = f/4 (exact).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from qang.qdemarc import h2, bb84_rate_from_qz

jax.config.update("jax_enable_x64", True)
RAD_PER_QGA = 2.0 * jnp.pi / 1000.0
I2 = jnp.eye(2, dtype=jnp.complex128)
X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)


def ket(alpha_qga: float, phi_qga: float) -> jnp.ndarray:
    th, ph = alpha_qga * RAD_PER_QGA, phi_qga * RAD_PER_QGA
    return jnp.array([jnp.cos(th / 2), jnp.exp(1j * ph) * jnp.sin(th / 2)])


BB84 = {  # (base, bit) → estado
    ("Z", 0): ket(0.0, 0.0), ("Z", 1): ket(500.0, 0.0),
    ("X", 0): ket(250.0, 0.0), ("X", 1): ket(250.0, 500.0),
}


def projector(psi):
    return jnp.outer(psi, jnp.conj(psi))


def depolarize(rho, p):
    return (1 - p) * rho + p * I2 / 2


def intercept_resend(rho, f: float):
    """Eve mide en Z o X al azar (fracción f de los pulsos) y reenvía el autoestado obtenido."""
    out = (1 - f) * rho
    for basis in ("Z", "X"):
        for b in (0, 1):
            P = projector(BB84[(basis, b)])
            out = out + (f / 2) * jnp.real(jnp.trace(P @ rho)) * P
    return out


def qber(f: float, p: float) -> float:
    """QBER medio sobre las 4 preparaciones con medida de Bob en la base correcta."""
    errs = []
    for (basis, b), psi in BB84.items():
        rho = depolarize(intercept_resend(projector(psi), f), p)
        wrong = projector(BB84[(basis, 1 - b)])
        errs.append(float(jnp.real(jnp.trace(wrong @ rho))))
    return sum(errs) / 4


def eve_info_intercept_resend(f: float) -> float:
    """Información de Eve por bit tamizado (IR): I_E = f/2 bits (acierta la base la mitad de las veces)."""
    return f / 2


def key_rate_individual(e: float) -> float:
    """r = 1 − h(e) − I_E con I_E = 2e para IR (e = f/4 ⇒ f/2 = 2e)."""
    return 1 - float(h2(e)) - 2 * e


def sifted_bits_for_qber_sigma(e: float, sigma_e: float) -> float:
    """N = e(1−e)/σ_e²; en m-qg: σ_m = 2000 σ_e."""
    return e * (1 - e) / sigma_e ** 2
