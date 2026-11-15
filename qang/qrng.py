"""qang.qrng — Números aleatorios certificados y auditoría de canal en m-qg: entropía mínima con
intervalo de confianza (posterior Bayes-Haar), presupuesto y detección de deriva por bloques. (§31, §41, §42)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def posterior_qz_quantiles(k0: int, N: int, q: float = 0.999):
    """Cuantiles [1−q, q] de qg_Z bajo posterior Beta(1+k0, 1+N−k0) (previa de Haar). NumPy (scipy opcional)."""
    try:
        from scipy.stats import beta
        lo, hi = beta.ppf(1 - q, 1 + k0, 1 + N - k0), beta.ppf(q, 1 + k0, 1 + N - k0)
    except Exception:  # aproximación normal del posterior
        a, b = 1 + k0, 1 + N - k0
        m, s = a / (a + b), np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        z = 3.09 if q >= 0.999 else 1.96
        lo, hi = m - z * s, m + z * s
    return 2 * lo - 1, 2 * hi - 1


def certified_min_entropy(k0: int, N: int, q: float = 0.999) -> float:
    """H_min^cert = −log2((1 + max|qg_Z| sobre el intervalo creíble)/2)  (cota inferior certificada)."""
    lo, hi = posterior_qz_quantiles(k0, N, q)
    qmax = max(abs(lo), abs(hi))
    return float(-np.log2((1 + qmax) / 2))


def point_min_entropy(k0: int, N: int) -> float:
    qz = 2 * k0 / N - 1
    return float(-np.log2((1 + abs(qz)) / 2))


def shots_for_certified_hmin(h_target: float, z: float = 3.09) -> float:
    """Con qg_Z verdadero = 0: qmax ≈ z/√N ⇒ N = (z / (2(2^{1−h}) − 1))²  … resuelto por qmax = 2·2^{−h} − 1."""
    qmax = 2 * 2 ** (-h_target) - 1
    return (z / qmax) ** 2


def block_drift_chi2(bits: np.ndarray, block: int):
    """χ² de homogeneidad de p0 entre bloques: detecta deriva del canal (qg_Z no estacionario)."""
    nb = len(bits) // block
    k = np.array([bits[i * block:(i + 1) * block].sum() for i in range(nb)])
    p = k.sum() / (nb * block)
    exp = block * p
    chi2 = float(np.sum((k - exp) ** 2 / (exp * (1 - p))))
    return chi2, nb - 1, 1000 * (1 - 2 * k / block)   # estadístico, g.d.l., qg_Z por bloque en m-qg
