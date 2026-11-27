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
    """Con qg_Z verdadero = 0: qmax ≈ z/√N y qmax = 2·2^{−h} − 1, por tanto N = (z/qmax)²."""
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


def toeplitz_matrix(n_input: int, n_output: int, seed: int = 0) -> np.ndarray:
    """Matriz de Toeplitz binaria T[i,j] = t[i-j+n_input-1] de forma (n_output, n_input)."""
    n, m = int(n_input), int(n_output)
    if n <= 0 or m <= 0:
        raise ValueError("n_input and n_output must be positive.")
    t = np.random.default_rng(seed).integers(0, 2, size=n + m - 1).astype(bool)
    i = np.arange(m)[:, None]
    j = np.arange(n)[None, :]
    return t[i - j + n - 1]


def toeplitz_extract(bits: np.ndarray, n_output: int, seed: int = 0) -> np.ndarray:
    """Extracción por hashing universal de Toeplitz: y = T x (mod 2), T sembrada por `seed`.

    Es exacta como familia de hash 2-universal; no acredita por sí sola la calidad criptográfica
    de la fuente (NIST SP 800-90B / IR 8446). Exige muestreo i.i.d., ya auditado por §21.2.
    """
    x = np.asarray(bits, dtype=bool).ravel()
    T = toeplitz_matrix(len(x), int(n_output), seed)
    return np.bitwise_xor.reduce(T & x[None, :], axis=1).astype(np.uint8)


def bell_chsh(correlators) -> dict:
    """CHSH: S = E00 + E01 + E10 - E11, cota local |S| ≤ 2 y cota de Tsirelson 2√2.

    Sólo evalúa correlaciones ya medidas: no certifica por sí mismo aleatoriedad
    independiente del dispositivo (requiere separación espacial, elección libre de
    ajustes y sin sesgo de muestreo; ver §41–42 y RESEARCH_NOTES §41).
    """
    e = np.asarray(correlators, dtype=float)
    if e.shape != (4,):
        raise ValueError("bell_chsh needs four correlators (E00, E01, E10, E11).")
    s = float(e[0] + e[1] + e[2] - e[3])
    return {"S": s, "local_bound": 2.0, "tsirelson": float(2 * np.sqrt(2)),
            "violates_local": bool(abs(s) > 2.0)}
