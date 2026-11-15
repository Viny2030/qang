"""qang.qecho — Error de lectura como mapa afín en m-qg (efecto de arrastre, F7), calibración por eco /
matriz de transferencia, criterio de NO inversión (§108) y filtro de sector de orden e² (lectura sin arrastre).
Backend NumPy/JAX. Alineado con `qang.calibration` (Cap. 22 antiguo, Apéndice C).

Lectura con errores de asignación e01 = P(leer 1 | 0), e10 = P(leer 0 | 1):
  m' = (1 − e01 − e10) m + 1000 (e10 − e01)      (afín en m-qg; 'arrastre' hacia el punto fijo m*)
  m* = 1000 (e10 − e01)/(e01 + e10)               (lectura sin arrastre: sesgo cero)
"""
from __future__ import annotations

import numpy as np


def readout_affine(m: float, e01: float, e10: float) -> float:
    return (1 - e01 - e10) * m + 1000 * (e10 - e01)


def drag_free_point(e01: float, e10: float) -> float:
    return 1000 * (e10 - e01) / (e01 + e10)


def invert_readout(m_obs: float, e01: float, e10: float) -> float:
    """Inversión de la matriz de transferencia (eco X / identidad): amplifica la varianza por (1 − e01 − e10)^−2."""
    return (m_obs - 1000 * (e10 - e01)) / (1 - e01 - e10)


def mse_raw_vs_inverted(m: float, e01: float, e10: float, N: int, N_cal: int):
    """MSE (m-qg²) de la lectura cruda (sesgo² + varianza) frente a la invertida (varianza amplificada + calibración).
    σ_m² = 1000²(1 − qg_Z'²)/N ; calibración: var(e) ≈ e(1−e)/N_cal propagada linealmente."""
    m_obs = readout_affine(m, e01, e10)
    q = m_obs / 1000
    var_m = 1000 ** 2 * (1 - q ** 2) / N
    bias = m_obs - m
    k = 1 - e01 - e10
    # inversión: m̂ = (m_obs − 1000(e10 − e01))/k ; derivadas respecto a e01, e10
    d_e01 = (1000 + (m_obs - 1000 * (e10 - e01)) / k) / k
    d_e10 = (-1000 + (m_obs - 1000 * (e10 - e01)) / k) / k
    var_cal = d_e01 ** 2 * e01 * (1 - e01) / N_cal + d_e10 ** 2 * e10 * (1 - e10) / N_cal
    mse_raw = bias ** 2 + var_m
    mse_inv = var_m / k ** 2 + var_cal
    return mse_raw, mse_inv, bias, np.sqrt(var_m)


def sector_readout_error_order(p_sector: np.ndarray, e01: float, e10: float):
    """n strings de peso 1 con flips independientes: masa que permanece en peso 1 tras los flips.
    Un flip saca del sector (filtrado); llegar a OTRO string de peso 1 exige 2 flips (prob e01·e10):
    sesgo residual O(e01·e10). Devuelve (p filtrada, sesgo máximo, cota e01·e10)."""
    n = len(p_sector)
    stay = (1 - e10) * (1 - e01) ** (n - 1)             # ningún flip
    hop = e10 * e01 * (1 - e01) ** (n - 2)              # el 1 cae y un 0 sube: salto e_k → e_j
    p_f = np.zeros(n)
    for k in range(n):
        p_f[k] += p_sector[k] * stay
        for j in range(n):
            if j != k:
                p_f[j] += p_sector[k] * hop
    p_f /= p_f.sum()
    return p_f, float(np.max(np.abs(p_f - p_sector))), e01 * e10
