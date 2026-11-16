"""qang.qchar — Cap. 30: caracterización de hardware en m-qg: T1 por decaimiento (instante óptimo 1.59 T1), calibración de
lectura (e01, e10) con barras de error, XEB/HOP → fidelidad, desplazamiento ZZ estático en qga y cuantización de control.
NumPy exacto (sin SciPy). m-qg = 1000·qg_Z; 1 vuelta = 1000 qga.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

def _minimize_scalar(f, a: float, b: float, tol: float = 1e-12) -> float:
    """Mínimo de una función unimodal en [a, b] por sección áurea (sin SciPy: el núcleo de qang sólo exige NumPy)."""
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while abs(b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


# ------------------------------------------------------------------ T1 por decaimiento
def t1_qz(t, T1):
    """Lectura Z tras esperar t desde |1>: qg_Z = 1 − 2 e^{−t/T1}; (m-qg = 1000·qg_Z)."""
    return 1.0 - 2.0 * np.exp(-np.asarray(t, dtype=float) / T1)


def fit_t1(ts: Sequence[float], qz: Sequence[float]) -> float:
    """Ajuste log-lineal: ln((1−qg_Z)/2) = −t/T1 (exacto sin ruido)."""
    y = np.log((1.0 - np.asarray(qz)) / 2.0)
    slope = np.polyfit(np.asarray(ts, dtype=float), y, 1)[0]
    return float(-1.0 / slope)


def fisher_t1(ts: Sequence[float], T1: float, shots: int) -> float:
    """Información de Fisher sobre T1 con `shots` disparos binomiales en cada instante: Σ N (∂p/∂T1)²/(p(1−p))."""
    t = np.asarray(ts, dtype=float)
    p = np.exp(-t / T1)
    dp = p * t / T1 ** 2
    return float(np.sum(shots * dp ** 2 / (p * (1 - p))))


def optimal_delay_ratio() -> float:
    """x = t/T1 que maximiza la información por disparo: x² e^{−x}/(1−e^{−x}); x* = 1.5936."""
    f = lambda x: -(x ** 2 * np.exp(-x) / (1 - np.exp(-x)))
    return float(_minimize_scalar(f, 0.1, 6.0, 1e-10))


def mle_t1(ts: Sequence[float], ks: Sequence[int], shots: int, T1_max: float = 1e4) -> float:
    """Máxima verosimilitud binomial de T1 (válida también con instantes repetidos)."""
    t = np.asarray(ts, dtype=float)
    k = np.asarray(ks, dtype=float)

    def nll(logT):
        p = np.clip(np.exp(-t / np.exp(logT)), 1e-12, 1 - 1e-12)
        return -np.sum(k * np.log(p) + (shots - k) * np.log(1 - p))

    return float(np.exp(_minimize_scalar(nll, np.log(1e-2), np.log(T1_max), 1e-9)))


def simulate_t1_std(ts: Sequence[float], T1: float, shots: int, reps: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    t = np.asarray(ts, dtype=float)
    est = [mle_t1(t, rng.binomial(shots, np.exp(-t / T1)), shots) for _ in range(reps)]
    return float(np.std(est))


# ------------------------------------------------------------------ lectura: e01, e10
def calibrate_readout(ones_when_0: int, N0: int, zeros_when_1: int, N1: int) -> Tuple[float, float, float, float]:
    """(e01, e10, σ_e01, σ_e10) con error binomial."""
    e01, e10 = ones_when_0 / N0, zeros_when_1 / N1
    return e01, e10, float(np.sqrt(e01 * (1 - e01) / N0)), float(np.sqrt(e10 * (1 - e10) / N1))


def shots_for_error(e: float, sigma: float) -> float:
    return e * (1 - e) / sigma ** 2


def correct_mqg(m_read: float, e01: float, e10: float) -> float:
    """Inversión de m' = (1 − e01 − e10)·m + 1000(e10 − e01)."""
    return (m_read - 1000.0 * (e10 - e01)) / (1.0 - e01 - e10)


# ------------------------------------------------------------------ XEB / HOP
def haar_state(n: int, rng) -> np.ndarray:
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return v / np.linalg.norm(v)


def heavy_output_probability(p_ideal: np.ndarray, p_noisy: np.ndarray) -> float:
    heavy = p_ideal > np.median(p_ideal)
    return float(p_noisy[heavy].sum())


HOP_PORTER_THOMAS = (1.0 + np.log(2.0)) / 2.0


def fidelity_from_hop(hop: float) -> float:
    """Con ruido despolarizante global: HOP = f·HOP_PT + (1−f)/2  ⇒  f = (HOP − 1/2)/(HOP_PT − 1/2)."""
    return (hop - 0.5) / (HOP_PORTER_THOMAS - 0.5)


def mean_hop(n: int, f: float, trials: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        p = np.abs(haar_state(n, rng)) ** 2
        pn = f * p + (1 - f) / 2 ** n
        out.append(heavy_output_probability(p, pn))
    return float(np.mean(out))


# ------------------------------------------------------------------ ZZ estático
def zz_phase_qga(zeta: float, t: float) -> float:
    """Fase relativa de un qubit según el estado del vecino con H = ζ ZZ/4: φ = ζ t (rad) = 1000 ζ t/(2π) qga."""
    return 1000.0 * zeta * t / (2 * np.pi)


def zz_ramsey_qgx(zeta: float, t: float, neighbor: int, basis: str = "X") -> float:
    """qg_X (o qg_Y con basis="Y") del qubit 0 (inicio |+>) con el vecino en |neighbor> bajo H = ζ ZZ/4 (simulación exacta)."""
    Z = np.diag([1.0, -1.0])
    H = zeta / 4.0 * np.kron(Z, Z)
    plus = np.array([1, 1]) / np.sqrt(2)
    b = np.array([1, 0]) if neighbor == 0 else np.array([0, 1])
    psi = np.exp(-1j * np.diag(H) * t) * np.kron(plus, b)   # H es diagonal: expm = fases
    X = np.array([[0, 1], [1, 0]]) if basis == "X" else np.array([[0, -1j], [1j, 0]])
    return float(np.real(psi.conj() @ np.kron(X, np.eye(2)) @ psi))


# ------------------------------------------------------------------ cuantización del control
def quantize(x: np.ndarray, levels: int) -> np.ndarray:
    """Redondeo al nivel más cercano de `levels` niveles uniformes en [0, 1) (vuelta completa)."""
    return np.round(np.asarray(x) * levels) / levels


def rms_quantization_error(levels: int, samples: int = 200000, seed: int = 0, unit_period: float = 1.0) -> float:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, samples)
    return float(np.sqrt(np.mean(((quantize(x, levels) - x) * unit_period) ** 2)))
