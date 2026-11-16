"""qang.qnano — Cap. 37: nanocuántica leída con la ecuación de qang (túnel, Landauer, ruido de partición, ocupación de un nivel,
centros NV). NumPy exacto; el ruido binomial se calcula con la distribución exacta, sin Monte Carlo, salvo la termometría.
m-qg = 1000·qg_Z; para una probabilidad p0: θ = 2 arccos √p0, qg_Z = 2p0 − 1.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

KAPPA_PER_NM_SQRT_EV = 5.123167  # sqrt(2 m_e · 1 eV)/ħ en nm⁻¹


def theta_from_p0(p0):
    return 2.0 * np.arccos(np.sqrt(np.clip(p0, 0.0, 1.0)))


def mqg_from_p0(p0):
    return 1000.0 * (2.0 * np.asarray(p0, dtype=float) - 1.0)


# ------------------------------------------------------------------ túnel
def barrier_T(a_nm: float, E_eV: float = 0.5, V_eV: float = 1.0) -> float:
    """Transmisión exacta de una barrera rectangular (E < V)."""
    kappa = KAPPA_PER_NM_SQRT_EV * math.sqrt(V_eV - E_eV)
    return 1.0 / (1.0 + V_eV ** 2 * math.sinh(kappa * a_nm) ** 2 / (4.0 * E_eV * (V_eV - E_eV)))


def kappa_per_nm(E_eV: float = 0.5, V_eV: float = 1.0) -> float:
    return KAPPA_PER_NM_SQRT_EV * math.sqrt(V_eV - E_eV)


def tunnel_slopes(widths_nm: Sequence[float], E_eV: float = 0.5, V_eV: float = 1.0) -> Tuple[float, float]:
    """Pendientes (por nm) de ln T y de ln(π − θ_T) frente al ancho (barreras gruesas): ≈ −2κ y ≈ −κ."""
    a = np.asarray(widths_nm, float)
    T = np.array([barrier_T(x, E_eV, V_eV) for x in a])
    return float(np.polyfit(a, np.log(T), 1)[0]), float(np.polyfit(a, np.log(math.pi - theta_from_p0(T)), 1)[0])


# ------------------------------------------------------------------ Landauer
def qpc_T(V: float, n_channels: int = 4, w: float = 0.45) -> np.ndarray:
    """Modelo de punto de silla: T_n = 1/(1 + e^{−2π(V − n − 1/2)/w}), n = 1..N."""
    n = np.arange(1, n_channels + 1)
    return 1.0 / (1.0 + np.exp(-2.0 * math.pi * (V - n - 0.5) / w))


def conductance_from_qg(T: np.ndarray) -> float:
    """G/G0 = N/2 + (1/2) Σ qg_n, con qg_n = 2 T_n − 1."""
    qg = 2.0 * np.asarray(T) - 1.0
    return len(qg) / 2.0 + 0.5 * float(qg.sum())


def recover_two_channels(G: float, S: float) -> Tuple[float, float]:
    """Dos transmisiones a partir de G = ΣT y del ruido S = ΣT(1−T) (ΣT² = G − S, T1·T2 = (G² − (G − S))/2)."""
    prod = (G * G - (G - S)) / 2.0
    disc = max(G * G - 4.0 * prod, 0.0)
    r = math.sqrt(disc)
    return (G + r) / 2.0, (G - r) / 2.0


# ------------------------------------------------------------------ ruido de partición (distribución binomial exacta)
def theta_std_sqrtN(T: float, N: int) -> float:
    """√N × desvío exacto del ángulo estimado θ̂ = 2 arccos √(k/N), k ~ Binomial(N, T)."""
    k = np.arange(N + 1)
    lg = np.array([math.lgamma(N + 1) - math.lgamma(i + 1) - math.lgamma(N - i + 1) for i in k])
    pmf = np.exp(lg + k * math.log(T) + (N - k) * math.log1p(-T))
    th = theta_from_p0(k / N)
    m = float((pmf * th).sum())
    return math.sqrt(float((pmf * (th - m) ** 2).sum()) * N)


def p_std_sqrtN(T: float) -> float:
    return math.sqrt(T * (1.0 - T))


# ------------------------------------------------------------------ ocupación de un nivel: termometría
def occupancy_qgz(x):
    """qg_Z = 2f − 1 = −tanh(x/2) con f = 1/(1 + e^x), x = ε/k_BT."""
    return -np.tanh(np.asarray(x, float) / 2.0)


def fit_kT_mle(eps: np.ndarray, ones: np.ndarray, N: int, lo: float = 0.2, hi: float = 5.0) -> float:
    """Máxima verosimilitud binomial de k_BT con la ocupación de Fermi (sección áurea)."""
    def nll(kt):
        f = np.clip(1.0 / (1.0 + np.exp(eps / kt)), 1e-12, 1 - 1e-12)
        return -np.sum(ones * np.log(f) + (N - ones) * np.log(1 - f))
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(c), nll(d)
    while b - a > 1e-10:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = nll(d)
    return 0.5 * (a + b)


def fit_kT_artanh(eps: np.ndarray, ones: np.ndarray, N: int) -> float:
    """Recta artanh(qg_Z) = −ε/(2 k_BT); descarta los puntos saturados (todas las lecturas iguales)."""
    q = 2.0 * ones / N - 1.0          # ones = niveles ocupados: p0 = f, qg_Z = 2f − 1 = −tanh(x/2)
    ok = np.abs(q) < 1.0
    y = np.arctanh(q[ok])
    slope = float((eps[ok] * y).sum() / (eps[ok] ** 2).sum())   # por el origen
    return -1.0 / (2.0 * slope)


def thermometry(N: int = 200, reps: int = 400, kT: float = 1.0, npts: int = 25, span: float = 4.0, seed: int = 34):
    """(media, desvío) de k_BT estimado por ML y por la recta en artanh."""
    rng = np.random.default_rng(seed)
    eps = np.linspace(-span, span, npts) * kT
    f = 1.0 / (1.0 + np.exp(eps / kT))
    a, b = [], []
    for _ in range(reps):
        ones = rng.binomial(N, f)
        a.append(fit_kT_mle(eps, ones, N))
        b.append(fit_kT_artanh(eps, ones, N))
    return (float(np.mean(a)), float(np.std(a))), (float(np.mean(b)), float(np.std(b)))


# ------------------------------------------------------------------ lectura de un espín por sensor de carga
def readout_affine(m, e0: float, e1: float):
    """m' = (1 − e0 − e1) m + 1000 (e1 − e0)."""
    return (1.0 - e0 - e1) * np.asarray(m, float) + 1000.0 * (e1 - e0)


# ------------------------------------------------------------------ centro NV
def nv_sigma_mqg(C: float, Nb: float, p0: float = 0.5) -> float:
    """σ exacta (propagación de Poisson) de la lectura de p0 en m-qg: n ~ Poisson(Nb[1 − C(1−p0)])."""
    return 2000.0 * math.sqrt(1.0 - C * (1.0 - p0)) / (C * math.sqrt(Nb))


def nv_photons_for(sigma_mqg: float, C: float, p0: float = 0.5) -> float:
    return (2000.0 * math.sqrt(1.0 - C * (1.0 - p0)) / (C * sigma_mqg)) ** 2


def odmr_fisher_crb_hz(fwhm_hz: float, C: float, counts_per_point: float, n_freq: int = 61, span: float = 3.0) -> float:
    """Cramér–Rao del centro de la lorentziana I(f) = 1 − C/(1+((f−f0)/(fwhm/2))²), cuentas Poisson, n_freq puntos."""
    f = np.linspace(-span, span, n_freq) * fwhm_hz
    h = fwhm_hz / 2.0
    L = 1.0 / (1.0 + (f / h) ** 2)
    mu = counts_per_point * (1.0 - C * L)
    dmu = -counts_per_point * C * (2.0 * f / h ** 2) * L ** 2    # d/df0 de mu = −d/df de mu
    F = float(np.sum(dmu ** 2 / mu))
    return 1.0 / math.sqrt(F)
