"""qang.qstring — Cap. 42: lo que de la teoría de cuerdas se puede leer y verificar con qang (NumPy; SciPy sólo en la parte XXZ).

Cuatro bloques, todos con verificador (`tools/verify_cap42.py`):
  A. cuerda bosónica de luz-cono: degeneraciones por nivel, dimensión crítica, masa sin masa cerrada (D=26, 24 transversales);
  B. el nivel N como peso ponderado: una lectura Z lineal (N = Σ_k k·(1−qg_Z,k)/2) y su filtro de nivel bajo T1;
  C. T-dualidad del bosón compactado: espectro invariante bajo R → α'/R y estados sin masa extra en el radio autodual;
  D. el radio (parámetro de Luttinger K) de una cadena XXZ leído con una sola base Z (varianza del peso de un bloque).
Nada de esto mide física de cuerdas: son identidades exactas del modelo y una lectura de un análogo de red. m-qg = 1000·qg_Z;
1 vuelta = 1000 qga (aquí no se usa qga). Qubit 0 = bit más significativo; el modo k = qubit k−1.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------------ A. cuerda bosónica (luz-cono)
def colored_partitions(n_max: int, colors: int = 24) -> List[int]:
    """Coeficientes de ∏_{n≥1}(1−qⁿ)^(−colors) hasta qⁿ_max: degeneración del nivel N con `colors` osciladores transversales."""
    c = [1] + [0] * n_max
    for n in range(1, n_max + 1):
        for _ in range(colors):
            for m in range(n, n_max + 1):
                c[m] += c[m - n]
    return c


def zero_point_energy(D: int) -> Fraction:
    """Energía del punto cero −(D−2)/24 (en unidades de nivel), exacta."""
    return Fraction(D - 2, 24)


def critical_dimension() -> int:
    """D tal que el primer nivel excitado (N=1) es sin masa: (D−2)/24 = 1 ⇒ D = 26."""
    return next(D for D in range(3, 200) if zero_point_energy(D) == 1)


def open_mass2_alpha(N: int, D: int = 26) -> Fraction:
    """α'·M² = N − (D−2)/24 para la cuerda abierta (N=0: taquión, N=1: sin masa en D=26)."""
    return N - zero_point_energy(D)


def closed_massless_decomposition(d: int = 24) -> Dict[str, int]:
    """Nivel N=Ñ=1 de la cuerda cerrada: d×d estados = simétrico sin traza + antisimétrico + traza."""
    sym, anti = d * (d + 1) // 2 - 1, d * (d - 1) // 2
    return {"total": d * d, "graviton": sym, "kalb_ramond": anti, "dilaton": 1}


# ------------------------------------------------------------------ B. nivel = peso ponderado
def level_of(x: int, n: int) -> int:
    """Nivel W(x) = Σ_k k·x_k, con el modo k = qubit k−1 (qubit 0 = bit más significativo)."""
    return int(sum((q + 1) for q in range(n) if (x >> (n - 1 - q)) & 1))


def level_vector(n: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    return sum(((idx >> (n - 1 - q)) & 1) * (q + 1) for q in range(n)).astype(int)


def distinct_partitions(n_max: int) -> List[int]:
    """Coeficientes de ∏_{k≥1}(1+q^k): particiones de N en partes distintas (modos fermiónicos de ocupación 0/1)."""
    c = [1] + [0] * n_max
    for k in range(1, n_max + 1):
        for m in range(n_max, k - 1, -1):
            c[m] += c[m - k]
    return c


def level_sector_size(N: int, n: int) -> int:
    """Número de cadenas de n qubits con nivel N."""
    return int(np.sum(level_vector(n) == N))


def level_from_z(p: np.ndarray, n: int) -> float:
    """⟨N⟩ = Σ_x p(x) W(x) = Σ_k k·(1000−m_k)/2000 con m_k = lectura qg_Z del qubit k−1 en m-qg (lineal en una sola base Z)."""
    return float(np.sum(np.asarray(p) * level_vector(n)))


def mode_readings_mqg(p: np.ndarray, n: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    return np.array([1000.0 * float(np.sum(p[((idx >> (n - 1 - q)) & 1) == 0]) - np.sum(p[((idx >> (n - 1 - q)) & 1) == 1])) for q in range(n)])


def level_from_readings(m_mqg: Sequence[float]) -> float:
    return float(sum((k + 1) * (1000.0 - m) / 2000.0 for k, m in enumerate(m_mqg)))


def level_variance(p: np.ndarray, n: int) -> float:
    W = level_vector(n)
    mu = float(np.sum(p * W))
    return float(np.sum(p * W * W) - mu * mu)


def level_shots(p: np.ndarray, n: int, sigma: float, z: float = 1.96) -> int:
    """Disparos para que ⟨N⟩ tenga error ≤ z·σ_N/√N_shots = sigma (mismos disparos Z, sin ajustes extra)."""
    return int(np.ceil(z ** 2 * level_variance(p, n) / sigma ** 2))


def damped_distribution(x: int, n: int, gamma: float) -> np.ndarray:
    """T1 de probabilidad γ sobre cada modo ocupado de |x⟩: distribución final sobre cadenas."""
    p = np.zeros(2 ** n)
    occ = [q for q in range(n) if (x >> (n - 1 - q)) & 1]
    for mask in range(2 ** len(occ)):
        y, pr = 0, 1.0
        for j, q in enumerate(occ):
            if (mask >> j) & 1:
                y |= 1 << (n - 1 - q)
                pr *= 1.0 - gamma
            else:
                pr *= gamma
        p[y] += pr
    return p


def level_filter_damped(x: int, n: int, gamma: float) -> Dict[str, float]:
    """Filtro de nivel N=W(x) sobre un estado de base bajo T1: cualquier decaimiento baja el nivel, así que se detecta siempre.
    Retención exacta (1−γ)^(peso de Hamming)."""
    p = damped_distribution(x, n, gamma)
    W = level_vector(n)
    kept = float(np.sum(p[W == level_of(x, n)]))
    hw = bin(x).count("1")
    return {"kept": kept, "kept_closed": (1.0 - gamma) ** hw, "mean_level_raw": float(np.sum(p * W)), "level_target": level_of(x, n)}


def level_vs_weight_blindness(n: int = 4) -> Dict[str, Tuple[int, int, int, int]]:
    """Dos errores exactos: el intercambio {3}↔{1,2} conserva el nivel y cambia el peso (el filtro de nivel no lo ve);
    el movimiento {1}→{2} conserva el peso y cambia el nivel (el filtro de peso no lo ve). Devuelve (nivel_a, nivel_b, peso_a, peso_b)."""
    s = lambda modes: sum(1 << (n - k) for k in modes)  # modo k = qubit k−1
    pairs = {"level_preserving": (s([3]), s([1, 2])), "weight_preserving": (s([1]), s([2]))}
    return {k: (level_of(a, n), level_of(b, n), bin(a).count("1"), bin(b).count("1")) for k, (a, b) in pairs.items()}


# ------------------------------------------------------------------ C. T-dualidad del bosón compactado
def momenta(n: int, w: int, R: float, alpha: float = 1.0) -> Tuple[float, float]:
    """p_L = n/R + wR/α', p_R = n/R − wR/α'."""
    return n / R + w * R / alpha, n / R - w * R / alpha


def conformal_weights(n: int, w: int, N: int, Nt: int, R: float, alpha: float = 1.0) -> Tuple[float, float]:
    """(h, h̄) = (α'p_L²/4 + N − 1, α'p_R²/4 + Ñ − 1). Estado físico: h = h̄ = 0 (sin masa)."""
    pl, pr = momenta(n, w, R, alpha)
    return alpha * pl ** 2 / 4 + N - 1, alpha * pr ** 2 / 4 + Nt - 1


def closed_spectrum(R: float, alpha: float = 1.0, nmax: int = 10, levmax: int = 3, d: int = 24, hmax: float = 2.0) -> Dict[Tuple[float, float], int]:
    """Estados con emparejamiento de nivel (n·w + N − Ñ = 0) y h ≤ hmax: {(h, h̄): degeneración}, con d osciladores por lado."""
    deg = colored_partitions(levmax, d)
    out: Dict[Tuple[float, float], int] = {}
    for n in range(-nmax, nmax + 1):
        for w in range(-nmax, nmax + 1):
            for N in range(levmax + 1):
                Nt = n * w + N
                if not 0 <= Nt <= levmax:
                    continue
                h, hb = conformal_weights(n, w, N, Nt, R, alpha)
                if h <= hmax + 1e-12 and hb <= hmax + 1e-12:
                    k = (round(h, 9), round(hb, 9))
                    out[k] = out.get(k, 0) + deg[N] * deg[Nt]
    return out


def t_dual_equal(R: float, alpha: float = 1.0, **kw) -> bool:
    """El espectro a R y a α'/R coincide (n ↔ w)."""
    return closed_spectrum(R, alpha, **kw) == closed_spectrum(alpha / R, alpha, **kw)


def massless_states(R: float, alpha: float = 1.0, **kw) -> int:
    """Número de estados con h = h̄ = 0 (nivel emparejado)."""
    return int(closed_spectrum(R, alpha, **kw).get((0.0, 0.0), 0))


def massless_breakdown(R: float, alpha: float = 1.0, nmax: int = 10, levmax: int = 3, d: int = 24) -> List[Tuple[int, int, int, int, int]]:
    """Estados sin masa con carga (n, w) ≠ (0, 0): lista (n, w, N, Ñ, degeneración)."""
    deg = colored_partitions(levmax, d)
    out = []
    for n in range(-nmax, nmax + 1):
        for w in range(-nmax, nmax + 1):
            if n == 0 and w == 0:
                continue
            for N in range(levmax + 1):
                Nt = n * w + N
                if 0 <= Nt <= levmax:
                    h, hb = conformal_weights(n, w, N, Nt, R, alpha)
                    if abs(h) < 1e-9 and abs(hb) < 1e-9:
                        out.append((n, w, N, Nt, deg[N] * deg[Nt]))
    return out


# ------------------------------------------------------------------ D. radio de una cadena XXZ con una sola base Z
def luttinger_k_exact(delta: float) -> float:
    """K = π/(2(π − arccos Δ)) para H = Σ S^x S^x + S^y S^y + Δ S^z S^z (K=1 en Δ=0, K=1/2 en Δ=1)."""
    return float(np.pi / (2.0 * (np.pi - np.arccos(delta))))


def dual_exponents(K: float) -> Tuple[float, float]:
    """Exponentes de las correlaciones escalonadas: ZZ ~ r^(−2K), XX ~ r^(−1/(2K)); T-dualidad K → 1/K los intercambia."""
    return 2.0 * K, 1.0 / (2.0 * K)


def xxz_ground_distribution(L: int, delta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Distribución Z (|amplitud|²) del estado fundamental del anillo XXZ de L sitios en el sector S^z=0. Requiere SciPy."""
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as sla
    except ImportError as e:  # pragma: no cover
        raise ImportError("xxz_ground_distribution necesita SciPy: pip install scipy") from e
    allx = np.arange(2 ** L, dtype=np.int64)
    pc = np.zeros(2 ** L, dtype=np.int64)
    for q in range(L):
        pc += (allx >> q) & 1
    st = allx[pc == L // 2]
    D = len(st)
    rows, cols, diag = [], [], np.zeros(D)
    for i in range(L):
        j = (i + 1) % L
        bi, bj = (st >> (L - 1 - i)) & 1, (st >> (L - 1 - j)) & 1
        diag += delta * np.where(bi == bj, 0.25, -0.25)
        m = bi != bj
        t = st[m] ^ (1 << (L - 1 - i)) ^ (1 << (L - 1 - j))
        rows.append(np.searchsorted(st, t))
        cols.append(np.nonzero(m)[0])
    H = sp.coo_matrix((np.full(sum(len(r) for r in rows), 0.5), (np.concatenate(rows), np.concatenate(cols))), shape=(D, D)).tocsr() + sp.diags(diag)
    if D <= 64:
        w, v = np.linalg.eigh(H.toarray())
        return st, v[:, 0] ** 2
    w, v = sla.eigsh(H, k=1, which="SA")
    return st, v[:, 0] ** 2


def block_weight_variance(states: np.ndarray, p: np.ndarray, L: int, ell: int) -> float:
    """Var(peso de Hamming de los primeros ell qubits) bajo la distribución Z: un solo histograma."""
    w = np.zeros(len(states))
    for q in range(ell):
        w += (states >> (L - 1 - q)) & 1
    mu = float(np.sum(p * w))
    return float(np.sum(p * w * w) - mu * mu)


def luttinger_k_from_variance(states: np.ndarray, p: np.ndarray, L: int, ell_min: int = 4) -> float:
    """K̂ = π²·pendiente de Var(bloque) frente a ln[(L/π) sin(πℓ/L)], con ℓ pares ≥ ell_min (las oscilaciones impares sesgan el ajuste)."""
    ells = [l for l in range(ell_min, L // 2 + 1) if l % 2 == 0]
    chord = np.array([(L / np.pi) * np.sin(np.pi * l / L) for l in ells])
    var = np.array([block_weight_variance(states, p, L, l) for l in ells])
    return float(np.polyfit(np.log(chord), var, 1)[0] * np.pi ** 2)
