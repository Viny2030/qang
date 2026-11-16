"""qang.qecc — Cap. 29: código AD de 4 qubits (Leung) con síndrome qg_ZZZZ en m-qg, repetición frente a borrado,
rango de Schmidt en el sector de peso como cota del corte de circuitos y coste en disparos de ZNE (Richardson).
NumPy/SciPy exacto (matrices densidad de 4 qubits). Convención: bit 1 = excitado; T1 lleva 1→0. m-qg = 1000·qg.
"""
from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------ código AD de 4 qubits (Leung 1997)
def _bits(z: int, n: int = 4):
    return [(z >> (n - 1 - q)) & 1 for q in range(n)]


def _basis(s: str) -> np.ndarray:
    v = np.zeros(16, dtype=complex)
    v[int(s, 2)] = 1.0
    return v


ZERO_L = (_basis("0000") + _basis("1111")) / np.sqrt(2)
ONE_L = (_basis("0011") + _basis("1100")) / np.sqrt(2)


def logical_state(a: complex, b: complex) -> np.ndarray:
    return a * ZERO_L + b * ONE_L


def _kraus_t1(g: float, n: int = 4):
    k0 = np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex)
    k1 = np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex)
    ops = []
    for bits in range(2 ** n):
        M = np.array([[1.0 + 0j]])
        for q in range(n):
            M = np.kron(M, k1 if (bits >> (n - 1 - q)) & 1 else k0)
        ops.append(M)
    return ops


def apply_t1(rho: np.ndarray, g: float) -> np.ndarray:
    return sum(K @ rho @ K.conj().T for K in _kraus_t1(g))


def parity_operator() -> np.ndarray:
    """ZZZZ = (−1)^{peso}: su lectura es el síndrome, +1000 m-qg (par) o −1000 m-qg (impar)."""
    return np.diag([(-1.0) ** bin(z).count("1") for z in range(16)]).astype(complex)


def parity_mqg(rho: np.ndarray) -> float:
    """qg_ZZZZ en m-qg = 1000·(1 − 2 P_impar)."""
    return float(1000.0 * np.real(np.trace(rho @ parity_operator())))


def recovery_operators() -> list:
    """{P_par, R_k}: par → identidad; impar → isometría v0_k→|0_L>, v1_k→|1_L> (k = qubit que saltó)."""
    ops = [np.diag([1.0 if bin(z).count("1") % 2 == 0 else 0.0 for z in range(16)]).astype(complex)]
    for k in range(4):
        low = np.zeros((16, 16), dtype=complex)
        for z in range(16):
            if _bits(z)[k] == 1:
                low[z ^ (1 << (3 - k)), z] = 1.0
        v0, v1 = low @ ZERO_L, low @ ONE_L
        v0, v1 = v0 / np.linalg.norm(v0), v1 / np.linalg.norm(v1)
        ops.append(np.outer(ZERO_L, v0.conj()) + np.outer(ONE_L, v1.conj()))
    return ops


def recover(rho: np.ndarray) -> np.ndarray:
    return sum(R @ rho @ R.conj().T for R in recovery_operators())


_PAULI6 = [(1, 0), (0, 1), (1, 1), (1, -1), (1, 1j), (1, -1j)]


def avg_fidelity_encoded(g: float, decode: bool = True) -> float:
    """Fidelidad lógica media sobre los 6 autoestados de Pauli, con T1 de tasa g en cada qubit físico."""
    f = 0.0
    for a, b in _PAULI6:
        v = np.array([a, b], dtype=complex)
        v = v / np.linalg.norm(v)
        psi = logical_state(*v)
        rho = apply_t1(np.outer(psi, psi.conj()), g)
        if decode:
            rho = recover(rho)
        f += float(np.real(psi.conj() @ rho @ psi))
    return f / 6.0


def avg_fidelity_bare(g: float) -> float:
    """Mismo promedio sin codificar: un qubit con T1 de tasa g."""
    k0 = np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex)
    k1 = np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex)
    f = 0.0
    for a, b in _PAULI6:
        v = np.array([a, b], dtype=complex)
        v = v / np.linalg.norm(v)
        rho = np.outer(v, v.conj())
        r = k0 @ rho @ k0.conj().T + k1 @ rho @ k1.conj().T
        f += float(np.real(v.conj() @ r @ v))
    return f / 6.0


def loglog_slope(gs: Sequence[float], ys: Sequence[float]) -> float:
    return float(np.polyfit(np.log(np.asarray(gs)), np.log(np.asarray(ys)), 1)[0])


def odd_probability(g: float) -> float:
    psi = logical_state(1 / np.sqrt(2), 1 / np.sqrt(2))
    r = apply_t1(np.outer(psi, psi.conj()), g)
    P = recovery_operators()[0]
    return float(1.0 - np.real(np.trace(P @ r)))


# ------------------------------------------------------------------ repetición: error de bit frente a borrado
def rep_logical_error_flip(n: int, p: float) -> float:
    """Código de repetición de n qubits, voto por mayoría, error de bit p (empate: 1/2)."""
    err = sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))
    if n % 2 == 0:
        err += 0.5 * comb(n, n // 2) * p ** (n // 2) * (1 - p) ** (n // 2)
    return err


def rep_logical_error_erasure(n: int, e: float) -> float:
    """Borrado heraldado (se sabe cuáles se perdieron): sólo falla si se pierden los n (se adivina: 1/2)."""
    return 0.5 * e ** n


def rep_error_by_enumeration(n: int, rate: float, mode: str) -> float:
    """Comprobación independiente por enumeración de los 2^n patrones."""
    tot = 0.0
    for pat in range(2 ** n):
        k = bin(pat).count("1")
        pr = rate ** k * (1 - rate) ** (n - k)
        if mode == "flip":
            tot += pr * (1.0 if k > n / 2 else (0.5 if k == n / 2 else 0.0))
        else:  # erasure
            tot += pr * (0.5 if k == n else 0.0)
    return tot


# ------------------------------------------------------------------ rango de Schmidt en el sector de peso (corte)
def schmidt_rank(psi: np.ndarray, n: int, m: int, tol: float = 1e-10) -> int:
    """Rango de Schmidt del corte primeros m qubits | resto."""
    M = np.asarray(psi).reshape(2 ** m, 2 ** (n - m))
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol * s[0]))


def sector_rank_bound(n: int, m: int, w: int) -> int:
    """Rango de Schmidt máximo (genérico) de un estado de peso w para el corte m | n−m:
    Σ_k min(C(m,k), C(n−m,w−k)), k = excitaciones en A. Para w=1 vale 2 sea cual sea n y m."""
    return sum(min(comb(m, k), comb(n - m, w - k)) for k in range(0, w + 1) if k <= m and w - k <= n - m)


def random_sector_state(n: int, w: int, rng) -> np.ndarray:
    psi = np.zeros(2 ** n, dtype=complex)
    for z in range(2 ** n):
        if bin(z).count("1") == w:
            psi[z] = rng.normal() + 1j * rng.normal()
    return psi / np.linalg.norm(psi)


def wires_for_rank(rank: int) -> int:
    return int(np.ceil(np.log2(rank)))


def wire_cut_shot_overhead(wires: int) -> int:
    """Modelo estándar de corte de cable por cuasi-probabilidad: ×16 disparos por cable (Peng et al.)."""
    return 16 ** wires


# ------------------------------------------------------------------ ZNE (Richardson) y coste en disparos
def richardson_weights(lams: Sequence[float]) -> np.ndarray:
    """c tal que Σ c_i f(λ_i) = f(0) para todo polinomio de grado < len(lams)."""
    l = np.asarray(lams, dtype=float)
    c = np.zeros(len(l))
    for i in range(len(l)):
        c[i] = np.prod([(0 - l[j]) / (l[i] - l[j]) for j in range(len(l)) if j != i])
    return c


def zne_variance_factor(lams: Sequence[float]) -> float:
    """Var[estimador]/σ² con σ igual en cada escala: Σ c_i²."""
    return float(np.sum(richardson_weights(lams) ** 2))


def zne_estimate(f, lams: Sequence[float]) -> float:
    return float(np.dot(richardson_weights(lams), [f(l) for l in lams]))
