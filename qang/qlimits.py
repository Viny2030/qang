"""qang.qlimits — Cap. 34: límites estructurales de la lectura qg, comprobados con código exacto (NumPy).
Ceguera de la base Z (estados de grafo), qg_S no monótona bajo T1, trampa del rango de arccos, red de peso 1 como forma
cuadrática, analizador de conservación de peso y error dentro del sector. m-qg = 1000·qg_Z; 1 vuelta = 1000 qga.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------ utilidades
def popcount(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    out = np.zeros(a.shape, dtype=int)
    while np.any(a):
        out += a & 1
        a = a >> 1
    return out


def z_distribution(psi: np.ndarray) -> np.ndarray:
    """Distribución de la base Z: |amplitud|²."""
    return np.abs(psi) ** 2


def bit(idx: np.ndarray, q: int, n: int) -> np.ndarray:
    """Valor del qubit q (q=0 = bit más significativo) en el índice idx."""
    return (idx >> (n - 1 - q)) & 1


def mean_qgz(p: np.ndarray, n: int) -> float:
    """qg_Z medio sobre los n qubits: 1 − 2·P(1) promedio."""
    idx = np.arange(2 ** n)
    return float(np.mean([1.0 - 2.0 * p[bit(idx, q, n) == 1].sum() for q in range(n)]))


def qgs_per_qubit(p: np.ndarray, n: int) -> float:
    """qg_S = H(p)/n en bits por qubit (1 = uniforme)."""
    q = p[p > 0]
    return float(-(q * np.log2(q)).sum() / n)


# ------------------------------------------------------------------ 1. estados de grafo: entrelazamiento invisible en Z
def plus_state(n: int) -> np.ndarray:
    return np.full(2 ** n, 2 ** (-n / 2), dtype=complex)


def ring_graph_state(n: int) -> np.ndarray:
    """H^{⊗n} seguido de una CZ por arista del anillo de n qubits."""
    psi = plus_state(n)
    idx = np.arange(2 ** n)
    for q in range(n):
        psi = psi * np.where(bit(idx, q, n) * bit(idx, (q + 1) % n, n) == 1, -1.0, 1.0)
    return psi


def stabilizer_expectation(psi: np.ndarray, i: int, n: int) -> float:
    """⟨X_i Z_{i−1} Z_{i+1}⟩ en el anillo (base X sólo en el qubit i)."""
    idx = np.arange(2 ** n)
    sign = np.where((bit(idx, (i - 1) % n, n) + bit(idx, (i + 1) % n, n)) % 2 == 1, -1.0, 1.0)
    flipped = idx ^ (1 << (n - 1 - i))
    return float(np.real(np.vdot(psi, sign * psi[flipped])))


# ------------------------------------------------------------------ 2. qg_S no monótona bajo T1
def damp_distribution(p: np.ndarray, n: int, gamma: float) -> np.ndarray:
    """Amortiguamiento de amplitud en cada qubit: las poblaciones de Z siguen una cadena de Markov 1→0 con prob. γ."""
    p = p.copy().reshape((2,) * n)
    for q in range(n):
        p0, p1 = np.take(p, 0, axis=q), np.take(p, 1, axis=q)
        p = np.stack([p0 + gamma * p1, (1 - gamma) * p1], axis=q)
    return p.reshape(-1)


def flip_distribution(p: np.ndarray, n: int, q_flip: float) -> np.ndarray:
    """Ruido simétrico (bit-flip con prob. q_flip en cada qubit, el efecto del ruido despolarizante sobre Z)."""
    p = p.copy().reshape((2,) * n)
    for q in range(n):
        p0, p1 = np.take(p, 0, axis=q), np.take(p, 1, axis=q)
        p = np.stack([(1 - q_flip) * p0 + q_flip * p1, (1 - q_flip) * p1 + q_flip * p0], axis=q)
    return p.reshape(-1)


def is_monotone(x: Sequence[float], increasing: bool = True, tol: float = 1e-12) -> bool:
    d = np.diff(np.asarray(x, dtype=float))
    return bool(np.all(d >= -tol) if increasing else np.all(d <= tol))


def t1_noise_scan(p: np.ndarray, n: int, gammas: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """(qg_S, qg_Z medio) frente a γ bajo amortiguamiento."""
    s, z = [], []
    for g in gammas:
        pg = damp_distribution(p, n, g)
        s.append(qgs_per_qubit(pg, n))
        z.append(mean_qgz(pg, n))
    return np.array(s), np.array(z)


def pair_separates(p: np.ndarray, n: int, g_lo: float, g_hi: float) -> Tuple[float, float, float, float]:
    """qg_S y qg_Z medio en dos γ; devuelve (S_lo, S_hi, Z_lo, Z_hi)."""
    s, z = t1_noise_scan(p, n, [g_lo, g_hi])
    return float(s[0]), float(s[1]), float(z[0]), float(z[1])


# ------------------------------------------------------------------ 3. trampa del rango de arccos
def cost_cos(theta: float, theta_star: float) -> float:
    """E(θ) = −cos(θ − θ*): mínimo −1 en θ = θ*."""
    return -float(np.cos(theta - theta_star))


def best_cost_clipped(theta_star: float, num: int = 200001) -> Tuple[float, float]:
    """Mejor valor alcanzable con θ = arccos(qg_Z) ∈ [0, π] (qg_Z ∈ [−1, 1]) y el θ que lo logra."""
    th = np.linspace(0.0, np.pi, num)
    e = -np.cos(th - theta_star)
    k = int(np.argmin(e))
    return float(e[k]), float(th[k])


# ------------------------------------------------------------------ 4. conservación de peso (analizador de compuertas)
def gate_on(n: int, U: np.ndarray, qubits: Sequence[int]) -> np.ndarray:
    """Embebe una compuerta de k qubits (qubits[0] = más significativo de U) en n qubits."""
    k = len(qubits)
    N = 2 ** n
    full = np.zeros((N, N), dtype=complex)
    for col in range(N):
        sub = 0
        for q in qubits:
            sub = (sub << 1) | int(bit(np.array(col), q, n))
        for row_sub in range(2 ** k):
            amp = U[row_sub, sub]
            if amp == 0:
                continue
            row = col
            for t, q in enumerate(qubits):
                b = (row_sub >> (k - 1 - t)) & 1
                row = (row & ~(1 << (n - 1 - q))) | (b << (n - 1 - q))
            full[row, col] += amp
    return full


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def cphase(phi: float) -> np.ndarray:
    return np.diag([1, 1, 1, np.exp(1j * phi)]).astype(complex)


def rbs(theta: float) -> np.ndarray:
    """RBS: rota {|01>,|10>} por θ, deja |00> y |11> (conserva el peso)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]], dtype=complex)


def preserves_weight(U: np.ndarray, n: int, tol: float = 1e-12) -> bool:
    """¿U es diagonal por bloques de peso de Hamming?"""
    idx = np.arange(2 ** n)
    w = popcount(idx)
    mask = w[:, None] != w[None, :]
    return bool(np.max(np.abs(U[mask])) < tol) if mask.any() else True


Gate = Tuple[str, np.ndarray, Tuple[int, ...]]


def qft_gates(n: int) -> List[Gate]:
    g: List[Gate] = []
    for j in range(n):
        g.append(("H", _H, (j,)))
        for k in range(j + 1, n):
            g.append(("CP", cphase(np.pi / 2 ** (k - j)), (k, j)))
    for j in range(n // 2):
        g.append(("SWAP", _SWAP, (j, n - 1 - j)))
    return g


def ghz_gates(n: int) -> List[Gate]:
    return [("H", _H, (0,))] + [("CNOT", _CNOT, (i, i + 1)) for i in range(n - 1)]


def unary_rbs_gates(n: int, thetas: Sequence[float]) -> List[Gate]:
    """Red RBS en escalera sobre n qubits (conserva el peso)."""
    return [("RBS", rbs(t), (i, i + 1)) for i, t in zip(range(n - 1), thetas)]


def circuit_unitary(n: int, gates: Sequence[Gate]) -> np.ndarray:
    U = np.eye(2 ** n, dtype=complex)
    for _, M, qs in gates:
        U = gate_on(n, M, qs) @ U
    return U


def weight_breakers(n: int, gates: Sequence[Gate]) -> List[str]:
    """Nombres de las compuertas que, solas, no conservan el peso."""
    return [name for name, M, qs in gates if not preserves_weight(gate_on(n, M, qs), n)]


# ------------------------------------------------------------------ 5. red de peso 1 = forma cuadrática clásica
def weight1_orthogonal(n: int, thetas: Sequence[float]) -> np.ndarray:
    """Matriz O (n×n) de la red RBS en escalera restringida al sector de una excitación."""
    O = np.eye(n)
    for i, t in enumerate(thetas[: n - 1]):
        R = np.eye(n)
        c, s = np.cos(t), np.sin(t)
        R[i, i], R[i, i + 1], R[i + 1, i], R[i + 1, i + 1] = c, -s, s, c
        O = R @ O
    return O


def weight1_qgz(O: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Forma cuadrática: qg_Z^{(i)} = 1 − 2 (O v)_i²."""
    return 1.0 - 2.0 * (O @ v) ** 2


def simulate_qgz_weight1(n: int, thetas: Sequence[float], v: np.ndarray) -> np.ndarray:
    """Simulación completa (2^n) del cargador unario + red RBS; qg_Z de cada qubit."""
    psi = np.zeros(2 ** n, dtype=complex)
    for i in range(n):
        psi[1 << (n - 1 - i)] = v[i]
    U = circuit_unitary(n, unary_rbs_gates(n, thetas))
    p = z_distribution(U @ psi)
    idx = np.arange(2 ** n)
    return np.array([1.0 - 2.0 * p[bit(idx, q, n) == 1].sum() for q in range(n)])


# ------------------------------------------------------------------ 6. error dentro del sector
def in_sector_error(eps: float) -> Dict[str, float]:
    """RBS ideal de π/2 sobre |10> frente a RBS(π/2+ε): el peso se conserva (retención 1) y el error queda en el sector."""
    psi0 = np.zeros(4, dtype=complex)
    psi0[2] = 1.0
    out = {}
    for name, th in (("ideal", np.pi / 2), ("error", np.pi / 2 + eps)):
        p = z_distribution(rbs(th) @ psi0)
        out[name + "_retained"] = float(p[1] + p[2])
        out[name + "_qgz0_mqg"] = 1000.0 * float(1.0 - 2.0 * (p[2] + p[3]))
    out["shift_mqg"] = out["error_qgz0_mqg"] - out["ideal_qgz0_mqg"]
    return out


# ------------------------------------------------------------------ 7. inventario de resultados negativos del libro de cuentas
def negative_inventory(rows: List[Dict]) -> Dict[str, List[Tuple[int, str]]]:
    """Filas con estatus principal against / no gain / mixed / unverified / not adopted (usa qang.qledger.load_ledger)."""
    from .qledger import primary_status

    inv: Dict[str, List[Tuple[int, str]]] = {k: [] for k in ("against", "no gain", "mixed", "unverified", "not adopted")}
    for r in rows:
        s = primary_status(r["status"])
        if s in inv:
            inv[s].append((r["cap"], r["id"]))
    return inv
