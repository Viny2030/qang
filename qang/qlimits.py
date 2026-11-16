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


# ------------------------------------------------------------------ 8. una sola base de lectura: no-go, C_Z, pureza, testigos
# Todas las bases en bits (log base 2). H_Z en bits; H_Z/n se rotula aparte. Qubit 0 = bit más significativo.
_I2 = np.eye(2, dtype=complex)
_PX = np.array([[0, 1], [1, 0]], dtype=complex)
_PZ = np.diag([1.0, -1.0]).astype(complex)
HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _kron(*ops: np.ndarray) -> np.ndarray:
    out = np.eye(1, dtype=complex)
    for o in ops:
        out = np.kron(out, o)
    return out


def shannon_bits(p: np.ndarray) -> float:
    """H(p) en bits, con 0·log 0 = 0."""
    q = np.asarray(p, dtype=float)
    q = q[q > 0]
    return float(-np.sum(q * np.log2(q)))


def h2(x: float) -> float:
    return shannon_bits(np.array([x, 1.0 - x]))


def z_marginals(p: np.ndarray, n: int) -> np.ndarray:
    """P(bit_i = 0) para cada qubit i."""
    idx = np.arange(2 ** n)
    return np.array([float(np.sum(p[bit(idx, q, n) == 0])) for q in range(n)])


def total_correlation_z(p: np.ndarray, n: int) -> float:
    """C_Z = Σ_i H(X_i) − H(X_1..X_n) = KL(p ‖ ∏ p_i), en bits. Es información de resultados, no de entrelazamiento."""
    return float(sum(h2(m) for m in z_marginals(p, n)) - shannon_bits(p))


def diagonal_twin(p: np.ndarray) -> np.ndarray:
    """σ_p = Σ_x p(x)|x⟩⟨x|: completamente separable y con la misma distribución Z que cualquier estado con diagonal p."""
    return np.diag(np.asarray(p, dtype=complex))


def purity_range_from_z(p: np.ndarray) -> Tuple[float, float]:
    """Rango exacto de Tr ρ² compatible con la diagonal p: [Σ p², 1] (ambos extremos alcanzables)."""
    return float(np.sum(np.asarray(p) ** 2)), 1.0


def pure_state_from_z(p: np.ndarray) -> np.ndarray:
    """|ψ_p⟩ = Σ √p(x)|x⟩: estado puro con la diagonal p."""
    return np.sqrt(np.asarray(p, dtype=complex))


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_bits(rho: np.ndarray) -> float:
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    return shannon_bits(np.clip(w, 0.0, None))


def relative_entropy_coherence_z(rho: np.ndarray) -> float:
    """C_rel,Z = S(Δ_Z ρ) − S(ρ) = H_Z − S(ρ); H_Z = C_rel,Z sólo si ρ es puro."""
    return shannon_bits(np.real(np.diag(rho))) - von_neumann_bits(rho)


def reduced_state(rho: np.ndarray, n: int, keep: Sequence[int]) -> np.ndarray:
    t = rho.reshape([2] * (2 * n))
    keep = list(keep)
    trace_out = [q for q in range(n) if q not in keep]
    for k, q in enumerate(sorted(trace_out, reverse=True)):
        m = t.ndim // 2
        t = np.trace(t, axis1=q, axis2=q + m)
    d = 2 ** len(keep)
    return t.reshape(d, d)


def local_purities(rho: np.ndarray, n: int) -> List[float]:
    return [purity(reduced_state(rho, n, [q])) for q in range(n)]


def partial_transpose_min_eig(rho: np.ndarray, n: int, n_a: int = 1) -> float:
    """Menor autovalor de la transpuesta parcial sobre los primeros n_a qubits (Peres): < 0 certifica entrelazamiento."""
    da, db = 2 ** n_a, 2 ** (n - n_a)
    t = rho.reshape(da, db, da, db).transpose(2, 1, 0, 3).reshape(2 ** n, 2 ** n)
    return float(np.min(np.linalg.eigvalsh((t + t.conj().T) / 2)))


def cz_matrix(n: int, a: int, b: int) -> np.ndarray:
    d = 2 ** n
    idx = np.arange(d)
    return np.diag(np.where((bit(idx, a, n) & bit(idx, b, n)) == 1, -1.0, 1.0)).astype(complex)


def path_graph_state(n: int) -> np.ndarray:
    """H^n seguido de CZ en el camino 0–1–…–(n−1) (n=2: arista; n=3: camino de 3)."""
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for q in range(n - 1):
        psi = cz_matrix(n, q, q + 1) @ psi
    return psi


def ghz_state_n(n: int) -> np.ndarray:
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = psi[-1] = 1 / np.sqrt(2)
    return psi


def apply_local(psi: np.ndarray, n: int, ops: Dict[int, np.ndarray]) -> np.ndarray:
    return _kron(*[ops.get(q, _I2) for q in range(n)]) @ psi


def witness_path(n: int) -> np.ndarray:
    """W′ = (n−1)·I − Σ_i K_i, K_i = X_i ∏_{j vecino} Z_j, camino de n qubits (Tóth–Gühne, n=2,3)."""
    W = (n - 1) * np.eye(2 ** n, dtype=complex)
    for i in range(n):
        ops = [_I2] * n
        ops[i] = _PX
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                ops[j] = _PZ
        W = W - _kron(*ops)
    return W


def witness_value(W: np.ndarray, rho: np.ndarray) -> float:
    return float(np.real(np.trace(W @ rho)))


def witness_radius(m: int, shots: int, delta: float = 0.05) -> float:
    """Radio conservador m·η, η = √(2 ln(2m/δ)/N): certificar sólo si Ŵ′ + m·η < 0."""
    return float(m * np.sqrt(2.0 * np.log(2.0 * m / delta) / shots))


def hoeffding_shots(eps: float, delta: float, n_means: int = 1) -> int:
    """N suficiente para |μ̂ − μ| ≤ ε con prob. ≥ 1−δ sobre n_means medias en [−1,1] (N ≥ 2ε⁻² ln(2n/δ))."""
    return int(np.ceil(2.0 * np.log(2.0 * n_means / delta) / eps ** 2))


def _t_for_error(f, eps: float, K: int) -> float:
    lo, hi = 0.0, 1.0 - 1.0 / K
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) <= eps else (lo, mid)
    return lo


def sufficient_shots_entropy(n: int, eps: float, delta: float, target: str = "H") -> int:
    """N suficiente (cota distribución-libre, deliberadamente holgada) para |Ĥ_Z − H_Z| ≤ ε o |Ĉ_Z − C_Z| ≤ ε.
    TV(p, p̂) ≤ t con N ≥ (K ln 2 + ln(1/δ)) / (2t²), K = 2ⁿ; continuidad: h2(t) + t log2(K−1) [H] o (n+1)h2(t) + t log2(K−1) [C]."""
    K = 2 ** n
    mult = 1.0 if target == "H" else float(n + 1)
    t = _t_for_error(lambda x: mult * h2(x) + x * np.log2(K - 1), eps, K)
    return int(np.ceil((K * np.log(2.0) + np.log(1.0 / delta)) / (2.0 * t ** 2)))


def plugin_cz_bias_uniform(n: int, shots: int) -> float:
    """Sesgo líder del C_Z de plug-in con p uniforme e independiente: (2ⁿ−1−n)/(2N ln 2) > 0 (artefacto estadístico)."""
    return float((2 ** n - 1 - n) / (2.0 * shots * np.log(2.0)))


def plugin_study_uniform(n: int, shots: int, reps: int, seed: int) -> Tuple[float, float]:
    """(media Ĥ_Z, media Ĉ_Z) en bits sobre `reps` histogramas multinomiales con p uniforme."""
    rng = np.random.default_rng(seed)
    K = 2 ** n
    hs, cs = [], []
    for _ in range(reps):
        ph = rng.multinomial(shots, np.full(K, 1.0 / K)) / shots
        hs.append(shannon_bits(ph))
        cs.append(total_correlation_z(ph, n))
    return float(np.mean(hs)), float(np.mean(cs))


def dephased_path3(t: float) -> np.ndarray:
    """ρ_t = (1−t)|G₃⟩⟨G₃| + t·I/8: toda estadística Z queda fija; W′ pasa de −1 a +2."""
    g = path_graph_state(3)
    return (1 - t) * np.outer(g, g.conj()) + t * np.eye(8, dtype=complex) / 8


def product_amp_damped(gamma: float) -> Dict[str, float]:
    """diag(γ,1−γ)⊗diag(γ,1−γ): producto en todo γ. H_Z = 2·h2(γ), z = 2γ−1, C_Z = 0, P = [γ²+(1−γ)²]²."""
    r = np.diag([gamma, 1 - gamma]).astype(complex)
    rho = np.kron(r, r)
    p = np.real(np.diag(rho))
    return {"H_Z": shannon_bits(p), "z": 2 * gamma - 1, "C_Z": total_correlation_z(p, 2), "P": purity(rho)}
