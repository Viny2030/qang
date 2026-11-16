"""qang.qchem — Cap. 28: testigo de electrones en m-qg, verificación por número/paridad (Hubbard de 2 sitios, 4 qubits),
PCA en el sector de peso 1 (red de RBS en qga), solapamientos y QAOA con restricción «exactamente K» (mezclador XY).
Backend JAX (jax<0.9) + NumPy para optimización clásica.

Convención JW: qubit q = espín-orbital q; ocupación n_q = (1 − qg_Z,q)/2; 1 vuelta = 1000 qga; m-qg = 1000·qg_Z.
El modelo químico es el dímero de Hubbard (4 qubits, 2 electrones), NO H2 STO-3G: tiene solución analítica
E0 = (U − sqrt(U² + 16 t²))/2 y permite verificar cada afirmación sin integrales externas.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_I = np.eye(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1.0, -1.0]).astype(complex)
_LOW = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1| : aniquila (bit 1 = ocupado)


def _kron(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


# ------------------------------------------------------------------ testigo de electrones
def mean_qz_ideal(n: int, N: int) -> float:
    """Lectura media ideal del registro con N electrones en n espín-orbitales: 1 − 2N/n."""
    return 1.0 - 2.0 * N / n


def electron_count_from_qz(qz) -> float:
    """N_e = Σ (1 − qg_Z,i)/2."""
    return float(jnp.sum((1.0 - jnp.asarray(qz)) / 2.0))


def witness_mqg(qz, N: int) -> float:
    """Testigo en m-qg: 1000·(media(qg_Z) − (1 − 2N/n)); 0 si el circuito conservó el número."""
    qz = jnp.asarray(qz)
    return float(1000.0 * (jnp.mean(qz) - mean_qz_ideal(qz.shape[0], N)))


# ------------------------------------------------------------------ Hubbard de 2 sitios (JW, 4 qubits)
def jw_annihilation(n: int) -> List[np.ndarray]:
    """a_j = Z⊗…⊗Z⊗|0><1|_j⊗I…; qubit 0 = bit más significativo."""
    ops = []
    for j in range(n):
        ops.append(_kron(*([_Z] * j + [_LOW] + [_I] * (n - j - 1))))
    return ops


def hubbard_dimer(t: float = 1.0, U: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (H, H_Z, H_hop, N, Sz). Orbitales: q0=0↑, q1=0↓, q2=1↑, q3=1↓."""
    a = jw_annihilation(4)
    ad = [x.conj().T for x in a]
    hop = np.zeros((16, 16), dtype=complex)
    for s in (0, 1):  # espín: q = site*2 + s
        i, j = s, 2 + s
        hop += -t * (ad[i] @ a[j] + ad[j] @ a[i])
    nop = [ad[q] @ a[q] for q in range(4)]
    HZ = U * (nop[0] @ nop[1] + nop[2] @ nop[3])
    N = sum(nop)
    Sz = 0.5 * (nop[0] - nop[1] + nop[2] - nop[3])
    return HZ + hop, HZ, hop, N, Sz


def ground_state_sector(H: np.ndarray, N: np.ndarray, n_target: int = 2) -> Tuple[float, np.ndarray]:
    """Energía y vector del estado fundamental dentro del sector N = n_target."""
    Nd = np.real(np.diag(N)).round().astype(int)
    idx = np.where(Nd == n_target)[0]
    w, V = np.linalg.eigh(H[np.ix_(idx, idx)])
    psi = np.zeros(H.shape[0], dtype=complex)
    psi[idx] = V[:, 0]
    return float(w[0]), psi


def _kraus_t1_all(g: float, n: int = 4) -> List[np.ndarray]:
    k0 = np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex)  # bit1=ocupado decae
    k1 = np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex)
    out = []
    for bits in range(2 ** n):
        ops = [k1 if (bits >> (n - 1 - q)) & 1 else k0 for q in range(n)]
        out.append(_kron(*ops))
    return out


def apply_t1_all(rho: np.ndarray, g: float, n: int = 4) -> np.ndarray:
    return sum(K @ rho @ K.conj().T for K in _kraus_t1_all(g, n))


def sector_projector(N: np.ndarray, keep: Sequence[int], mod: int = 0) -> np.ndarray:
    """Proyector sobre N ∈ keep (mod=0) o sobre N mod `mod` ∈ keep (paridad: mod=2; N mod 4: mod=4)."""
    Nd = np.real(np.diag(N)).round().astype(int)
    sel = np.isin(Nd % mod if mod else Nd, list(keep))
    return np.diag(sel.astype(float)).astype(complex)


def filtered_energy(rho: np.ndarray, H: np.ndarray, P: np.ndarray | None) -> Tuple[float, float]:
    """(energía, fracción conservada) tras proyectar con P (None = sin filtro)."""
    if P is None:
        return float(np.real(np.trace(rho @ H))), 1.0
    r = P @ rho @ P
    frac = float(np.real(np.trace(r)))
    return float(np.real(np.trace(r @ H)) / frac), frac


def check_costs(n: int) -> dict:
    """CNOT de la verificación por ancilla: paridad = n CNOT; N mod 4 = segundo ancilla (≈ 2n CNOT)."""
    return {"parity_cnot": n, "mod4_cnot": 2 * n, "ancillas_parity": 1, "ancillas_mod4": 2}


# ------------------------------------------------------------------ PCA en el sector de peso 1
def pca_unary(X: np.ndarray, w: np.ndarray | None = None):
    """ρ = Σ w_j x_j x_jᵀ con x_j normalizados (estado mezcla del sector de peso 1); devuelve (λ desc, V)."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    w = np.ones(len(X)) / len(X) if w is None else np.asarray(w) / np.sum(w)
    rho = (Xn * w[:, None]).T @ Xn
    lam, V = np.linalg.eigh(rho)
    o = np.argsort(lam)[::-1]
    return lam[o], V[:, o], rho


def givens_decomposition(V: np.ndarray) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    """V ortogonal (n×n) = D · Π G(i,i+1; θ) con n(n−1)/2 rotaciones adyacentes (RBS) y D = diag(±1).
    Devuelve ([(i, i+1, θ_qga)], D) con G_m…G_1·Vᵀ = D (ver apply_givens)."""
    n = V.shape[0]
    A = V.T.copy()
    rots = []
    for col in range(n - 1):
        for row in range(n - 1, col, -1):
            a, b = A[row - 1, col], A[row, col]
            th = np.arctan2(b, a)
            c, s = np.cos(th), np.sin(th)
            G = np.eye(n)
            G[row - 1, row - 1] = c; G[row - 1, row] = s
            G[row, row - 1] = -s;    G[row, row] = c
            A = G @ A
            rots.append((row - 1, row, float(th * 1000.0 / (2 * np.pi))))
    return rots, np.diag(np.sign(np.diag(A)))


def apply_givens(rho: np.ndarray, rots, D: np.ndarray) -> np.ndarray:
    """Aplica a ρ la rotación Vᵀ = G_1ᵀ…G_mᵀ·D (primero D, luego las RBS en orden inverso con ángulo −θ):
    ρ' = Vᵀ ρ V, diagonal en los autovalores si V diagonaliza ρ."""
    n = rho.shape[0]
    M = D.copy()
    for (i, j, q) in reversed(rots):
        th = -q * 2 * np.pi / 1000.0
        G = np.eye(n)
        G[i, i] = np.cos(th); G[i, j] = np.sin(th)
        G[j, i] = -np.sin(th); G[j, j] = np.cos(th)
        M = G @ M
    return M @ rho @ M.T


# ------------------------------------------------------------------ solapamientos: recursos
def overlap_resources(n: int) -> dict:
    """Test de inversión (unario): n qubits, 0 ancillas, conserva el peso. Swap test: 2n + 1 qubits, 1 ancilla."""
    return {"inversion_qubits": n, "swap_qubits": 2 * n + 1,
            "inversion_ancillas": 0, "swap_ancillas": 1, "inversion_in_sector": True, "swap_in_sector": False}


# ------------------------------------------------------------------ QAOA «exactamente K»
def random_cost(n: int, seed: int = 7) -> np.ndarray:
    """c(x) = Σ a_i x_i + Σ_{i<j} b_ij x_i x_j sobre las 2^n cadenas (bit más significativo = qubit 0)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = np.triu(rng.normal(size=(n, n)), 1)
    c = np.zeros(2 ** n)
    for z in range(2 ** n):
        x = np.array([(z >> (n - 1 - q)) & 1 for q in range(n)], dtype=float)
        c[z] = a @ x + x @ b @ x
    return c


def _weights(n: int) -> np.ndarray:
    return np.array([bin(z).count("1") for z in range(2 ** n)])


def dicke_state(n: int, K: int) -> np.ndarray:
    w = _weights(n)
    psi = (w == K).astype(complex)
    return psi / np.linalg.norm(psi)


def _xy_ring(n: int) -> np.ndarray:
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for q in range(n):
        r = (q + 1) % n
        for P in (_X, _Y):
            ops = [_I] * n
            ops[q] = P
            ops[r] = P
            H += 0.5 * _kron(*ops)
    return H


def _x_mixer(n: int) -> np.ndarray:
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for q in range(n):
        ops = [_I] * n
        ops[q] = _X
        H += _kron(*ops)
    return H


def qaoa_state(c: np.ndarray, n: int, params: np.ndarray, mixer: str, psi0: np.ndarray) -> np.ndarray:
    """p capas exp(−iβ H_M) exp(−iγ C); params = [γ1,β1,γ2,β2,…]."""
    Hm = _xy_ring(n) if mixer == "xy" else _x_mixer(n)
    w, V = np.linalg.eigh(Hm)
    psi = psi0.copy()
    for k in range(len(params) // 2):
        g, b = params[2 * k], params[2 * k + 1]
        psi = np.exp(-1j * g * c) * psi
        psi = V @ (np.exp(-1j * b * w) * (V.conj().T @ psi))
    return psi


def qaoa_optimize(c: np.ndarray, n: int, K: int, p: int, mixer: str, penalty: float = 0.0,
                  restarts: int = 12, seed: int = 0):
    """Minimiza ⟨c + μ(w−K)²⟩ (μ=penalty sólo para el mezclador X desde |+>^n). Devuelve (params, estado)."""
    from scipy.optimize import minimize
    w = _weights(n)
    cost = c + penalty * (w - K) ** 2
    psi0 = dicke_state(n, K) if mixer == "xy" else np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        x0 = rng.uniform(0, np.pi, size=2 * p)
        r = minimize(lambda x: float(np.real(np.vdot(qaoa_state(cost, n, x, mixer, psi0),
                                                     cost * qaoa_state(cost, n, x, mixer, psi0)))),
                     x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, qaoa_state(cost, n, best.x, mixer, psi0)


def feasible_mass(psi: np.ndarray, n: int, K: int) -> float:
    return float(np.sum(np.abs(psi[_weights(n) == K]) ** 2))


def approx_ratio(psi: np.ndarray, c: np.ndarray, n: int, K: int) -> float:
    """(c_max − ⟨c⟩_factible)/(c_max − c_min) dentro del sector factible (1 = óptimo); ⟨c⟩ condicionado a w=K."""
    m = _weights(n) == K
    p = np.abs(psi[m]) ** 2
    p = p / p.sum()
    cs = c[m]
    return float((cs.max() - p @ cs) / (cs.max() - cs.min()))


def greedy_ratio(c: np.ndarray, n: int, K: int) -> float:
    """Voraz de intercambio (1 pasada de mejora local desde 'primeros K') contra el sector factible."""
    m = np.where(_weights(n) == K)[0]
    cs = c[m]
    cur = int(m[0])
    improved = True
    while improved:
        improved = False
        for q1 in range(n):
            for q2 in range(n):
                b1, b2 = (cur >> (n - 1 - q1)) & 1, (cur >> (n - 1 - q2)) & 1
                if b1 == 1 and b2 == 0:
                    nxt = cur ^ (1 << (n - 1 - q1)) ^ (1 << (n - 1 - q2))
                    if c[nxt] < c[cur]:
                        cur, improved = nxt, True
    return float((cs.max() - c[cur]) / (cs.max() - cs.min()))
