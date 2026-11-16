"""qang.qcft — Cap. 33: alcance de las medidas qang frente a una CFT de 1+1 dimensiones (cadena XX = bosón libre, c = 1,
el análogo de la hoja de mundo de la cuerda bosónica) y límites frente a la teoría de cuerdas. NumPy exacto.

Cadena abierta de n sitios, H = −Σ (c†_i c_{i+1} + h.c.), N = n/2 fermiones (≡ cadena XX por Jordan–Wigner).
Convención: bit 1 = ocupado ⇒ qg_Z,i = 1 − 2⟨n_i⟩; m-qg = 1000·qg.
"""
from __future__ import annotations

import numpy as np


def correlation_matrix(n: int, N: int | None = None) -> np.ndarray:
    """C_ij = ⟨c†_i c_j⟩ del estado fundamental (determinante de Slater de los N autoestados más bajos)."""
    N = n // 2 if N is None else N
    h = -(np.eye(n, k=1) + np.eye(n, k=-1))
    w, V = np.linalg.eigh(h)
    occ = V[:, :N]
    return occ @ occ.T


def local_qgz(C: np.ndarray) -> np.ndarray:
    """qg_Z,i = 1 − 2 C_ii (adimensional; ×1000 para m-qg)."""
    return 1.0 - 2.0 * np.diag(C)


def connected_qgzz(C: np.ndarray, i: int, j: int) -> float:
    """⟨Z_i Z_j⟩ − ⟨Z_i⟩⟨Z_j⟩ = −4 C_ij² (Wick, C real)."""
    return float(-4.0 * C[i, j] ** 2)


def block_entropy_bits(C: np.ndarray, l: int) -> float:
    """Entropía de von Neumann (bits) de los primeros l sitios, a partir de los autovalores de C_A."""
    nu = np.clip(np.linalg.eigvalsh(C[:l, :l]), 1e-15, 1 - 1e-15)
    return float(-np.sum(nu * np.log2(nu) + (1 - nu) * np.log2(1 - nu)))


def local_qgs_bits(C: np.ndarray, i: int) -> float:
    """qg_S de un sitio = entropía binaria de p0 = (1 + qg_Z)/2 (bits)."""
    p = float(np.clip((1.0 + local_qgz(C)[i]) / 2.0, 1e-15, 1 - 1e-15))
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def fit_central_charge(n: int, ls=None) -> tuple:
    """Ajusta S(l) = (c/6) log2[(2n/π) sin(π l/n)] + s0 (cadena abierta, bloque en el borde) con l par entre n/8 y n/2 (los l chicos arrastran correcciones de red)."""
    C = correlation_matrix(n)
    ls = np.arange(2 * (n // 16), n // 2 + 1, 2) if ls is None else np.asarray(ls)
    x = np.log2((2 * n / np.pi) * np.sin(np.pi * ls / n))
    S = np.array([block_entropy_bits(C, int(l)) for l in ls])
    A = np.vstack([x, np.ones_like(x)]).T
    (a, s0), *_ = np.linalg.lstsq(A, S, rcond=None)
    return float(6.0 * a), float(s0)


def zz_power_law_exponent(n: int = 200, rmin: int = 3, rmax: int = 31) -> float:
    """Pendiente log-log de |⟨ZZ⟩_c| en función de la distancia r impar, desde el centro de una cadena abierta."""
    C = correlation_matrix(n)
    i0 = n // 2 - 16
    rs = np.arange(rmin, rmax + 1, 2)
    y = np.array([abs(connected_qgzz(C, i0, i0 + int(r))) for r in rs])
    return float(np.polyfit(np.log(rs), np.log(y), 1)[0])


# ------------------------------------------------------------------ comprobación contra el estado completo (n pequeño)
def exact_state_xx(n: int, N: int | None = None) -> np.ndarray:
    """Estado fundamental de la cadena XX (H = −½ Σ (XX+YY) ≡ fermiones libres) en el sector de peso N, vector de 2^n."""
    N = n // 2 if N is None else N
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]])
    I = np.eye(2)

    def op(P, a, Q=None, b=None):
        ops = [I] * n
        ops[a] = P
        if Q is not None:
            ops[b] = Q
        M = np.array([[1.0 + 0j]])
        for o in ops:
            M = np.kron(M, o)
        return M

    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for k in range(n - 1):
        H += -0.5 * (op(X, k, X, k + 1) + op(Y, k, Y, k + 1))
    wts = np.array([bin(z).count("1") for z in range(2 ** n)])
    idx = np.where(wts == N)[0]
    w, V = np.linalg.eigh(H[np.ix_(idx, idx)])
    psi = np.zeros(2 ** n, dtype=complex)
    psi[idx] = V[:, 0]
    return psi


def spin_zz_connected(psi: np.ndarray, n: int, i: int, j: int) -> float:
    z = np.array([(-1.0) ** bin(s).count("1") for s in range(2 ** n)])  # placeholder para Z total
    def zi(a):
        return np.array([1.0 - 2.0 * ((s >> (n - 1 - a)) & 1) for s in range(2 ** n)])
    p = np.abs(psi) ** 2
    return float(np.sum(p * zi(i) * zi(j)) - np.sum(p * zi(i)) * np.sum(p * zi(j)))


def spin_block_entropy_bits(psi: np.ndarray, n: int, l: int) -> float:
    M = psi.reshape(2 ** l, 2 ** (n - l))
    s = np.linalg.svd(M, compute_uv=False) ** 2
    s = s[s > 1e-15]
    return float(-np.sum(s * np.log2(s)))
