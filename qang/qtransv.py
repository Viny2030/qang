"""qang.qtransv — Cap. 38: aplicaciones transversales leídas con qang (fermiones y Jordan–Wigner, eco de Hahn, Ising transversal,
Aharonov–Bohm, rotación de Wigner, Fresnel, vacío de Unruh). NumPy exacto. m-qg = 1000·qg; 1 vuelta = 1000 qga.
Una fase (rotación alrededor de Z) no cambia qg_Z: se lee en qg_X o qg_Y.
"""
from __future__ import annotations

import math
from functools import reduce
from typing import List, Tuple

import numpy as np

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], float)
Y = np.array([[0, -1j], [1j, 0]])
Zm = np.diag([1.0, -1.0])
SP = np.array([[0, 1], [0, 0]], float)   # |0> -> estado vacío; σ⁺ lleva |0>→|1> con la convención de abajo


def _kron(*ops):
    return reduce(np.kron, ops)


# ------------------------------------------------------------------ Jordan–Wigner y el signo del intercambio
def jw_creation(n: int) -> List[np.ndarray]:
    """a_j† = (∏_{k<j} Z_k) σ⁺_j con σ⁺ = |1><0| (el qubit en |1> es un modo ocupado)."""
    sigp = np.array([[0, 0], [1, 0]], float)
    out = []
    for j in range(n):
        out.append(_kron(*([Zm] * j + [sigp] + [I2] * (n - j - 1))))
    return out


def anticommutation_error(n: int = 4) -> float:
    cr = jw_creation(n)
    an = [c.T for c in cr]
    e = 0.0
    for i in range(n):
        for j in range(n):
            e = max(e, np.abs(an[i] @ cr[j] + cr[j] @ an[i] - (np.eye(2 ** n) if i == j else 0)).max())
            e = max(e, np.abs(an[i] @ an[j] + an[j] @ an[i]).max())
    return float(e)


def exchange_qgx(fermionic: bool) -> float:
    """qg_X del auxiliar de un test de Hadamard que controla el intercambio de los modos 0 y 1 ocupados."""
    cr = jw_creation(2)
    vac = np.zeros(4); vac[0] = 1.0
    psi = cr[0] @ cr[1] @ vac
    SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    CZ = np.diag([1.0, 1.0, 1.0, -1.0])
    U = SWAP @ CZ if fermionic else SWAP
    return float(np.real(psi.conj() @ U @ psi))      # Hadamard test: ⟨X_aux⟩ = Re⟨ψ|U|ψ⟩


# ------------------------------------------------------------------ Larmor y eco de Hahn (analítico)
def _rot(axis: np.ndarray, ang: float) -> np.ndarray:
    return math.cos(ang / 2) * I2 - 1j * math.sin(ang / 2) * axis


def hahn_echo_sim(delta_tau: float, eps: float = 0.0, delta_tau2: float = None) -> float:
    """⟨X⟩ tras R_y(π/2) − (fase δτ) − R_x(π(1+ε)) − (fase δτ₂) para un espín con desintonía δ (2×2 exacto)."""
    d2 = delta_tau if delta_tau2 is None else delta_tau2
    psi = np.array([1.0, 0.0], complex)
    psi = _rot(Y, math.pi / 2) @ psi
    psi = _rot(Zm, delta_tau) @ psi
    psi = _rot(X, math.pi * (1 + eps)) @ psi
    psi = _rot(Zm, d2) @ psi
    return float(np.real(psi.conj() @ X @ psi))


def hahn_echo_ensemble(sigma_tau: float, eps: float = 0.0, nodes: int = 120) -> float:
    """Promedio gaussiano (cuadratura de Gauss–Hermite) de hahn_echo_sim con desintonía estática δτ = σ τ ξ."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    vals = np.array([hahn_echo_sim(sigma_tau * xi, eps) for xi in x])
    return float((w * vals).sum() / w.sum())


def hahn_echo_qgx_drift(sigma_tau: float, drift_frac: float) -> float:
    """⟨cos Δ⟩ de un eco cuya desintonía cambia drift_frac·σ entre las dos mitades: exp(−(drift_frac·σ·τ)²/2)."""
    return math.exp(-0.5 * (drift_frac * sigma_tau) ** 2)


def free_decay_qgx(sigma_tau_total: float) -> float:
    """⟨cos(δ t)⟩ de una desintonía gaussiana: exp(−(σ t)²/2)."""
    return math.exp(-0.5 * sigma_tau_total ** 2)


# ------------------------------------------------------------------ Ising con campo transverso (diagonalización exacta)
def tfim_hamiltonian(n: int, g: float, j: float = 1.0) -> np.ndarray:
    N = 2 ** n
    H = np.zeros((N, N))
    idx = np.arange(N)
    for i in range(n - 1):
        zi = 1 - 2 * ((idx >> (n - 1 - i)) & 1)
        zj = 1 - 2 * ((idx >> (n - 2 - i)) & 1)
        H[idx, idx] += -j * zi * zj
    for i in range(n):
        H[idx ^ (1 << (n - 1 - i)), idx] += -g
    return H


def tfim_mx_mqg(n: int, g: float = 1.0) -> float:
    """m_X = 1000·⟨X⟩ medio en el estado fundamental (g/J = g)."""
    H = tfim_hamiltonian(n, g)
    w, v = np.linalg.eigh(H)
    psi = v[:, 0]
    idx = np.arange(2 ** n)
    mx = np.mean([psi @ psi[idx ^ (1 << (n - 1 - i))] for i in range(n)])
    return float(1000.0 * mx)


# ------------------------------------------------------------------ Aharonov–Bohm
def ab_qgz(phi_over_phi0):
    """p0 = cos²(πΦ/Φ0) ⇒ qg_Z = cos(2πΦ/Φ0)."""
    return np.cos(2.0 * np.pi * np.asarray(phi_over_phi0, float))


# ------------------------------------------------------------------ Wigner
def boost(v) -> np.ndarray:
    v = np.asarray(v, float)
    b = np.linalg.norm(v)
    g = 1.0 / math.sqrt(1.0 - b * b)
    L = np.eye(4)
    L[0, 0] = g
    if b > 0:
        n = v / b
        L[0, 1:] = L[1:, 0] = g * v
        L[1:, 1:] += (g - 1.0) * np.outer(n, n)
    return L


def wigner_rotation(v1, v2) -> Tuple[float, float]:
    """(Ω numérico por descomposición polar de Λ = B2·B1, Ω de la fórmula cerrada) en radianes."""
    Lam = boost(v2) @ boost(v1)
    P2 = Lam @ Lam.T
    w, U = np.linalg.eigh(P2)
    P = U @ np.diag(np.sqrt(w)) @ U.T
    R = np.linalg.inv(P) @ Lam
    num = math.acos(max(-1.0, min(1.0, (np.trace(R[1:, 1:]) - 1.0) / 2.0)))
    g1, g2 = 1 / math.sqrt(1 - np.dot(v1, v1)), 1 / math.sqrt(1 - np.dot(v2, v2))
    g = g1 * g2 * (1 + float(np.dot(v1, v2)))
    cosO = (1 + g + g1 + g2) ** 2 / ((1 + g) * (1 + g1) * (1 + g2)) - 1
    return num, math.acos(max(-1.0, min(1.0, cosO)))


# ------------------------------------------------------------------ Fresnel
def fresnel(n1: float, n2: float, ti: float):
    """(r_s, r_p) complejos para incidencia ti (rad), con reflexión total si sin tt > 1."""
    st = n1 * math.sin(ti) / n2
    ct = np.sqrt(1.0 - st ** 2 + 0j)
    ci = math.cos(ti)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
    return rs, rp


def tir_phase_difference(n: float, ti: float) -> float:
    """δ_p − δ_s (rad) en reflexión total desde el vidrio (n = n_vidrio/n_aire)."""
    rs, rp = fresnel(n, 1.0, ti)
    return float(np.angle(rp) - np.angle(rs))


# ------------------------------------------------------------------ Bogoliubov / Unruh
def vacuum_weight(r: float) -> float:
    return 1.0 / math.cosh(r) ** 2


def gudermannian(r: float) -> float:
    return math.atan(math.sinh(r))


def thermal_entropy_nats(r: float) -> float:
    c2, s2 = math.cosh(r) ** 2, math.sinh(r) ** 2
    return c2 * math.log(c2) - (s2 * math.log(s2) if s2 > 0 else 0.0)


def thermal_entropy_sum(r: float, nmax: int = 80) -> float:
    q = math.tanh(r) ** 2
    n = np.arange(nmax)
    p = (1 - q) * q ** n
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())
