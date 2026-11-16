"""qang.qtimeloop — Cap. 41: viajes en el tiempo (curvas temporales cerradas simuladas como mapas) leídos con qang: balance de
información de la postselección, la paradoja del abuelo y puntos fijos de Deutsch por ángulo en qga. Sólo NumPy.
qga: 1 vuelta = 1000 qga. m-qg = 1000·qg. Se simulan mapas; la existencia física de CTC es `unverified`.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


# ------------------------------------------------------------------ postselección con C = 2·diag(1, a)
def pctc_p_post(a: float, theta: float) -> float:
    """p_post = cos²(θ/2) + a² sin²(θ/2) para C = 2·diag(1, a) (a = cos(α/2) si U = |0⟩⟨0|⊗I + |1⟩⟨1|⊗R_y(α))."""
    return math.cos(theta / 2) ** 2 + a * a * math.sin(theta / 2) ** 2


def pctc_theta_out(a: float, theta: float) -> float:
    """Ángulo polar tras el mapa: tan(θ'/2) = a·tan(θ/2)."""
    return 2.0 * math.atan2(a * math.sin(theta / 2), math.cos(theta / 2))


def pctc_amplification(a: float, theta: float) -> float:
    """dθ'/dθ = a / (cos²(θ/2) + a² sin²(θ/2)); vale 1/a en el polo sur θ = π."""
    return a / pctc_p_post(a, theta)


def pctc_fisher(a: float, theta: float) -> Tuple[float, float, float, float]:
    """(información de Fisher por disparo aceptado, p_post, Fisher de la tasa de éxito, Fisher total por copia de entrada).
    Para estados puros F_acc = (dθ'/dθ)²; el total es p·F_acc + (dp/dθ)²/(p(1−p)) y vale exactamente 1."""
    p = pctc_p_post(a, theta)
    f_acc = pctc_amplification(a, theta) ** 2
    dp = -(1.0 - a * a) * math.sin(theta) / 2.0
    f_bern = dp * dp / (p * (1.0 - p)) if 0.0 < p < 1.0 else 0.0
    return f_acc, p, f_bern, p * f_acc + f_bern


# ------------------------------------------------------------------ paradoja del abuelo y puntos fijos de Deutsch
def ptrace(rho: np.ndarray, keep: int) -> np.ndarray:
    r = rho.reshape(2, 2, 2, 2)
    return np.einsum("ijkj->ik", r) if keep == 0 else np.einsum("ijil->jl", r)


def pctc_C(U: np.ndarray) -> np.ndarray:
    """C = Tr_CTC U (qubit 0 = CR, qubit 1 = CTC)."""
    u = U.reshape(2, 2, 2, 2)
    return np.einsum("ajbj->ab", u)


def pctc_success(U: np.ndarray, rho: np.ndarray) -> float:
    """p_post = Tr(CρC†)/4."""
    C = pctc_C(U)
    return float(np.real(np.trace(C @ rho @ C.conj().T)) / 4.0)


def deutsch_iterate(U: np.ndarray, rho: np.ndarray, iters: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Punto fijo de ρ_CTC ← Tr_CR[U(ρ⊗ρ_CTC)U†] desde I/2; devuelve (ρ_CTC, ρ_CR de salida)."""
    c = I2 / 2
    for _ in range(iters):
        c = ptrace(U @ np.kron(rho, c) @ U.conj().T, keep=1)
    return c, ptrace(U @ np.kron(rho, c) @ U.conj().T, keep=0)


def bloch_mqg(rho: np.ndarray) -> np.ndarray:
    return np.array([1000.0 * float(np.real(np.trace(rho @ s))) for s in (SX, SY, SZ)])


def bloch_rotation(axis, alpha_qga: float) -> np.ndarray:
    """Matriz 3×3 de la rotación de ángulo alpha_qga (1000 qga = una vuelta) alrededor de `axis` (Rodrigues)."""
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    a = 2.0 * math.pi * alpha_qga / 1000.0
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)


def fixed_dimension(R: np.ndarray, tol: float = 1e-9) -> int:
    """Dimensión del conjunto de vectores de Bloch fijos por R (núcleo de R − I)."""
    s = np.linalg.svd(R - np.eye(3), compute_uv=False)
    return int(np.sum(s < tol))


def shots_for_successes(n_success: float, p_post: float) -> float:
    return n_success / p_post
