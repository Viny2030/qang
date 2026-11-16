"""qang.qanti — Cap. 40: materia y antimateria leídas con qang, sobre un qubit de Dirac (c = 1) y sobre pares. Sólo NumPy.
m-qg = 1000·qg (lectura); qga = ángulo, 1 vuelta = 1000 qga. Complementa al Apéndice G (ahí: ⟨β⟩ = cos θ_r y CPT en cuatro componentes).
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def dirac_h(p: float, m: float) -> np.ndarray:
    """H = p σx + m σz (1+1 dimensiones, c = 1)."""
    return p * SX + m * SZ


def eigenstate(p: float, m: float, sign: int = +1) -> np.ndarray:
    """Autoestado de energía sign·E, E = sqrt(p² + m²)."""
    w, v = np.linalg.eigh(dirac_h(p, m))
    return v[:, 1 if sign > 0 else 0]


def bloch_mqg(psi: np.ndarray) -> np.ndarray:
    """(qg_X, qg_Y, qg_Z) en m-qg."""
    psi = np.asarray(psi, dtype=complex)
    return np.array([1000.0 * float(np.real(np.vdot(psi, s @ psi))) for s in (SX, SY, SZ)])


def conjugate(psi: np.ndarray) -> np.ndarray:
    """Conjugación de carga del qubit de Dirac: ψ ↦ σy ψ* (antiunitaria; lleva H a −H)."""
    return SY @ np.conj(np.asarray(psi, dtype=complex))


def self_conjugate_overlap(psi: np.ndarray) -> float:
    """|⟨ψ|σy ψ*⟩|: 0 para todo estado puro de un qubit (el mapa antípoda no tiene puntos fijos)."""
    psi = np.asarray(psi, dtype=complex)
    return float(abs(np.vdot(psi, conjugate(psi))))


def gamma_of(p: float, m: float) -> float:
    return math.sqrt(p * p + m * m) / m


def zb_qgz_mqg(p: float, m: float, t: float, psi0: np.ndarray = None) -> float:
    """qg_Z(t) en m-qg de un estado que empieza en |0⟩ (o psi0) bajo H, por diagonalización exacta (Zitterbewegung)."""
    psi0 = np.array([1.0, 0.0], dtype=complex) if psi0 is None else np.asarray(psi0, dtype=complex)
    w, v = np.linalg.eigh(dirac_h(p, m))
    psi = v @ (np.exp(-1j * w * t) * (v.conj().T @ psi0))
    return float(1000.0 * np.real(np.vdot(psi, SZ @ psi)))


def zb_analytic_mqg(p: float, m: float, t: float) -> float:
    """1000·(m²/E² + p²/E² cos 2Et) = 1000·(1/γ² + β² cos 2Et)."""
    E2 = p * p + m * m
    return 1000.0 * (m * m / E2 + (p * p / E2) * math.cos(2.0 * math.sqrt(E2) * t))


def lz_probability(delta: float, v: float) -> float:
    """Landau–Zener: H = (v t/2) σz + (Δ/2) σx; probabilidad de seguir en el estado diabático = exp(−π Δ²/(2 v))."""
    return math.exp(-math.pi * delta * delta / (2.0 * v))


def pair_probability(m: float, eE: float) -> float:
    """Creación de pares por modo en un campo uniforme (c = ħ = 1): exp(−π m²/(eE)) = lz_probability(2m, 2eE)."""
    return math.exp(-math.pi * m * m / eE)


def lz_numeric(delta: float, v: float, T: float, steps: int) -> float:
    """Probabilidad diabática por propagación 2×2 de punto medio desde −T hasta +T partiendo del estado diabático."""
    dt = 2.0 * T / steps
    psi = np.array([1.0, 0.0], dtype=complex)
    for k in range(steps):
        t = -T + (k + 0.5) * dt
        ax, az = delta / 2.0, v * t / 2.0
        a = math.hypot(ax, az)
        c, s = math.cos(a * dt), math.sin(a * dt)
        U = np.array([[c - 1j * s * az / a, -1j * s * ax / a], [-1j * s * ax / a, c + 1j * s * az / a]])
        psi = U @ psi
    return float(abs(psi[0]) ** 2)


def pair_reading_mqg(P: float) -> float:
    """Lectura de la ocupación vacío/par: qg = 1 − 2P, en m-qg."""
    return 1000.0 * (1.0 - 2.0 * P)


def singlet_readings() -> Tuple[Tuple[float, float, float], float]:
    """Par de espín en singlete: lecturas locales (qg_Z del qubit 0 y del 1, en m-qg) y correlación qg_ZZ."""
    psi = np.array([0, 1, -1, 0], dtype=complex) / math.sqrt(2.0)
    z0 = np.real(np.vdot(psi, np.kron(SZ, np.eye(2)) @ psi))
    z1 = np.real(np.vdot(psi, np.kron(np.eye(2), SZ) @ psi))
    zz = np.real(np.vdot(psi, np.kron(SZ, SZ) @ psi))
    return (1000.0 * float(z0), 1000.0 * float(z1), 1000.0 * float(zz)), 1000.0 * float(zz)


def shots_to_certify_branch(gamma: float, z: float = 3.0) -> float:
    """Disparos para que el signo de la lectura ±1000/γ quede a z desviaciones del cero: N = z²(γ² − 1)."""
    return z * z * (gamma * gamma - 1.0)
