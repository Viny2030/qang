"""qang.qdynamics — Dinámica XY/XXZ en el sector de peso 1 (caminata cuántica), Trotter con RBS,
canales afines de Bloch (`qang.channels`) y evolución en tiempo imaginario variacional de 1 qubit (varQTE).
Backend JAX (jax<0.9). (Cap. 24 antiguo)

XY: H = J Σ_k (X_k X_{k+1} + Y_k Y_{k+1})/2 conserva el peso; en el sector de peso 1 es la adyacencia J·A_n
(caminata cuántica continua). exp(−i t J (XY+YX)/2) sobre un par = RBS(J t) (Givens) ⇒ Trotter = capas de RBS.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from qang.qweightqnn import _rbs, _apply2, RAD_PER_QGA

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------- XY en el sector de peso 1
def xy_sector_hamiltonian(n: int, J: float = 1.0, periodic: bool = False) -> jnp.ndarray:
    A = np.zeros((n, n))
    for k in range(n - 1):
        A[k, k + 1] = A[k + 1, k] = J
    if periodic:
        A[0, n - 1] = A[n - 1, 0] = J
    return jnp.asarray(A)


def exact_walk(n: int, t: float, start: int = 0, J: float = 1.0) -> jnp.ndarray:
    """p_k(t) = |<e_k| e^{−iHt} |e_start>|² (exact, diagonalización)."""
    H = xy_sector_hamiltonian(n, J)
    w, V = jnp.linalg.eigh(H)
    U = V @ jnp.diag(jnp.exp(-1j * w * t)) @ jnp.conj(V).T
    return jnp.abs(U[:, start]) ** 2


def trotter_walk_rbs(n: int, t: float, steps: int, start: int = 0, J: float = 1.0) -> Tuple[jnp.ndarray, float]:
    """Trotter par-impar con RBS(J dt) por enlace; devuelve (p_k, ángulo RBS por paso en qga)."""
    dt = t / steps
    ang = J * dt                                   # rad; signo: RBS(θ) = exp(−iθ(XY+YX)/2) con la convención de _rbs
    psi = jnp.zeros(2 ** n, dtype=jnp.complex128).at[1 << (n - 1 - start)].set(1.0)
    even = [(k, k + 1) for k in range(0, n - 1, 2)]
    odd = [(k, k + 1) for k in range(1, n - 1, 2)]
    for _ in range(steps):
        for (i, j) in even:
            psi = _apply2(psi, _rbs(-ang), i, j, n)
        for (i, j) in odd:
            psi = _apply2(psi, _rbs(-ang), i, j, n)
    e = np.array([1 << (n - 1 - k) for k in range(n)])
    return jnp.abs(psi[e]) ** 2, float(ang / RAD_PER_QGA)


def walk_qz(p: jnp.ndarray) -> jnp.ndarray:
    """Lectura por sitio: qg_Z^{(k)} = 1 − 2 p_k (un solo excitado)."""
    return 1.0 - 2.0 * p


# ---------------------------------------------------------------- canales afines de Bloch
class AffineChannel:
    """r ↦ M r + c. Composición, aplicación y déficit disipativo qg_τ = √(1 − |r'|²)."""
    def __init__(self, M, c):
        self.M = jnp.asarray(M, dtype=jnp.float64); self.c = jnp.asarray(c, dtype=jnp.float64)

    def __call__(self, r):
        return self.M @ jnp.asarray(r) + self.c

    def then(self, other: "AffineChannel") -> "AffineChannel":
        return AffineChannel(other.M @ self.M, other.M @ self.c + other.c)

    @staticmethod
    def depolarizing(V):
        return AffineChannel(V * jnp.eye(3), jnp.zeros(3))

    @staticmethod
    def amplitude_damping(g):
        return AffineChannel(jnp.diag(jnp.array([jnp.sqrt(1 - g), jnp.sqrt(1 - g), 1 - g])), jnp.array([0.0, 0.0, g]))

    @staticmethod
    def dephasing(p):
        return AffineChannel(jnp.diag(jnp.array([1 - 2 * p, 1 - 2 * p, 1.0])), jnp.zeros(3))

    @staticmethod
    def readout(e01, e10):
        """Sólo afecta a la lectura Z (Cap. 25): qg_Z ↦ (1−e01−e10) qg_Z + (e10−e01)."""
        return AffineChannel(jnp.diag(jnp.array([1.0, 1.0, 1 - e01 - e10])), jnp.array([0.0, 0.0, e10 - e01]))


def dissipative_deficit(r):
    return jnp.sqrt(jnp.clip(1.0 - jnp.sum(jnp.asarray(r) ** 2), 0.0))


# ---------------------------------------------------------------- varQTE de 1 qubit
def imaginary_time_qz_exact(omega: float, tau: jnp.ndarray) -> jnp.ndarray:
    """H = ω Z/2 desde |+>: ρ(τ) ∝ e^{−τH}|+><+|e^{−τH} ⇒ qg_Z(τ) = −tanh(ωτ)  (exact)."""
    return -jnp.tanh(omega * tau)


def varqte_ry_mclachlan(omega: float, tau_max: float, steps: int) -> jnp.ndarray:
    """Ansatz |ψ(θ)> = RY(θ)|0>, θ en qga; McLachlan: g θ̇ = −Re<∂θψ|H|ψ> + <H><∂θψ|ψ>.
    Con g = (2π/1000)²/4 (métrica FS en qga) y H = ωZ/2: θ̇_rad = ω sinθ ⇒ integra con Euler explícito."""
    th = jnp.array(jnp.pi / 2)                      # |+>
    dtau = tau_max / steps
    g = 1.0 / 4.0                                   # <∂θψ|∂θψ> − |<ψ|∂θψ>|² en rad
    out = []
    for _ in range(steps):
        # C = −Re<∂θψ|H|ψ> + <H><∂θψ|ψ> ; para RY(θ)|0>: <H> = ω cosθ/2, Re<∂θψ|H|ψ> = −ω sinθ/4
        C = omega * jnp.sin(th) / 4.0
        th = th + dtau * C / g
        out.append(jnp.cos(th))
    return jnp.stack(out)
