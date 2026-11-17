"""qang.qagent — Cap. 44: agentes híbridos clásico–cuánticos leídos con qang (solo NumPy).

Un agente de dos acciones cuya política es la medida de un qubit: π(0|θ)=cos²(θ/2)=p0, π(1|θ)=sin²(θ/2); qg_Z=cosθ.
Escalas: m-qg = 1000·qg_Z (lectura); θ en radianes (1 vuelta = 1000 qga; π/2 rad = 250 qga). Cinco bloques con verificador
(`tools/verify_cap44.py`):
  A. política de Born y su gradiente exacto (información de Fisher 1 en θ);
  B. flujo del gradiente: la parametrización en θ es el gradiente natural del logit (tiempos cerrados);
  C. el polo y la exploración: gradiente que se anula, varianza finita, piso de exploración;
  D. la lectura de decisión (confianza, entropía, disparos para el signo);
  E. orquestación de herramientas: reparto de disparos entre K lecturas (Neyman) con la información de qang.
Nada de esto es una ventaja medida: son identidades exactas del modelo y presupuestos de recursos.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from qang import probability as _pr

__all__ = [
    "born_policy", "exploration_floor", "expected_gradient", "score", "gradient_second_moment", "gradient_variance",
    "episodes_for_gradient_sign", "logit_of", "flow_theta", "time_to_wrong_probability", "logit_vanilla_time",
    "decision_report", "neyman_allocation", "allocation_variance", "uniform_to_neyman_ratio", "run_bandit",
]


# ------------------------------------------------------------------ A. política de Born
def born_policy(theta: float) -> Tuple[float, float]:
    """(π(0), π(1)) = (cos²(θ/2), sin²(θ/2)) del qubit R_Y(θ)|0⟩ medido en Z."""
    p0 = float(np.cos(theta / 2.0) ** 2)
    return p0, 1.0 - p0


def exploration_floor(p_min: float) -> float:
    """θ_min = 2 arcsin(√p_min): con θ ∈ [θ_min, π−θ_min] ninguna acción tiene probabilidad menor que p_min."""
    return float(2.0 * np.arcsin(np.sqrt(p_min)))


def score(action: int, theta: float) -> float:
    """∂θ log π(a|θ): −tan(θ/2) para a=0 y cot(θ/2) para a=1."""
    return float(-np.tan(theta / 2.0)) if action == 0 else float(1.0 / np.tan(theta / 2.0))


def expected_gradient(theta: float, r0: float, r1: float) -> float:
    """∂θ J con J = r0 π(0) + r1 π(1): −(r0−r1) sinθ/2, exacta."""
    return float(-(r0 - r1) * np.sin(theta) / 2.0)


def gradient_second_moment(theta: float, r0: float, r1: float) -> float:
    """E[g²] del estimador REINFORCE g = r_a ∂θ log π(a) con recompensas deterministas: r0² sin²(θ/2) + r1² cos²(θ/2)
    (la información de Fisher de la política es 1 para todo θ)."""
    return float(r0 ** 2 * np.sin(theta / 2.0) ** 2 + r1 ** 2 * np.cos(theta / 2.0) ** 2)


def gradient_variance(theta: float, r0: float, r1: float) -> float:
    """Var(g) = E[g²] − (∂θ J)²."""
    return gradient_second_moment(theta, r0, r1) - expected_gradient(theta, r0, r1) ** 2


def episodes_for_gradient_sign(theta: float, r0: float, r1: float, z: float = 1.96) -> float:
    """Episodios para que el signo del gradiente medio quede a z desviaciones: z²·Var(g)/(∂θ J)². Diverge en los polos
    (el gradiente se anula, la varianza no)."""
    m = expected_gradient(theta, r0, r1)
    if m == 0.0:
        return float("inf")
    return float(z * z * gradient_variance(theta, r0, r1) / (m * m))


# ------------------------------------------------------------------ B. flujo del gradiente
def logit_of(theta: float) -> float:
    """φ = ln(π(0)/π(1)) = 2 ln cot(θ/2)."""
    return float(2.0 * np.log(1.0 / np.tan(theta / 2.0)))


def flow_theta(theta0: float, eta: float, delta: float, t: float) -> float:
    """Flujo del gradiente en θ con paso η y Δ=r0−r1: tan(θ_t/2) = tan(θ0/2)·exp(−ηΔt/2), exacto. Equivale a φ_t = φ0 + ηΔt."""
    return float(2.0 * np.arctan(np.tan(theta0 / 2.0) * np.exp(-eta * delta * t / 2.0)))


def time_to_wrong_probability(theta0: float, eta: float, delta: float, p_wrong: float) -> float:
    """Tiempo del flujo en θ para que π(acción peor) ≤ p_wrong (Δ>0): (logit(p_wrong) − φ0)/(ηΔ)."""
    phi_target = float(np.log((1.0 - p_wrong) / p_wrong))
    return (phi_target - logit_of(theta0)) / (eta * delta)


def logit_vanilla_time(phi0: float, eta: float, delta: float, p_wrong: float) -> float:
    """Tiempo del gradiente plano en el logit (φ̇ = ηΔ π0 π1) hasta π(peor) ≤ p_wrong: [F(φ*) − F(φ0)]/(ηΔ), F(φ)=2 sinhφ + 2φ.
    Crece como 1/p_wrong; el de θ, como ln(1/p_wrong)."""
    phi_t = float(np.log((1.0 - p_wrong) / p_wrong))
    F = lambda x: 2.0 * np.sinh(x) + 2.0 * x
    return float((F(phi_t) - F(phi0)) / (eta * delta))


# ------------------------------------------------------------------ D. lectura de decisión
def decision_report(theta: float, z: float = 3.0) -> Dict[str, float]:
    """Lectura de la decisión con las medidas de qang: qg_Z, m-qg, entropía qg_S (bits), qg_SNR, disparos para el signo a z
    desviaciones y acción voraz (0 si qg_Z>0)."""
    qz = float(np.cos(theta))
    p0 = (1.0 + qz) / 2.0
    h = 0.0 if p0 in (0.0, 1.0) else float(-(p0 * np.log2(p0) + (1 - p0) * np.log2(1 - p0)))
    snr = float(_pr.quantum_snr(qz)) if abs(qz) < 1.0 else float("inf")
    shots = float("inf") if abs(qz) < 1e-12 else (0.0 if abs(qz) == 1.0 else float(_pr.shots_for_sign(qz, z)))
    return {"qg_Z": qz, "m_Z": 1000.0 * qz, "qg_S": h, "qg_SNR": snr, "shots_for_sign": shots,
            "greedy_action": 0.0 if qz > 0 else 1.0}


# ------------------------------------------------------------------ E. orquestación de herramientas
def neyman_allocation(thetas: Sequence[float], budget: float) -> np.ndarray:
    """Reparto de `budget` disparos entre K lecturas qg_Z que minimiza Σ Var(qg_Z,i) = Σ sin²θ_i/n_i: n_i ∝ sinθ_i."""
    s = np.abs(np.sin(np.asarray(thetas, dtype=float)))
    return budget * s / s.sum()


def allocation_variance(thetas: Sequence[float], n: Sequence[float]) -> float:
    """Σ_i sin²θ_i / n_i, la suma de varianzas de las K lecturas qg_Z."""
    th, n = np.asarray(thetas, dtype=float), np.asarray(n, dtype=float)
    return float(np.sum(np.sin(th) ** 2 / n))


def uniform_to_neyman_ratio(thetas: Sequence[float]) -> float:
    """Cociente de varianzas reparto uniforme / óptimo: K·Σ sin²θ_i / (Σ sinθ_i)² ≥ 1 (Cauchy–Schwarz), igual a 1 si todos
    los sinθ_i coinciden."""
    s = np.abs(np.sin(np.asarray(thetas, dtype=float)))
    return float(len(s) * np.sum(s ** 2) / np.sum(s) ** 2)


# ------------------------------------------------------------------ el agente
def run_bandit(theta0: float, means: Tuple[float, float], eta: float, steps: int, rng: np.random.Generator,
               p_floor: Optional[float] = None) -> np.ndarray:
    """Agente de dos acciones con política de Born, REINFORCE sin línea base y recompensas Bernoulli de medias `means`.
    En cada paso: la acción es un disparo del qubit (p0 = cos²(θ/2)), θ ← θ + η·R·∂θ log π(a). Con `p_floor`, θ se recorta a
    [θ_min, π−θ_min]. Devuelve la trayectoria de θ (steps+1 valores)."""
    lo = exploration_floor(p_floor) if p_floor else 1e-9
    th = float(theta0)
    path = [th]
    for _ in range(steps):
        a = 0 if rng.random() < np.cos(th / 2.0) ** 2 else 1
        r = 1.0 if rng.random() < means[a] else 0.0
        th = float(np.clip(th + eta * r * score(a, th), lo, np.pi - lo))
        path.append(th)
    return np.array(path)
