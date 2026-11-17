"""qang.qrobot — Cap. 43: qué le da y qué no le da qang a un robot (solo NumPy).

Un qubit por articulación: la articulación de ángulo θ se prepara como R_Y(θ)|0⟩, de modo que p0 = cos²(θ/2),
qg_Z = cosθ, qg_X = sinθ. Escalas: m-qg = 1000·qg_Z (lectura); el ángulo se da en radianes (1 vuelta = 2π rad = 1000 qga).
Seis bloques, todos con verificador (`tools/verify_cap43.py`):
  A. cinemática plana (directa, jacobiana, inversa cerrada de 2 eslabones) — clásica, de referencia;
  B. la lectura de una articulación: varianza exacta a primer orden, cota de Cramér–Rao y trampa del signo;
  C. de la articulación al efector: tolerancia → disparos;
  D. presupuesto de tiempo real: disparos por ciclo de control y error resultante;
  E. seguimiento de una articulación que deriva (Kalman escalar, varianza estacionaria cerrada);
  F. selección de trayectorias con Grover: consultas cuánticas frente a clásicas (modelo de oráculo).
Nada de esto es una ventaja medida: son identidades exactas del modelo y presupuestos de recursos.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "planar_fk", "planar_jacobian", "two_link_ik",
    "joint_readings", "joint_from_readings", "joint_variance_z", "joint_variance_atan2", "joint_crb_two_bases",
    "joint_sigma_qga", "joint_parameter_shift", "shots_for_joint", "end_effector_rms", "shots_for_end_effector",
    "shots_per_cycle", "realtime_joint_sigma", "max_control_rate",
    "steady_state_variance", "kalman_scalar_track",
    "preflight", "grover_angle", "grover_optimal_iterations", "grover_success", "grover_simulate",
    "classical_expected_queries", "grover_query_ratio", "grover_break_even_overhead",
]


# ------------------------------------------------------------------ A. cinemática plana (referencia clásica)
def planar_fk(thetas: Sequence[float], links: Sequence[float]) -> np.ndarray:
    """Posición (x, y) del efector de un brazo plano; los ángulos son relativos (cada uno respecto del eslabón previo)."""
    th = np.cumsum(np.asarray(thetas, dtype=float))
    L = np.asarray(links, dtype=float)
    return np.array([np.sum(L * np.cos(th)), np.sum(L * np.sin(th))])


def planar_jacobian(thetas: Sequence[float], links: Sequence[float]) -> np.ndarray:
    """Jacobiana 2×n de `planar_fk` respecto de los ángulos relativos."""
    th = np.cumsum(np.asarray(thetas, dtype=float))
    L = np.asarray(links, dtype=float)
    n = len(L)
    J = np.zeros((2, n))
    for j in range(n):
        J[0, j] = -np.sum(L[j:] * np.sin(th[j:]))
        J[1, j] = np.sum(L[j:] * np.cos(th[j:]))
    return J


def two_link_ik(target: Sequence[float], l1: float, l2: float, elbow: int = 1) -> Optional[Tuple[float, float]]:
    """Cinemática inversa cerrada de un brazo plano de 2 eslabones: (θ1, θ2) relativos, o None si el blanco no se alcanza.
    `elbow` = +1 / −1 elige la rama (θ2 ≥ 0 / ≤ 0)."""
    x, y = float(target[0]), float(target[1])
    c2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if abs(c2) > 1.0 + 1e-12:
        return None
    c2 = float(np.clip(c2, -1.0, 1.0))
    t2 = elbow * float(np.arccos(c2))
    t1 = float(np.arctan2(y, x) - np.arctan2(l2 * np.sin(t2), l1 + l2 * np.cos(t2)))
    return t1, t2


# ------------------------------------------------------------------ B. la lectura de una articulación
def joint_readings(theta: float) -> Dict[str, float]:
    """Lecturas exactas del qubit R_Y(θ)|0⟩: p0, qg_Z = cosθ, qg_X = sinθ, y las lecturas en m-qg."""
    return {"p0": float(np.cos(theta / 2) ** 2), "qg_Z": float(np.cos(theta)), "qg_X": float(np.sin(theta)),
            "m_Z": float(1000 * np.cos(theta)), "m_X": float(1000 * np.sin(theta))}


def joint_from_readings(qg_z: float, qg_x: float) -> float:
    """θ = atan2(qg_X, qg_Z) ∈ (−π, π]: con las dos bases el signo se recupera; con qg_Z solo, no (θ y −θ dan lo mismo)."""
    return float(np.arctan2(qg_x, qg_z))


def joint_variance_z(n_shots: int) -> float:
    """Varianza de θ̂ = arccos(qg_Z-hat) a primer orden con N disparos en Z: sin²θ/(N sin²θ) = 1/N, exacta para todo θ∈(0,π).
    Es también la cota de Cramér–Rao de ese experimento (la información de Fisher en Z es N, constante)."""
    return 1.0 / n_shots


def joint_variance_atan2(theta: float, n_shots: int) -> float:
    """Varianza a primer orden de θ̂ = atan2(qg_X-hat, qg_Z-hat) con N disparos en cada base: (cos⁴θ + sin⁴θ)/N."""
    return float((np.cos(theta) ** 4 + np.sin(theta) ** 4) / n_shots)


def joint_crb_two_bases(n_shots: int) -> float:
    """Cota de Cramér–Rao de θ con N disparos en Z y N en X: 1/(2N) (la información de Fisher suma N + N)."""
    return 1.0 / (2.0 * n_shots)


def joint_sigma_qga(n_shots: int) -> float:
    """Desviación del ángulo de una articulación en qga con N disparos en Z: (1000/2π)/√N = 159.15/√N."""
    return 1000.0 / (2.0 * np.pi) / np.sqrt(n_shots)


def joint_parameter_shift(theta: float) -> float:
    """∂qg_Z/∂θ por la regla del desplazamiento de parámetro, [qg_Z(θ+π/2) − qg_Z(θ−π/2)]/2 = −sinθ. El desplazamiento de π/2 rad
    son 250 qga. Se optimiza en θ (o qga), no en qg_Z: el gradiente plano en qg_Z diverge en los polos (Cap. 34)."""
    return float((np.cos(theta + np.pi / 2) - np.cos(theta - np.pi / 2)) / 2.0)


def shots_for_joint(eps_rad: float, z: float = 1.96) -> int:
    """Disparos en Z para que z·σ_θ ≤ ε (ε en rad): N = ⌈(z/ε)²⌉."""
    return int(np.ceil((z / eps_rad) ** 2))


# ------------------------------------------------------------------ C. de la articulación al efector
def end_effector_rms(thetas: Sequence[float], links: Sequence[float], n_shots: int) -> float:
    """Error cuadrático medio del efector a primer orden cuando cada articulación se lee con N disparos en Z (errores
    independientes): sqrt(tr(J Jᵀ)/N)."""
    J = planar_jacobian(thetas, links)
    return float(np.sqrt(np.trace(J @ J.T) / n_shots))


def shots_for_end_effector(thetas: Sequence[float], links: Sequence[float], eps: float, z: float = 1.0) -> int:
    """Disparos por articulación para que z·RMS del efector ≤ ε (ε en las unidades de los eslabones): ⌈z²·tr(J Jᵀ)/ε²⌉."""
    J = planar_jacobian(thetas, links)
    return int(np.ceil(z * z * np.trace(J @ J.T) / (eps * eps)))


# ------------------------------------------------------------------ D. presupuesto de tiempo real
def shots_per_cycle(rate_hz: float, t_shot_s: float, t_overhead_s: float = 0.0) -> int:
    """Disparos que caben en un ciclo de control de frecuencia `rate_hz`: ⌊(1/rate − t_overhead)/t_shot⌋ (qubits de las
    articulaciones medidos en paralelo en el mismo dispositivo)."""
    avail = 1.0 / rate_hz - t_overhead_s
    return max(0, int(np.floor(avail / t_shot_s + 1e-12)))


def realtime_joint_sigma(rate_hz: float, t_shot_s: float, t_overhead_s: float = 0.0) -> float:
    """σ_θ (rad) alcanzable por ciclo, 1/√N con N = `shots_per_cycle`; inf si no cabe ningún disparo."""
    n = shots_per_cycle(rate_hz, t_shot_s, t_overhead_s)
    return float("inf") if n == 0 else float(1.0 / np.sqrt(n))


def max_control_rate(eps_rad: float, t_shot_s: float, t_overhead_s: float = 0.0, z: float = 1.0) -> float:
    """Frecuencia máxima del lazo (Hz) para que z·σ_θ ≤ ε con una sola articulación: 1/(t_overhead + N t_shot), N=⌈(z/ε)²⌉."""
    n = int(np.ceil((z / eps_rad) ** 2))
    return 1.0 / (t_overhead_s + n * t_shot_s)


# ------------------------------------------------------------------ E. articulación que deriva (Kalman escalar)
def steady_state_variance(q: float, r: float) -> Tuple[float, float]:
    """Kalman escalar para x_{t+1}=x_t+w (var q), y_t=x_t+v (var r): (a priori, a posteriori) estacionarias.
    P⁻ = (q + √(q²+4qr))/2, P = P⁻ − q."""
    prior = 0.5 * (q + np.sqrt(q * q + 4.0 * q * r))
    return float(prior), float(prior - q)


def kalman_scalar_track(y: Sequence[float], r: float, q: float, x0: float = 0.0, p0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Filtro de Kalman escalar: devuelve (estimaciones, varianzas a posteriori) para la serie de lecturas `y`."""
    x, p = float(x0), float(p0)
    xs, ps = [], []
    for yt in y:
        p_prior = p + q
        k = p_prior / (p_prior + r)
        x = x + k * (yt - x)
        p = (1.0 - k) * p_prior
        xs.append(x)
        ps.append(p)
    return np.array(xs), np.array(ps)


# ------------------------------------------------------------------ chequeo previo a la acción
def preflight(thetas: Sequence[float], links: Sequence[float], eps: float, rate_hz: float, t_shot_s: float,
              t_overhead_s: float = 0.0, z: float = 1.0) -> Dict[str, float]:
    """Chequeo previo a ejecutar la acción: ¿los disparos que caben en el ciclo alcanzan la tolerancia `eps` del efector?
    Devuelve disparos requeridos y disponibles por articulación, σ_θ en rad y en qga, RMS del efector con lo disponible,
    y `feasible` (1.0 si disponibles ≥ requeridos, 0.0 si no)."""
    need = shots_for_end_effector(thetas, links, eps, z)
    have = shots_per_cycle(rate_hz, t_shot_s, t_overhead_s)
    rms = end_effector_rms(thetas, links, have) if have > 0 else float("inf")
    sig = realtime_joint_sigma(rate_hz, t_shot_s, t_overhead_s)
    return {"shots_required": float(need), "shots_available": float(have), "sigma_theta_rad": sig,
            "sigma_theta_qga": float(sig * 1000.0 / (2.0 * np.pi)), "ee_rms": rms, "feasible": 1.0 if have >= need else 0.0}


# ------------------------------------------------------------------ F. Grover para elegir trayectorias
def grover_angle(n_candidates: int, n_marked: int) -> float:
    """θ0 con sin²θ0 = M/N."""
    return float(np.arcsin(np.sqrt(n_marked / n_candidates)))


def grover_optimal_iterations(n_candidates: int, n_marked: int) -> int:
    """k = ⌊π/(4θ0)⌋ (el entero que maximiza sin²((2k+1)θ0) antes del primer máximo)."""
    return int(np.floor(np.pi / (4.0 * grover_angle(n_candidates, n_marked))))


def grover_success(n_candidates: int, n_marked: int, k: int) -> float:
    """Probabilidad de medir un candidato marcado tras k iteraciones: sin²((2k+1)θ0), exacta."""
    return float(np.sin((2 * k + 1) * grover_angle(n_candidates, n_marked)) ** 2)


def grover_simulate(n_candidates: int, marked: Sequence[int], k: int) -> float:
    """Simulación explícita (vector de estado de longitud N) de k iteraciones: probabilidad total en `marked`. Verifica
    `grover_success`."""
    N = n_candidates
    psi = np.full(N, 1.0 / np.sqrt(N))
    mk = np.zeros(N, dtype=bool)
    mk[list(marked)] = True
    for _ in range(k):
        psi[mk] *= -1.0
        psi = 2.0 * psi.mean() - psi
    return float(np.sum(psi[mk] ** 2))


def classical_expected_queries(n_candidates: int, n_marked: int) -> float:
    """Evaluaciones esperadas del costo hasta hallar un candidato marcado, sin repetición y sin estructura: (N+1)/(M+1)."""
    return (n_candidates + 1.0) / (n_marked + 1.0)


def grover_query_ratio(n_candidates: int, n_marked: int) -> float:
    """Cociente de consultas clásicas esperadas sobre consultas de Grover (k óptimo, más la medición final)."""
    return classical_expected_queries(n_candidates, n_marked) / (grover_optimal_iterations(n_candidates, n_marked) + 1)


def grover_break_even_overhead(n_candidates: int, n_marked: int) -> float:
    """Cuánto puede costar una consulta cuántica (oráculo coherente de colisiones) en múltiplos de una evaluación clásica
    para que Grover siga empatando: (N+1)/(M+1) / (k+1). Por encima, pierde."""
    return grover_query_ratio(n_candidates, n_marked)
