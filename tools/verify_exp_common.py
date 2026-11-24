"""Utilidades compartidas para los verificadores experimentales F1--F8.

Resolucion angular fijada por el proyecto: N = 65536 shots,
sigma = qang.qga.angle_sigma_qga(N) qga y umbral de falsacion 5*sigma.
Cada estimacion independiente es un ensayo de Bernoulli: exito si la distancia
circular al valor predicho no excede el umbral. Bajo la hipotesis nula de fase
uniforme en [0, 1000) qga, p0 = 2*delta/1000. Todo resultado se etiqueta
explicitamente como ``exact``, ``simulado`` o ``unverified``.
"""
from __future__ import annotations

import cmath
import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from qang.geometric import geometric_phase_qg, pancharatnam_phase
from qang.qecho import invert_readout, readout_affine
from qang.qfalsify import binom_upper, decide
from qang.qga import angle_sigma_qga, azimuth_qga
from qang.phase import QangPhi

QGA_PER_TURN: float = 1000.0
SHOTS: int = 65_536
E01: float = 0.02
E10: float = 0.06
SEED: int = 21
N_ESTIMATES: int = 600
ALPHA: float = 0.05
SIGMA_QGA: float = float(angle_sigma_qga(SHOTS))
DELTA_5SIGMA_QGA: float = 3.108
P0_HIT: float = 2.0 * DELTA_5SIGMA_QGA / QGA_PER_TURN

RESOLUTION_TABLE: Tuple[Tuple[int, float, float], ...] = (
    (16_384, 1.243, 6.217),
    (32_768, 0.879, 4.396),
    (65_536, 0.622, 3.108),
)


def circular_distance_qga(a: float, b: float) -> float:
    """Distancia minima en el circulo de 1000 qga entre dos lecturas angulares."""
    d = abs((a - b) % QGA_PER_TURN)
    return min(d, QGA_PER_TURN - d)


def phase_qga_from_complex(z: complex) -> float:
    """Invierte un numero unitario con QangPhi.from_complex y devuelve qga."""
    return float((QangPhi.from_complex(z).phi * QGA_PER_TURN) % QGA_PER_TURN)


def signal_phase_qga(signal_turns: float) -> float:
    """Codifica una senal adimensional como fase qg_Phi y la lee en qga."""
    return phase_qga_from_complex(cmath.exp(2j * math.pi * float(signal_turns)))


def noisy_phase_samples(signal_turns: float, rng: np.random.Generator) -> np.ndarray:
    """Muestrea estimaciones de fase con la resolucion fijada y lectura affine invertida."""
    phi_true = signal_phase_qga(signal_turns)
    sigma_rad = 2.0 * math.pi * SIGMA_QGA / QGA_PER_TURN
    theta = 2.0 * math.pi * phi_true / QGA_PER_TURN + rng.normal(0.0, sigma_rad, N_ESTIMATES)
    x_true = QGA_PER_TURN * np.cos(theta)
    y_true = QGA_PER_TURN * np.sin(theta)
    x_obs = np.asarray([readout_affine(float(x), E01, E10) for x in x_true])
    x_cor = np.asarray([invert_readout(float(x), E01, E10) for x in x_obs])
    return np.asarray([azimuth_qga(float(x), float(y)) for x, y in zip(x_cor, y_true)])


def falsify_angular(prediction_qga: float, samples: Sequence[float]) -> Tuple[str, int, float]:
    """Devuelve (decision, aciertos, p_valor) segun qang.qfalsify."""
    hits = int(sum(circular_distance_qga(float(s), prediction_qga) <= DELTA_5SIGMA_QGA for s in samples))
    p_value = float(binom_upper(hits, len(samples), P0_HIT))
    return decide(hits, len(samples), P0_HIT, ALPHA), hits, p_value


def geometric_cone_qga(qg_z: float) -> float:
    """Fase geometrica de un cono cerrado, leida con geometric_phase_qg."""
    theta = math.acos(max(-1.0, min(1.0, float(qg_z))))
    path = [
        (math.sin(theta) * math.cos(a), math.sin(theta) * math.sin(a), float(qg_z))
        for a in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
    ]
    holonomy = geometric_phase_qg(path)
    return phase_qga_from_complex(holonomy.value)


def pancharatnam_qga(states: Iterable[np.ndarray]) -> float:
    """Fase de Pancharatnam de un lazo cerrado, expresada en qga."""
    gamma = pancharatnam_phase(list(states))
    return float((gamma / (2.0 * math.pi) * QGA_PER_TURN) % QGA_PER_TURN)


def emit(title: str, rows: Sequence[Tuple[str, object, str]]) -> None:
    """Imprime una fila por resultado con su etiqueta de evidencia."""
    print(title)
    for name, value, status in rows:
        print(f"  {name}: {value} [{status}]")
