"""
qang.qga -- the angular qang (qga): a thousand-based angle scale (book, chapter 1, section "La escala angular").

The *Geometric and Dimensional Foundation of the Qang System* measures angles in thousandths of a turn. qang keeps
the milli-qang (m-qg) for what it has always been, the reading m = 1000 * qg_Z, which is linear in the probability
and is not an angle (``qang.probability``). The angle scale gets its own name, qga:

    1 turn = 2*pi rad = 1000 qga = 10^6 m-qga,        theta[rad] = 2*pi * theta_qga / 1000

With the Bloch polar angle in qga, the Born rule and the reading are

    p0 = cos^2(pi * theta_qga / 1000),        m = 1000 * cos(2*pi * theta_qga / 1000) m-qg

so the pole |0> is at 0 qga, the equator at 250 qga and the pole |1> at 500 qga; the phase runs over [0, 1000) qga.
The two scales complement each other: qga is uniform in the angle, m-qg is uniform in the probability.

Geometry in qga only carries constant factors: the Fubini-Study metric is (pi^2/10^6)(d theta_qga^2 +
sin^2(theta) d phi_qga^2) and the Fisher information of one Z shot is 4 pi^2 10^-6 qga^-2 everywhere, so the qang
angle estimated from N shots has standard deviation (1000 / 2 pi) / sqrt(N) = 159.15 / sqrt(N) qga for any state.
A change of unit rescales every gradient by the same constant 2 pi / 1000: it does not change barren plateaus.

Three claims of the source document are not adopted (the book says why): p0 = 0.5 is at 250 qga, not 125; the metric
(1/4) d theta^2 has Gaussian curvature 4, not 1; and (mc^2/E)^2 is not p0 of the same angle. Its relativistic and
antimatter extension is an unverified proposal and is not part of this module. Standard library only.
"""

from __future__ import annotations

import math
from typing import Tuple

from .core import MILLIQANG_PER_QANG

__all__ = [
    "QGA_PER_TURN",
    "MILLI_QGA_PER_QGA",
    "FISHER_PER_QGA2",
    "radians_from_qga",
    "qga_from_radians",
    "degrees_from_qga",
    "qga_from_degrees",
    "p0_from_qga",
    "qga_from_p0",
    "milliqang_from_qga",
    "qga_from_milliqang",
    "azimuth_qga",
    "angle_sigma_qga",
    "fubini_study_metric_qga",
    "cone_phase_qga",
]

QGA_PER_TURN = 1000.0
"""One full turn, 2*pi rad, is 1000 qga."""

MILLI_QGA_PER_QGA = 1000.0
"""1 qga = 1000 m-qga; 1 m-qga is about 6.283e-6 rad. Not to be confused with the m-qg reading."""

FISHER_PER_QGA2 = (2.0 * math.pi / QGA_PER_TURN) ** 2
"""Fisher information of one Z-basis shot about the qang angle, in qga^-2: 4 pi^2 10^-6, the same for every state."""


def radians_from_qga(theta_qga: float) -> float:
    """Angle in radians of an angle in qga: theta = 2*pi * theta_qga / 1000."""
    return 2.0 * math.pi * float(theta_qga) / QGA_PER_TURN


def qga_from_radians(theta: float) -> float:
    """Angle in qga of an angle in radians: theta_qga = 1000 * theta / (2*pi)."""
    return QGA_PER_TURN * float(theta) / (2.0 * math.pi)


def degrees_from_qga(theta_qga: float) -> float:
    """Angle in degrees of an angle in qga (1000 qga = 360 degrees)."""
    return 360.0 * float(theta_qga) / QGA_PER_TURN


def qga_from_degrees(degrees: float) -> float:
    """Angle in qga of an angle in degrees (360 degrees = 1000 qga)."""
    return QGA_PER_TURN * float(degrees) / 360.0


def p0_from_qga(theta_qga: float) -> float:
    """Born probability of outcome 0 for the Bloch polar angle theta_qga: cos^2(pi * theta_qga / 1000)."""
    return math.cos(math.pi * float(theta_qga) / QGA_PER_TURN) ** 2


def qga_from_p0(p0: float) -> float:
    """The qang angle of a probability, in qga: (1000 / pi) * arccos(sqrt(p0)), in [0, 500]."""
    if not 0.0 <= p0 <= 1.0:
        raise ValueError(f"p0 must lie in [0, 1], got {p0}.")
    return QGA_PER_TURN / math.pi * math.acos(math.sqrt(p0))


def milliqang_from_qga(theta_qga: float) -> float:
    """The m-qg reading of the Bloch polar angle theta_qga: 1000 * cos(2*pi * theta_qga / 1000)."""
    return MILLIQANG_PER_QANG * math.cos(radians_from_qga(theta_qga))


def qga_from_milliqang(m: float) -> float:
    """The qang angle, in qga, of an m-qg reading: (500 / pi) * arccos(m / 1000), in [0, 500]."""
    if not -MILLIQANG_PER_QANG <= m <= MILLIQANG_PER_QANG:
        raise ValueError(f"m must lie in [-1000, 1000] m-qg, got {m}.")
    return qga_from_radians(math.acos(m / MILLIQANG_PER_QANG))


def azimuth_qga(qg_x: float, qg_y: float) -> float:
    """The azimuthal phase atan2(qg_Y, qg_X) in qga, in [0, 1000). Undefined at the poles (qg_X = qg_Y = 0).

    The ratio cancels the length of the transverse component, so the phase survives noise that shortens qg_X and
    qg_Y equally (depolarizing, dephasing, T1); a channel that treats X and Y differently shifts it (book, ch. 5).
    """
    if qg_x == 0.0 and qg_y == 0.0:
        raise ValueError("the azimuth is undefined at the poles (qg_X = qg_Y = 0).")
    return qga_from_radians(math.atan2(qg_y, qg_x) % (2.0 * math.pi)) % QGA_PER_TURN


def angle_sigma_qga(n_shots: int) -> float:
    """Standard deviation of the qang angle estimated from n_shots Z shots, in qga: (1000 / 2 pi) / sqrt(n).

    It is the same for every state (the arcsine stabilization); it is asymptotic and fails near the poles with few
    shots, where the exact intervals of ``qang.statistics`` apply.
    """
    if n_shots <= 0:
        raise ValueError("n_shots must be positive.")
    return QGA_PER_TURN / (2.0 * math.pi) / math.sqrt(n_shots)


def fubini_study_metric_qga(theta_qga: float) -> Tuple[float, float]:
    """Diagonal of the one-qubit Fubini-Study metric in (theta_qga, phi_qga): (pi^2/10^6, pi^2/10^6 sin^2 theta)."""
    c = (math.pi / QGA_PER_TURN) ** 2
    return c, c * math.sin(radians_from_qga(theta_qga)) ** 2


def cone_phase_qga(qg_z: float) -> float:
    """Geometric (Berry) phase of a loop at constant height qg_Z, in qga modulo 1000: 1000 * p0 = 500 * (1 + qg_Z).

    The phase in turns is p0 (book, ch. 7); in thousandths of a turn it is (1000 + m) / 2 for the reading m in m-qg.
    """
    if not -1.0 <= qg_z <= 1.0:
        raise ValueError(f"qg_Z must lie in [-1, 1], got {qg_z}.")
    return (QGA_PER_TURN * (1.0 + qg_z) / 2.0) % QGA_PER_TURN
