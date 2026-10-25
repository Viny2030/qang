"""
qang.integrations.qiskit_dynamics — pulse-level qg readings and the pulse-area rule (book, §160 and chapter 2).

A resonant drive of Rabi rate Omega(t) rotates the qubit by the pulse area A = integral Omega dt, and the qubit
reads qg_Z = cos A. ``rabi_qg`` solves the driven qubit with Qiskit Dynamics (lab frame or rotating frame) and
returns that reading; ``area_error_qg`` gives the reading when the amplitude is off by a factor (1 + eps), which is
``qang.pulse_area.rx_qg_z_systematic_amplitude``. The minimal-area rule of §160 (run theta mod 2 pi on the short
side) follows: the same eps costs less on a smaller area.

Qiskit Dynamics 0.6 requires ``qiskit<=1.3``, so it does not install next to Qiskit 2.x; use a separate environment
(``pip install "qang[dynamics]"``).
"""

from __future__ import annotations

import numpy as np

from . import require

require("qiskit_dynamics", "dynamics")
from qiskit_dynamics import Solver  # noqa: E402

__all__ = ["rabi_qg", "area_error_qg"]

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def rabi_qg(area: float, duration: float = 50.0, frequency: float = 5.0, rotating_frame: bool = True,
            shape: str = "square", n_steps: int = 2001) -> float:
    """qg_Z after a resonant pulse of total area ``area`` (radians) on |0>; ``frequency`` and ``duration`` in
    consistent units (GHz and ns by default). shape: "square" or "gaussian" (sigma = duration / 6)."""
    t = np.linspace(0.0, duration, n_steps)
    if shape == "square":
        env = np.ones_like(t)
    elif shape == "gaussian":
        env = np.exp(-0.5 * ((t - duration / 2) / (duration / 6)) ** 2)
    else:
        raise ValueError("shape must be 'square' or 'gaussian'")
    amp = area / np.trapezoid(env, t)
    w = 2 * np.pi * frequency

    def envelope(s):
        return amp * np.interp(s, t, env)

    # H = w/2 Z + Omega(t) cos(w t) X; in the frame of w/2 Z and with the rotating-wave approximation the drive is
    # Omega(t)/2 X, so the area of Omega is the rotation angle.
    solver = Solver(static_hamiltonian=0.5 * w * _Z, hamiltonian_operators=[_X],
                    rotating_frame=0.5 * w * _Z if rotating_frame else None,
                    rwa_cutoff_freq=1.5 * w if rotating_frame else None)

    from qiskit_dynamics.signals import Signal

    sol = solver.solve(t_span=[0.0, duration], y0=np.array([1.0, 0.0], dtype=complex),
                       signals=[Signal(envelope, carrier_freq=frequency)], atol=1e-10, rtol=1e-10)
    psi = sol.y[-1]
    return float(abs(psi[0]) ** 2 - abs(psi[1]) ** 2)


def area_error_qg(theta: float, eps: float) -> float:
    """qg_Z = cos((1 + eps) theta): the reading of an RX(theta) whose amplitude is off by (1 + eps)."""
    return float(np.cos((1.0 + eps) * theta))
