"""
qang.cirq_gate — qg as native gates for Cirq.

Completes Future Research Direction #1 ("implement qg as a native unit/type
in quantum-computing SDKs (Qiskit, Cirq, PennyLane wrappers) ...") for
Cirq, alongside the existing Qiskit integration in qang.qiskit_gate.

Two constructors, mirroring qang.qiskit_gate's RQangGate / FullRQangGate:

  * ``rqang_gate(qang)``       qg_Z only -> ``cirq.ry(theta)``, Cirq's own
                                 native Y-rotation gate (no wrapper class
                                 needed -- this is the idiomatic Cirq way).
  * ``full_rqang_gate(qang)``  full (qg_Z, phi) -> a ``cirq.MatrixGate``
                                 wrapping the explicit U(theta, phi, 0)
                                 unitary, exactly matching
                                 Qang.to_statevector().

This module is optional: importing ``qang`` itself never requires Cirq.
Only importing *this* module does, and it raises a clear, actionable error
if Cirq is not installed.
"""

from __future__ import annotations

import cmath
import math
from typing import Union

from .core import Qang

try:
    import cirq
    import numpy as np

    _CIRQ_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when cirq is absent
    _CIRQ_AVAILABLE = False


def _require_cirq():
    if not _CIRQ_AVAILABLE:
        raise ImportError(
            "qang.cirq_gate requires Cirq. Install it with `pip install cirq`."
        )


def _as_polar_qang(qang_val: Union[float, "Qang"]) -> "Qang":
    qg = qang_val if isinstance(qang_val, Qang) else Qang(qang_val, mode="polar")
    if qg.mode != "polar":
        raise ValueError("cirq_gate functions require a polar-mode Qang (qg_Z).")
    return qg


if _CIRQ_AVAILABLE:

    def rqang_gate(qang_val: Union[float, Qang]) -> "cirq.Gate":
        """
        A gate parameterized directly by qg_Z (polar qang): prepares
        P(|0>) = (1 + qg_Z) / 2, P(|1>) = (1 - qg_Z) / 2 starting from |0>,
        with no control over the relative phase. Cirq's native ry(theta)
        IS the right primitive here -- no custom Gate subclass needed.
        """
        qg = _as_polar_qang(qang_val)
        return cirq.ry(qg.to_theta())

    def full_rqang_gate(qang: Qang) -> "cirq.Gate":
        """
        A gate parameterized by the full (qg_Z, phi) Bloch-sphere Qang,
        i.e. it prepares exactly the state Qang.to_statevector() describes,
        starting from |0>. Built as an explicit 2x2 unitary
        (U(theta, phi, lambda=0), the same convention as Qiskit's UGate)
        wrapped in cirq.MatrixGate, so it matches qang.qiskit_gate's
        FullRQangGate bit-for-bit.
        """
        qg = _as_polar_qang(qang)
        theta, phi = qg.to_angles(unit="rad")
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        phase = cmath.exp(1j * phi)
        matrix = np.array(
            [[c, -s], [phase * s, phase * c]],
            dtype=complex,
        )
        return cirq.MatrixGate(matrix, name=f"RQang3D({qg.value:+.4f},{phi:.4f})")

    def append_qang(
        circuit: "cirq.Circuit", qang: Qang, qubit: "cirq.Qid", full: bool = True
    ) -> "cirq.Circuit":
        """Convenience: append the right gate for ``qang`` to ``qubit`` of ``circuit``."""
        gate = full_rqang_gate(qang) if full else rqang_gate(qang)
        circuit.append(gate.on(qubit))
        return circuit

else:  # pragma: no cover - exercised only when cirq is absent

    def rqang_gate(*args, **kwargs):
        _require_cirq()

    def full_rqang_gate(*args, **kwargs):
        _require_cirq()

    def append_qang(*args, **kwargs):
        _require_cirq()
