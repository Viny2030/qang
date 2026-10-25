"""
qang.integrations.braket — the qang rotation and the filtered readout on Amazon Braket (book, chapter 32).

Braket is no longer only a result format (qang.interop): ``run_qg`` runs a circuit on any Braket device (the
``LocalSimulator`` by default, or an AWS device the user has access to) and returns the qg readings.

  * ``rqang(circuit, qg, qubit)``                         append ry(qubit, arccos qg);
  * ``run_qg(circuit, shots, weight=None, device=None)``   run, then qg_Z profile (+ filter) of the result.

Braket's ``measurement_counts`` put qubit 0 first; qang.interop reads them in that order.
"""

from __future__ import annotations

from typing import Optional

from . import require
from ._sdk import readout, theta_from_qg
from ..interop import canonical_counts

require("braket.circuits", "braket")

__all__ = ["rqang", "run_qg"]


def rqang(circuit, qg: float, qubit: int):
    circuit.ry(qubit, theta_from_qg(qg))
    return circuit


def run_qg(circuit, shots: int, weight: Optional[int] = None, device=None) -> dict:
    if device is None:
        device = require("braket.devices", "braket").LocalSimulator()
    result = device.run(circuit, shots=shots).result()
    return readout(canonical_counts(result), weight)
