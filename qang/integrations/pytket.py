"""
qang.integrations.pytket — the qang rotation and qg readings on pytket circuits (book, chapter 32).

pytket gives rotation angles in half-turns (units of pi), not radians: ``rqang`` converts, so that
``circ.Ry(theta/pi, q)`` prepares qg_Z = qg. Readings come from any pytket ``BackendResult`` (counts as tuples,
qubit 0 first, through qang.interop) or, without a backend, from the circuit's own statevector (pytket's ILO-BE
order, qubit 0 = most significant bit).

  * ``rqang(circuit, qg, qubit)``          append Ry(arccos(qg)/pi);
  * ``qg_readout(result, weight)``         readings of a BackendResult;
  * ``statevector_qg(circuit, weight)``    exact (qg_Z, filtered qg_Z, kept fraction) from get_statevector().
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from . import require
from ._readings import filtered_qg_z_from_probs, qg_z_from_probs
from ._sdk import readout, theta_from_qg
from ..interop import canonical_counts

require("pytket", "pytket")

__all__ = ["rqang", "qg_readout", "statevector_qg"]


def rqang(circuit, qg: float, qubit: int):
    circuit.Ry(theta_from_qg(qg) / math.pi, qubit)
    return circuit


def qg_readout(result, weight: Optional[int] = None) -> dict:
    return readout(canonical_counts(result), weight)


def statevector_qg(circuit, weight: Optional[int] = None) -> dict:
    probs = np.abs(np.asarray(circuit.get_statevector())) ** 2
    n = circuit.n_qubits
    out = {"qg_z": qg_z_from_probs(probs, n)}
    if weight is not None:
        out["filtered_qg_z"], out["kept_fraction"] = filtered_qg_z_from_probs(probs, n, weight)
    return out
