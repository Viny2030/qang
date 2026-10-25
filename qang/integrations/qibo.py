"""
qang.integrations.qibo — the qang rotation and the filtered readout on Qibo circuits (book, chapter 32).

  * ``rqang(circuit, qg, qubit)``     appends RY(arccos qg) so that the qubit reads qg_Z = qg;
  * ``canonical_counts(result)``      {(b_0, ..., b_{n-1}): count} from a Qibo result (``frequencies()``,
                                      qubit 0 first, the same order as Qibo's bitstrings);
  * ``qg_readout(result, weight)``    qg_Z profile, and the filtered profile with its kept fraction.
"""

from __future__ import annotations

from typing import Optional

from . import require
from ._sdk import readout, theta_from_qg

qibo = require("qibo", "qibo")

__all__ = ["rqang", "canonical_counts", "qg_readout"]


def rqang(circuit, qg: float, qubit: int):
    circuit.add(qibo.gates.RY(qubit, theta_from_qg(qg)))
    return circuit


def canonical_counts(result) -> dict:
    return {tuple(int(b) for b in k): int(v) for k, v in result.frequencies(binary=True).items()}


def qg_readout(result, weight: Optional[int] = None) -> dict:
    return readout(canonical_counts(result), weight)
