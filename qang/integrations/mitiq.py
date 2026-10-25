"""
qang.integrations.mitiq — the qg filter, its cost rule and ZNE of filtered readings in Mitiq (book, chapter 21).

Mitiq already post-selects on Hamming weight (``mitiq.rem.post_select``). What qang adds on top:

  * ``qg_z(result)`` / ``filtered_qg_z(result, weight)``
        the per-qubit qg_Z of a ``MeasurementResult`` (bit i = qubit_indices[i]), raw or after the filter, with the
        kept fraction;
  * ``qg_executor(sampler, weight=None, qubits=None)``
        turns a sampler (circuit -> ``MeasurementResult``, or circuit -> probability vector over 2^n outcomes with
        qubit 0 as the most significant bit) into the float-valued executor Mitiq's techniques expect: the mean
        qg_Z of ``qubits``, filtered on ``weight`` if given;
  * ``execute_with_filter_zne(circuit, sampler, weight, scale_factors)``
        zero-noise extrapolation of the filtered reading (chapter 21, F6: under T1 plus dephasing the filter leaves
        only the dephasing, which ZNE then removes);
  * ``filter_rule(z_filtered, z_raw, kept, shots)``
        the closed-form decision of chapter 21: filtered MSE (1 - z_f^2) / (K S) against raw bias^2 + (1 - z_r^2) / S
        (simulated / rule 1.00-1.05 in the book).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from . import require
from ._readings import filtered_qg_z_from_probs, qg_z_from_probs

mitiq = require("mitiq", "mitiq")
from mitiq import zne  # noqa: E402
from mitiq.rem import post_select  # noqa: E402

__all__ = ["qg_z", "filtered_qg_z", "qg_executor", "execute_with_filter_zne", "filter_rule"]


def _bits(result) -> np.ndarray:
    return np.asarray([[int(b) for b in s] for s in result.result], dtype=np.uint8)


def qg_z(result) -> np.ndarray:
    """qg_Z of every bit of a MeasurementResult: 1 - 2 P(bit = 1)."""
    return 1.0 - 2.0 * _bits(result).mean(axis=0)


def filtered_qg_z(result, weight: int):
    """(qg_Z of every bit after keeping the bitstrings of Hamming weight ``weight``, kept fraction)."""
    kept = post_select(result, lambda b: sum(int(x) for x in b) == weight)
    if not kept.result:
        raise ValueError(f"no bitstring has weight {weight}.")
    return qg_z(kept), len(kept.result) / len(result.result)


def qg_executor(sampler: Callable, weight: Optional[int] = None, qubits: Optional[Sequence[int]] = None) -> Callable:
    """Executor circuit -> float for Mitiq: mean qg_Z of ``qubits`` (all if None), filtered on ``weight``."""

    def run(circuit) -> float:
        out = sampler(circuit)
        if isinstance(out, np.ndarray):
            n = int(round(np.log2(out.size)))
            q = qg_z_from_probs(out, n) if weight is None else filtered_qg_z_from_probs(out, n, weight)[0]
        else:
            q = qg_z(out) if weight is None else filtered_qg_z(out, weight)[0]
        q = np.asarray(q)
        return float(q.mean() if qubits is None else q[list(qubits)].mean())

    return run


def execute_with_filter_zne(circuit, sampler: Callable, weight: Optional[int], scale_factors=(1.0, 2.0, 3.0),
                            qubits: Optional[Sequence[int]] = None, factory=None, scale_noise=None) -> float:
    """ZNE (Richardson, global folding by default) of the qg reading, filtered on ``weight`` (``None`` = raw)."""
    factory = factory or zne.inference.RichardsonFactory(scale_factors=list(scale_factors))
    scale_noise = scale_noise or zne.scaling.fold_global
    return float(zne.execute_with_zne(circuit, qg_executor(sampler, weight, qubits), factory=factory,
                                      scale_noise=scale_noise))


def filter_rule(z_filtered: float, z_raw: float, kept: float, shots: int) -> dict:
    """Chapter 21 rule: predicted MSE with the filter, (1 - z_f^2)/(K S), and without it, (z_r - z_f)^2 + (1 - z_r^2)/S
    (the filtered value stands in for the truth, exact under equal T1). ``use_filter`` is the cheaper choice."""
    mse_f = (1.0 - z_filtered**2) / (kept * shots)
    mse_r = (z_raw - z_filtered) ** 2 + (1.0 - z_raw**2) / shots
    return {"mse_filtered": mse_f, "mse_raw": mse_r, "use_filter": bool(mse_f < mse_r)}
