"""Shared helpers for the SDK adapters (qibo, pytket, braket, cudaq): the qang angle and the canonical readout."""

from __future__ import annotations

import math
from typing import Optional

from ..interop import filtered_qg_z_profile, qg_z_profile, weight_filter


def theta_from_qg(qg: float) -> float:
    """RY angle that prepares qg_Z = qg from |0>: theta = arccos(qg)."""
    if not -1.0 <= qg <= 1.0:
        raise ValueError("qg must be in [-1, 1].")
    return math.acos(qg)


def readout(counts, weight: Optional[int] = None) -> dict:
    """Readings of canonical counts {(b_0, ..., b_{n-1}): count}: qg_Z profile, and with ``weight`` also the
    filtered profile and the kept fraction."""
    out = {"qg_z": [q for q, _ in qg_z_profile(counts)], "shots": sum(counts.values())}
    if weight is not None:
        out["kept_fraction"] = weight_filter(counts, weight)[1]
        out["filtered_qg_z"] = [q for q, _ in filtered_qg_z_profile(counts, weight)]
    return out
