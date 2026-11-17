"""Cirq: perfil qg_Z desde un cirq.Result."""
from __future__ import annotations


def qg_z_from_result(result):
    from qang.interop import canonical_counts, qg_z_profile
    return qg_z_profile(canonical_counts(result))
