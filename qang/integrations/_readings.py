"""Shared NumPy helpers for the integrations: readings from a probability vector over 2^n outcomes.

Convention: qubit 0 is the FIRST bit of an outcome (qang.interop's canonical form). For a probability vector
indexed by the integer outcome, qubit q is bit (n - 1 - q) of the index, as in qang.qml.
"""

from __future__ import annotations

import numpy as np


def bits_table(n: int) -> np.ndarray:
    """(2^n, n) table of bits; column q is qubit q (qubit 0 = most significant bit of the index)."""
    idx = np.arange(2**n)
    return (idx[:, None] >> np.arange(n)[::-1]) & 1


def qg_z_from_probs(probs, n: int) -> np.ndarray:
    """qg_Z of each qubit from a probability vector: 1 - 2 P(qubit q = 1)."""
    p = np.asarray(probs, dtype=float)
    return 1.0 - 2.0 * (p @ bits_table(n))


def filtered_qg_z_from_probs(probs, n: int, weight: int):
    """(filtered qg_Z per qubit, kept fraction): <P_w Z_q> / <P_w>."""
    p = np.asarray(probs, dtype=float)
    b = bits_table(n)
    mask = b.sum(axis=1) == weight
    kept = float(p[mask].sum())
    if kept <= 0:
        raise ValueError(f"no probability in the weight-{weight} sector.")
    return 1.0 - 2.0 * (p[mask] @ b[mask]) / kept, kept


def milliqang(qg):
    """Reading in milli-qang: m = 1000 qg."""
    return 1000.0 * np.asarray(qg, dtype=float)
