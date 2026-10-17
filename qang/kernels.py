"""
Quantum kernels with the qg filter (RESEARCH_NOTES §161).

The fidelity kernel of the unary (weight-1) amplitude encoding is K(x, y) = |<x|y>|^2 = (x . y)^2 for unit
vectors: a degree-2 polynomial kernel, classically easy, used here as a controlled test of how device noise
distorts a Gram matrix and what repairs it. It is estimated with the inversion test of §149
(``qang.ibm.unary_overlap_circuit``: load x, undo the loader of y, K = P(only qubit 0 excited)); every gate
conserves the Hamming weight, so the qg filter (keep the weight-1 shots) applies. The usual repair is the
diagonal normalization K_ij / sqrt(K_ii K_jj), which needs the diagonal circuits K(x, x) (ideally 1). In qg
terms K = cos^2(phi) for the angle phi between the two data vectors, so arccos(2K - 1) = 2 phi is a qang angle.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def unary_embedding(X, const: float = 1.0) -> np.ndarray:
    """Rows of X with a constant feature appended, scaled to unit norm (one qubit per entry)."""
    X = np.asarray(X, float)
    E = np.hstack([X, np.full((X.shape[0], 1), const)])
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def exact_kernel(A, B) -> np.ndarray:
    """K_ij = (a_i . b_j)^2 for unit rows."""
    return (np.asarray(A, float) @ np.asarray(B, float).T) ** 2


def kernel_from_counts(counts: Dict[str, int]) -> dict:
    """{"raw", "filtered", "kept"} from inversion-test counts (bitstrings, qubit 0 rightmost)."""
    tot = sum(counts.values())
    e0 = w1 = 0
    for k, c in counts.items():
        b = k.replace(" ", "")
        if b.count("1") == 1:
            w1 += c
            if b[-1] == "1":
                e0 += c
    return {"raw": e0 / tot, "filtered": e0 / w1 if w1 else float("nan"), "kept": w1 / tot}


def normalize_gram(K, diag_rows: Sequence[float], diag_cols: Sequence[float]) -> np.ndarray:
    """K_ij / sqrt(d_i d_j): the standard repair of a noisy fidelity kernel."""
    return np.asarray(K, float) / np.sqrt(np.outer(diag_rows, diag_cols))


def alignment(K, y) -> float:
    """Kernel-target alignment <K, y y^T>_F / (||K||_F ||y y^T||_F), labels y in {-1, +1}."""
    K = np.asarray(K, float)
    Y = np.outer(y, y)
    return float(np.sum(K * Y) / (np.linalg.norm(K) * np.linalg.norm(Y)))


def kernel_qang_angle(K) -> np.ndarray:
    """arccos(2K - 1) = 2 phi, phi the angle between the two data vectors."""
    return np.arccos(np.clip(2 * np.asarray(K, float) - 1, -1, 1))
