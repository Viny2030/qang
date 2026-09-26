"""
Tests for examples/qrng_qg_certified.py: certified min-entropy of a qubit
QRNG in qg units, and the naive vs qg estimators.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

import qrng_qg_certified as Q  # noqa: E402


def _helstrom_guess(bloch):
    """Guessing probability of Z by an adversary holding the purification."""
    x, y, z = bloch
    rho = 0.5 * np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]])
    w, v = np.linalg.eigh(rho)
    psi = sum(math.sqrt(max(w[k], 0)) * np.kron(v[:, k], np.eye(2)[k]) for k in range(2))  # |psi>_{AE}
    psi = psi.reshape(2, 2)
    e0, e1 = psi[0], psi[1]  # unnormalized conditional states of E
    m = np.outer(e0, e0.conj()) - np.outer(e1, e1.conj())
    return 0.5 * (1 + np.abs(np.linalg.eigvalsh(m)).sum())


@pytest.mark.parametrize("bloch", [(1, 0, 0), (0.6, 0.3, 0.2), (0.2, 0, 0), (0, 0, 0.9), (0.5, 0.5, 0.5)])
def test_certified_min_entropy_formula(bloch):
    r_perp = math.hypot(bloch[0], bloch[1])
    assert -math.log2(_helstrom_guess(bloch)) == pytest.approx(Q.h_min_certified(r_perp), abs=1e-10)


def test_pure_state_reduces_to_classical_bias():
    theta = 0.7
    assert Q.h_min_certified(math.sin(theta)) == pytest.approx(Q.h_min_naive((1 + math.cos(theta)) / 2))


def test_naive_estimate_is_unsafe_under_dephasing_qg_is_safe():
    t = Q.table(n_test=10000, reps=60, seed=3)
    for name in ("T2: V = 0.8", "T2: V = 0.5", "thermal p = 0.05, V = 0.9"):
        r = t[name]
        assert r["naive"][1] == 1.0 and r["naive"][0] > 0.9
        assert r["qg +/- pairs"][1] == 0.0 and r["qg calibrated"][1] == 0.0
        assert r["qg calibrated"][0] > r["qg +/- pairs"][0]
        assert r["qg calibrated"][0] > 0.8 * r["truth"]


def test_one_sided_readout_offset_can_overestimate():
    rng = np.random.default_rng(4)
    bloch = Q.device_state(0.2)
    unsafe = np.mean([Q.estimates(bloch, 10000, rng)["qg one-sided"] > Q.truth(bloch) for _ in range(300)])
    assert unsafe > 0.02
