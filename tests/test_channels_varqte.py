"""Tests for qang.channels and qang.varqte (§162)."""

import math

import numpy as np
import pytest

from qang import channels as CH


def _rho(r):
    return (np.eye(2) + sum(x * P for x, P in zip(r, CH._P))) / 2


def _r(rho):
    return np.array([np.real(np.trace(P @ rho)) for P in CH._P])


@pytest.mark.parametrize("kraus", [CH.amplitude_damping(0.3), CH.phase_damping(0.4), CH.depolarizing(0.2),
                                   CH.thermal_relaxation(1.0, 3.0, 2.5, 0.1)])
def test_every_channel_is_affine_in_the_qg_vector(kraus):
    assert np.allclose(sum(K.conj().T @ K for K in kraus), np.eye(2))
    rng = np.random.default_rng(0)
    for _ in range(5):
        r = rng.normal(size=3)
        r *= rng.uniform(0, 1) / np.linalg.norm(r)
        assert np.allclose(CH.qg_after(kraus, r), _r(CH.apply_kraus(kraus, _rho(r))), atol=1e-12)


def test_closed_forms():
    r = np.array([0.3, -0.2, 0.5])
    g = 0.3
    assert np.allclose(CH.qg_after(CH.amplitude_damping(g), r), [0.3 * math.sqrt(0.7), -0.2 * math.sqrt(0.7),
                                                                 0.7 * 0.5 + 0.3])
    assert np.allclose(CH.qg_after(CH.depolarizing(0.2), r), 0.8 * r)
    t, T1, T2, p1 = 1.0, 3.0, 2.5, 0.1
    out = CH.qg_after(CH.thermal_relaxation(t, T1, T2, p1), r)
    assert out[2] == pytest.approx(math.exp(-t / T1) * 0.5 + (1 - math.exp(-t / T1)) * (1 - 2 * p1))
    assert out[0] == pytest.approx(math.exp(-t / T2) * 0.3)


def test_varqte_ansatz_conserves_weight():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    from qang import varqte as VQ

    th = np.random.default_rng(1).normal(size=VQ.n_params(2))
    p = Statevector(VQ.ansatz(th, 2)).probabilities()
    w = np.array([bin(i).count("1") for i in range(32)])
    assert p[w != 2].sum() < 1e-12
