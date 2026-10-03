"""Tests for qang.polarization (RESEARCH_NOTES §83): Stokes parameters are qg
values, Mueller matrices are the qg gate rule, Malus's law in qg units."""

import math

import numpy as np
import pytest

from qang import formulation as F
from qang import polarization as P

H = np.array([1, 0])
V = np.array([0, 1])
D = np.array([1, 1]) / math.sqrt(2)
A = np.array([1, -1]) / math.sqrt(2)
R = np.array([1, 1j]) / math.sqrt(2)
L = np.array([1, -1j]) / math.sqrt(2)


def qg_from_formulation(E):
    q = F.qg_values(np.asarray(E, dtype=complex))
    return {k: q.get(k, 0.0) for k in "XYZ"}


@pytest.mark.parametrize("E,expect", [(H, ("Z", 1)), (V, ("Z", -1)), (D, ("X", 1)), (A, ("X", -1)),
                                      (R, ("Y", 1)), (L, ("Y", -1))])
def test_basis_states(E, expect):
    q = P.jones_to_qg(E)
    k, v = expect
    assert q[k] == pytest.approx(v)
    assert sum(abs(q[j]) for j in "XYZ" if j != k) == pytest.approx(0, abs=1e-12)


def test_stokes_qg_round_trip_and_formulation():
    rng = np.random.default_rng(1)
    for _ in range(20):
        E = rng.normal(size=2) + 1j * rng.normal(size=2)
        E /= np.linalg.norm(E)
        q = P.jones_to_qg(E)
        ref = qg_from_formulation(E)
        assert all(q[k] == pytest.approx(ref[k], abs=1e-12) for k in "XYZ")
        S = P.qg_to_stokes(q, 3.0)
        assert P.stokes_to_qg(S) == pytest.approx(q)
    assert P.jones_to_qg(R, circular_sign=-1)["Y"] == pytest.approx(-1)


def test_degree_of_polarization_and_purity():
    rng = np.random.default_rng(2)
    for _ in range(10):
        E = rng.normal(size=2) + 1j * rng.normal(size=2)
        E /= np.linalg.norm(E)
        p = rng.uniform(0, 1)
        S = P.depolarizer(1 - p) @ P.jones_to_stokes(E)
        assert P.degree_of_polarization(S) == pytest.approx(1 - p)
        rho = F.state_from_qg({"I": 1.0, **{k: v for k, v in P.stokes_to_qg(S).items()}}, 1)
        assert P.purity_from_stokes(S) == pytest.approx(float(np.real(np.trace(rho @ rho))))


def test_mueller_of_lossless_elements_is_the_qg_gate_rule():
    rng = np.random.default_rng(3)
    elements = [P.half_wave_plate(0.3), P.quarter_wave_plate(1.1), P.rotator(0.7), P.wave_plate(0.4, -0.2)]
    for J in elements:
        M = P.mueller_from_jones(J)
        for _ in range(5):
            E = rng.normal(size=2) + 1j * rng.normal(size=2)
            E /= np.linalg.norm(E)
            q_in = qg_from_formulation(E)
            q_rule = F.apply_gate({"I": 1.0, **q_in}, J)
            S_out = M @ P.qg_to_stokes(q_in)
            q_out = P.stokes_to_qg(S_out)
            assert all(q_out[k] == pytest.approx(q_rule.get(k, 0.0), abs=1e-12) for k in "XYZ")
        assert np.allclose(P.mueller_from_ptm(P.ptm_from_mueller(M)), M)


def test_malus_law():
    for theta in np.linspace(0, math.pi, 7):
        M = P.mueller_from_jones(P.linear_polarizer(theta))
        for E in (H, D, R, np.array([0.6, 0.8j])):
            q = P.jones_to_qg(E)
            assert (M @ P.qg_to_stokes(q))[0] == pytest.approx(P.malus_intensity(q, theta))
        assert P.malus_intensity(P.jones_to_qg(H), theta) == pytest.approx(math.cos(theta) ** 2)


def test_half_wave_plate_rotates_linear_polarization_by_twice_the_angle():
    for alpha, theta in ((0.2, 0.5), (1.0, -0.3), (0.0, math.pi / 8)):
        E = np.array([math.cos(alpha), math.sin(alpha)])
        out = P.half_wave_plate(theta) @ E
        expect = np.array([math.cos(2 * theta - alpha), math.sin(2 * theta - alpha)])
        assert P.jones_to_qg(out) == pytest.approx(P.jones_to_qg(expect))


def test_qg_from_counts():
    out = P.qg_from_counts(900, 100, 500, 500, 520, 480, method="wilson")
    assert out["Z"] == pytest.approx(0.8)
    assert out["X"] == pytest.approx(0.0)
    assert out["P"] == pytest.approx(math.sqrt(0.8**2 + 0.04**2))
    assert set(out["intervals"]) == {"X", "Y", "Z"}
    with pytest.raises(ValueError):
        P.qg_from_counts(0, 0, 1, 1, 1, 1)
