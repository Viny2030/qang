"""Tests for qang.neutrino (§151)."""

import math

import numpy as np
import pytest

from qang import neutrino as NU

qiskit = pytest.importorskip("qiskit")


def test_qg_z_is_the_spherical_law_of_cosines():
    for th in (0.1, 0.4, 0.7):
        for d in np.linspace(0, 2 * math.pi, 9):
            assert NU.two_flavour_qg_z(th, d) == pytest.approx(1 - 2 * NU.two_flavour_probability(th, d), abs=1e-12)


def test_qang_angle_at_maximum_is_four_theta():
    for th in (0.05, 0.3, 0.7):
        assert NU.two_flavour_qang_angle(th, math.pi) == pytest.approx(4 * th, abs=1e-12)
        assert NU.mixing_angle_from_maximum(NU.two_flavour_qg_z(th, math.pi)) == pytest.approx(th, abs=1e-12)


def test_two_flavour_circuit_matches_formula():
    from qiskit.quantum_info import Statevector

    for th in (0.2, 0.6):
        for d in (0.3, 1.7, math.pi):
            p = Statevector(NU.two_flavour_circuit(th, d, measure=False)).probabilities()[1]
            assert p == pytest.approx(NU.two_flavour_probability(th, d), abs=1e-12)


def test_pmns_is_unitary_and_probabilities_conserve():
    U = NU.pmns(0.58, 0.15, 0.85, -1.2)
    assert np.allclose(U @ U.conj().T, np.eye(3), atol=1e-12)
    P = NU.three_flavour_probabilities(1300, 2.0, -1.2)
    assert np.allclose(P.sum(axis=1), 1, atol=1e-12)
    assert np.allclose(P.sum(axis=0), 1, atol=1e-12)


def test_three_flavour_circuit_matches_formula_and_preserves_weight():
    from qiskit.quantum_info import Statevector

    for anti in (False, True):
        for dcp in (0.0, -math.pi / 2):
            for E in (0.8, 2.6):
                P = NU.three_flavour_probabilities(1300, E, dcp, anti)
                for a, f in enumerate(NU.FLAVOURS):
                    sv = Statevector(NU.three_flavour_circuit(f, 1300, E, dcp, anti, measure=False)).data
                    amp2 = np.abs(sv) ** 2
                    assert amp2[[1, 2, 4]].sum() == pytest.approx(1, abs=1e-12)
                    assert np.allclose(amp2[[1, 2, 4]], P[a], atol=1e-12)


def test_cp_asymmetry_vanishes_without_cp_violation():
    assert NU.cp_asymmetry(1300, 2.6, 0.0) == pytest.approx(0, abs=1e-12)
    assert NU.cp_asymmetry(1300, 2.6, -math.pi / 2) > 0.02
    assert NU.first_maximum_energy(1300) == pytest.approx(2.632, abs=1e-3)


def test_matter_reduces_to_vacuum_and_conserves_probability():
    P = NU.matter_probabilities(1300, 2.0, 0.7, 0.0)
    assert np.allclose(P, NU.three_flavour_probabilities(1300, 2.0, 0.7), atol=1e-12)
    Pm = NU.matter_probabilities(1300, 2.0, 0.7, 2.85)
    assert np.allclose(Pm.sum(axis=1), 1, atol=1e-12)
    assert NU.cp_asymmetry(1300, 2.6, 0.0, rho=2.85) > 0.01  # matter alone fakes CP violation


def test_matter_trotter_converges_and_preserves_weight():
    from qiskit.quantum_info import Statevector

    P = NU.matter_probabilities(1300, 2.6, -math.pi / 2, 2.85, antineutrino=True)
    errs = []
    for r in (2, 8):
        a2 = np.abs(Statevector(NU.matter_trotter_circuit("mu", 1300, 2.6, -math.pi / 2, 2.85, r,
                                                          antineutrino=True, measure=False)).data) ** 2
        assert a2[[1, 2, 4]].sum() == pytest.approx(1, abs=1e-12)
        errs.append(np.abs(a2[[1, 2, 4]] - P[1]).max())
    assert errs[1] < errs[0] / 8  # second order: error ~ r^-2


def test_collective_hamiltonian_conserves_mass_number_not_flavour_number():
    H = NU.collective_mass_hamiltonian([0.25, 0.5, 0.75, 1.0], 2.0)
    w = np.array([bin(k).count("1") for k in range(16)])
    assert np.abs(H * w[None, :] - w[:, None] * H).max() < 1e-12
    P0 = NU.collective_exact([0.25, 0.5, 0.75, 1.0], 2.0, 0.195, "eexx", 0.0, measure_basis="flavour")
    P1 = NU.collective_exact([0.25, 0.5, 0.75, 1.0], 2.0, 0.195, "eexx", 3.0, measure_basis="flavour")
    assert np.allclose(P0, [0, 0, 1, 1], atol=1e-12)
    assert abs(P1.sum() - P0.sum()) > 1e-3  # total flavour number is not conserved
    sec = NU.mass_sector_probabilities("eexx", 0.195)
    assert sec.sum() == pytest.approx(1, abs=1e-12)
    P = NU.collective_exact([0.25, 0.5, 0.75, 1.0], 2.0, 0.195, "eexx", 3.0)
    assert P.sum() == pytest.approx(np.dot(np.arange(5), sec), abs=1e-12)  # total nu_2 number is


def test_collective_trotter_converges_to_exact():
    from qiskit.quantum_info import Statevector

    om, mu, th = [0.25, 0.5, 0.75, 1.0], 2.0, 0.195
    idx = np.arange(16)
    for ib, ini, mb in (("mass", "0011", "mass"), ("flavour", "eexx", "flavour")):
        ex = NU.collective_exact(om, mu, th, ini, 2.0, ib, mb)
        errs = []
        for r in (4, 16):
            p = Statevector(NU.collective_trotter_circuit(om, mu, th, ini, 2.0, r, ib, mb, measure=False)).probabilities()
            errs.append(np.abs(np.array([p[(idx >> i) & 1 == 1].sum() for i in range(4)]) - ex).max())
        assert errs[1] < errs[0] / 4
