"""Tests for qang.qpon (§157)."""

import math

import pytest

from qang import qpon as Q

qiskit = pytest.importorskip("qiskit")


def test_optimum_is_the_maximum_of_the_closed_form():
    for eta, kap in ((0.25, 1.0), (0.05, 1.0), (0.25, 0.25), (0.1, 0.7)):
        a_star = Q.optimal_angle(eta, kap)
        F_star = Q.average_fidelity(a_star, eta, kap)
        assert F_star == pytest.approx(Q.optimal_fidelity(eta, kap), abs=1e-12)
        for d in (-0.05, 0.05):
            assert Q.average_fidelity(a_star + d, eta, kap) < F_star
        assert math.cos(2 * a_star) == pytest.approx(Q.optimal_qg_z(eta, kap), abs=1e-12)
        assert Q.resource_angle_from_qg_z(Q.optimal_qg_z(eta, kap)) == pytest.approx(a_star, abs=1e-12)


def test_limits():
    assert Q.optimal_fidelity(1.0) == pytest.approx(1.0)  # lossless: Bell state, perfect teleportation
    assert Q.optimal_angle(1.0) == pytest.approx(math.pi / 4)
    eta, kap = Q.link(0, 8, assisted=True)  # no fibre loss: the splitter is fully heralded
    assert Q.optimal_fidelity(eta, kap) == pytest.approx(1.0)
    assert math.tan(Q.optimal_angle(eta, kap)) == pytest.approx(math.sqrt(8))
    eta, kap = Q.link(0, 8)
    assert Q.average_fidelity(math.pi / 4, eta, kap) < Q.CLASSICAL_FIDELITY < Q.optimal_fidelity(eta, kap)


@pytest.mark.parametrize("link", [(0, 4), (20, 8), (10, 1)])
def test_circuit_matches_closed_form(link):
    from qiskit.quantum_info import Statevector

    L, N = link
    T = Q.fibre_transmission(L)
    for a in (0.3, math.pi / 4, 1.2):
        cs = [{k: int(round(v * 1e9)) for k, v in
               Statevector(Q.teleport_circuit(a, T, N, i, measure=False)).probabilities_dict().items()}
              for i in range(6)]
        for assisted in (False, True):
            eta, kap = Q.link(L, N, assisted)
            r = Q.fidelity_from_counts(cs, assisted)
            assert r["fidelity"] == pytest.approx(Q.average_fidelity(a, eta, kap), abs=1e-6)
            assert r["kept"] == pytest.approx(Q.heralding_probability(a, kap), abs=1e-6)
