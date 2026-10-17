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


def _exact_counts(qc):
    from qiskit.quantum_info import Statevector

    return {k: int(round(v * 1e9)) for k, v in Statevector(qc).probabilities_dict().items()}


@pytest.mark.parametrize("link", [(0, 4), (20, 8)])
def test_fidelity_from_three_correlators(link):
    L, N = link
    T = Q.fibre_transmission(L)
    for a in (0.3, 1.2):
        for assisted in (False, True):
            eta, kap = Q.link(L, N, assisted)
            v = [Q.correlator_from_counts(_exact_counts(Q.resource_circuit(a, T, N, b, measure=False)), assisted)[0]
                 for b in "XYZ"]
            assert Q.fidelity_from_correlators(*v) == pytest.approx(Q.average_fidelity(a, eta, kap), abs=1e-6)


@pytest.mark.parametrize("eta", [1.0, 0.25, 0.05])
def test_dual_rail_heralds_every_loss(eta):
    for a in (math.pi / 4, 0.4):
        r = Q.dual_rail_fidelity_from_counts(
            [_exact_counts(Q.dual_rail_teleport_circuit(a, eta, i, measure=False)) for i in range(6)])
        for k, v in Q.dual_rail_fidelities(a, eta).items():
            assert r[k] == pytest.approx(v, abs=1e-6)
    assert Q.dual_rail_fidelities(math.pi / 4, eta)["filtered"] == pytest.approx(1.0)
