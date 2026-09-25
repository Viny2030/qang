"""
Tests for examples/qaoa_k_constraint_qg.py: QAOA for maximum K-vertex
cover with the qg constraint filter, qg starts and the XY baseline.
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("scipy")

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

import qaoa_k_constraint_qg as Q  # noqa: E402


def test_constraint_is_the_register_mean_qg_z():
    for k in range(Q.N_NODES + 1):
        mask = Q.WEIGHT == k
        mean_qg = (1 - 2 * Q.BITS[mask] ).mean(axis=1)
        assert np.allclose(mean_qg, 1 - 2 * k / Q.N_NODES)
    assert Q.QG_BUDGET == pytest.approx(0.25)


def test_dicke_state_is_uniform_over_weight_k():
    qc = QuantumCircuit(Q.N_NODES)
    Q.dicke_state(qc)
    psi = Statevector(qc).data
    target = np.zeros(2**Q.N_NODES)
    for c in itertools.combinations(range(Q.N_NODES), Q.K_CHOSEN):
        target[sum(1 << i for i in c)] = 1
    target /= np.linalg.norm(target)
    assert abs(np.vdot(target, psi)) ** 2 == pytest.approx(1.0, abs=1e-12)


def test_ising_decomposition_reproduces_the_cost():
    edges = Q.random_graph(1)
    for v in ("standard", "xy"):
        f = Q.objective_values(edges, v)
        h, J = Q.ising_terms(f)
        z = 1 - 2 * Q.BITS
        rec = f.mean() + sum(c * z[:, i] for i, c in h.items()) + sum(c * z[:, i] * z[:, j] for (i, j), c in J.items())
        assert np.allclose(rec, f)


def test_penalty_ground_state_is_the_constrained_optimum():
    for g in range(5):
        edges = Q.random_graph(g)
        opt, sols = Q.brute_force(edges)
        f = Q.objective_values(edges, "standard")
        assert set(np.nonzero(f == f.min())[0]) == set(sols)


def test_classical_baselines():
    edges = Q.random_graph(0)
    opt, sols = Q.brute_force(edges)
    assert opt == 10 and len(sols) == 3
    assert Q.greedy(edges) <= opt
    c = Q.lp_relaxation(edges)
    assert c.sum() == pytest.approx(Q.K_CHOSEN) and np.all((c > -1e-9) & (c < 1 + 1e-9))


def test_starts_in_qg_units():
    edges = Q.random_graph(2)
    zero = np.zeros(2)
    for v, expected in (("standard", 0.0), ("qg_budget", Q.QG_BUDGET)):
        p = Q.ideal_distribution(edges, v, zero)
        assert p @ (1 - 2 * Q.WEIGHT / Q.N_NODES) == pytest.approx(expected, abs=1e-12)
    qg = Q.start_qg("qg_warm", edges)
    assert np.all(np.abs(qg) <= 0.5 + 1e-12)  # pole damping


def test_xy_qaoa_conserves_the_constraint_exactly():
    edges = Q.random_graph(3)
    p = Q.ideal_distribution(edges, "xy", np.array([0.4, 0.3, 0.7, 1.1]))
    assert p[~Q.FEASIBLE].sum() < 1e-12


def test_filter_metrics():
    edges = Q.random_graph(0)
    uniform = np.full(2**Q.N_NODES, 1 / 2**Q.N_NODES)
    m = Q.sample_metrics(uniform, edges, filtered=True)
    base = Q.random_feasible_baseline(edges)
    assert m["p_opt"] == pytest.approx(base["p_opt"]) and m["ratio"] == pytest.approx(base["ratio"])
    assert m["kept"] == pytest.approx(Q.FEASIBLE.mean())


def test_noisy_xy_qaoa_filter_detects_and_removes_errors():
    pytest.importorskip("qiskit_aer")
    rows = Q.experiment(graph_seeds=[0], p=1, device="all_to_all", shots=4000)
    xy = next(r for r in rows if r["variant"] == "xy")
    assert xy["ideal"]["kept"] == pytest.approx(1.0)
    assert xy["noisy"]["kept"] < 0.8  # witness: errors leave the constraint subspace
    assert xy["noisy_f"]["p_opt"] > 1.5 * xy["noisy"]["p_opt"]
    std = next(r for r in rows if r["variant"] == "standard")
    base = next(r for r in rows if r["variant"] == "random_feasible")
    assert abs(std["noisy_f"]["ratio"] - base["ratio"]) < 0.1  # penalty QAOA p=1 ~ random feasible guess
