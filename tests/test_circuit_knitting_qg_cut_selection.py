"""
Tests for examples/circuit_knitting_qg_cut_selection.py: choosing which
gates to cut with the qg cost formula.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np
import pytest

from circuit_knitting_qg_cut_selection import (  # noqa: E402
    COUNT_LABELS,
    E2E_GATES,
    QG_LABELS,
    _cut_mask,
    capacity_partitions,
    gate_log_cost,
    instance,
    rule_overheads,
)


def _bell(n):
    b = [1]
    row = [1]
    for _ in range(n):
        new = [row[-1]]
        for x in row:
            new.append(new[-1] + x)
        row = new
        b.append(row[0])
    return b[n]


def test_partition_enumeration():
    assert len(capacity_partitions(5, 5)) == _bell(5)
    parts = capacity_partitions(8, 4)
    assert len(parts) == len(set(parts)) == 3795
    assert all(max(np.bincount(p)) <= 4 for p in parts)


def test_gate_cost_is_periodic_and_free_at_multiples_of_pi():
    assert gate_log_cost(0.0) == pytest.approx(0.0, abs=1e-12)
    assert gate_log_cost(math.pi) == pytest.approx(0.0, abs=1e-6)
    assert gate_log_cost(math.pi / 2) == pytest.approx(math.log(9.0))
    assert gate_log_cost(0.3) == pytest.approx(gate_log_cost(math.pi - 0.3))


def test_qg_rule_is_never_worse_than_the_other_rules():
    parts = np.array(capacity_partitions(8, 4))
    for kind in ("trotter", "qaoa"):
        for s in range(15):
            r = rule_overheads(instance(kind, s), parts)
            assert r["qg"] <= r["count"] + 1e-9 and r["qg"] <= r["weakest"] + 1e-9


def test_weakest_bond_rule_fails_with_large_angles():
    parts = np.array(capacity_partitions(8, 4))
    gaps = [rule_overheads(instance("qaoa", s), parts) for s in range(30)]
    assert np.mean([g["weakest"] - g["qg"] for g in gaps]) > math.log(1.5)


def test_e2e_labels_are_the_two_rules_choices():
    parts = np.array(capacity_partitions(6, 3))
    cut = _cut_mask(E2E_GATES, parts)
    logc = np.array([gate_log_cost(t) for *_, t in E2E_GATES])
    by_count = parts[cut.sum(axis=1) == cut.sum(axis=1).min()]
    assert len(by_count) == 1
    as_str = lambda p: "".join("AB"[x] for x in p)  # noqa: E731
    assert as_str(by_count[0]) == COUNT_LABELS
    assert as_str(parts[np.argmin(cut @ logc)]) == QG_LABELS


def test_addon_find_cuts_matches_the_qg_optimum():
    pytest.importorskip("qiskit_addon_cutting")
    from circuit_knitting_qg_cut_selection import addon_log_overhead

    parts = np.array(capacity_partitions(8, 4))
    for kind in ("trotter", "qaoa"):
        for s in range(3):
            g = instance(kind, s)
            assert addon_log_overhead(g) == pytest.approx(rule_overheads(g, parts)["qg"], abs=1e-6)


def test_e2e_qg_cut_has_lower_error_with_proportional_shots():
    pytest.importorskip("qiskit_addon_cutting")
    pytest.importorskip("qiskit_aer")
    from circuit_knitting_qg_cut_selection import e2e_errors

    e = e2e_errors(total_shots=60000, seeds=range(3), allocation="proportional")
    assert e["qg"]["overhead"] == pytest.approx(7.44, abs=0.01)
    assert e["count"]["overhead"] == pytest.approx(81.0)
    assert e["qg"]["rmse"] < 0.6 * e["count"]["rmse"]
