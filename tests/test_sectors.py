"""Tests for qang.sectors.reweight_sectors (§154)."""

import numpy as np
import pytest


def test_reweight_sectors_generalises_the_filter():
    from qang.sectors import filter_distribution, reweight_sectors

    rng = np.random.default_rng(3)
    p = rng.random(16)
    f, kept = filter_distribution(p, 4, 2)
    r, kept_r = reweight_sectors(p, 4, [0, 0, 1, 0, 0])
    assert np.allclose(f, r) and kept == pytest.approx(kept_r)
    target = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    r, _ = reweight_sectors(p, 4, target)
    w = np.array([bin(k).count("1") for k in range(16)])
    assert np.allclose(np.bincount(w, weights=r, minlength=5), target)


def test_stabilizer_shot_budget_matches_brute_force():
    """2026 context: the 0/1-knapsack behind arXiv:2607.01072 (optimal QED+PEC symmetry-check
    selection under a cost budget), checked exactly against brute force. Status: sin_verificar."""
    from itertools import combinations

    from qang.sectors import stabilizer_shot_budget

    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(1, 9))
        w = rng.uniform(0, 5, n)
        c = rng.integers(0, 6, n)
        budget = int(rng.integers(0, 15))
        chosen, got = stabilizer_shot_budget(w, c, budget)

        assert sum(c[i] for i in chosen) <= budget
        assert abs(sum(w[i] for i in chosen) - got) < 1e-9

        want = max(
            (sum(w[i] for i in combo)
             for k in range(n + 1)
             for combo in combinations(range(n), k)
             if sum(c[i] for i in combo) <= budget),
            default=0.0,
        )
        assert abs(got - want) < 1e-9


def test_stabilizer_shot_budget_validates_input():
    from qang.sectors import stabilizer_shot_budget

    with pytest.raises(ValueError):
        stabilizer_shot_budget([1.0, 2.0], [1], 5)  # mismatched lengths
    with pytest.raises(ValueError):
        stabilizer_shot_budget([1.0], [1], -1)  # negative budget
    with pytest.raises(ValueError):
        stabilizer_shot_budget([1.0], [-1], 5)  # negative cost
    with pytest.raises(ValueError):
        stabilizer_shot_budget([1.0] * 21, [1] * 21, 5)  # over exact_max_n
