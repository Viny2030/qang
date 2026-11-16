import numpy as np
import pytest

from qang.qstring import (colored_partitions, critical_dimension, closed_massless_decomposition, level_vector, level_sector_size,
                          distinct_partitions, level_from_z, mode_readings_mqg, level_from_readings, level_filter_damped,
                          level_vs_weight_blindness, t_dual_equal, massless_states, luttinger_k_exact, dual_exponents,
                          xxz_ground_distribution, luttinger_k_from_variance)


def test_string_counting_and_critical_dimension():
    assert colored_partitions(4, 24) == [1, 24, 324, 3200, 25650]
    assert critical_dimension() == 26
    d = closed_massless_decomposition(24)
    assert d["graviton"] + d["kalb_ramond"] + d["dilaton"] == d["total"] == 576


def test_level_is_linear_z_reading_and_sector_counts():
    assert [level_sector_size(N, 10) for N in range(11)] == distinct_partitions(10)
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(64))
    assert abs(level_from_readings(mode_readings_mqg(p, 6)) - level_from_z(p, 6)) < 1e-12


def test_level_filter_retention_and_blindness():
    r = level_filter_damped(0b101101, 6, 0.3)
    assert abs(r["kept"] - 0.7 ** 4) < 1e-12
    b = level_vs_weight_blindness(4)
    assert b["level_preserving"][0] == b["level_preserving"][1] and b["level_preserving"][2] != b["level_preserving"][3]
    assert b["weight_preserving"][2] == b["weight_preserving"][3] and b["weight_preserving"][0] != b["weight_preserving"][1]


def test_t_duality_and_self_dual_enhancement():
    for R in (0.7, 1.0, 1.3):
        assert t_dual_equal(R)
    assert massless_states(1.3) == 576 and massless_states(1.0) == 676 and massless_states(0.7) == massless_states(1 / 0.7)


def test_luttinger_formulas_and_small_chain():
    assert abs(luttinger_k_exact(0.0) - 1.0) < 1e-12 and abs(luttinger_k_exact(1.0) - 0.5) < 1e-12
    a, b = dual_exponents(1.5)
    assert abs(a * b - 1) < 1e-12
    pytest.importorskip("scipy")
    st, p = xxz_ground_distribution(12, 0.0)
    assert abs(luttinger_k_from_variance(st, p, 12) - 1.0) < 0.08
