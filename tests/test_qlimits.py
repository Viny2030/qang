import numpy as np
from qang.qlimits import (ring_graph_state, plus_state, z_distribution, stabilizer_expectation, damp_distribution, t1_noise_scan,
                          is_monotone, best_cost_clipped, qft_gates, ghz_gates, unary_rbs_gates, circuit_unitary, preserves_weight,
                          weight_breakers, weight1_orthogonal, weight1_qgz, simulate_qgz_weight1, in_sector_error, mean_qgz)


def test_graph_state_blind_in_z():
    g, p = ring_graph_state(6), plus_state(6)
    assert np.max(np.abs(z_distribution(g) - z_distribution(p))) == 0.0
    assert np.allclose([stabilizer_expectation(g, i, 6) for i in range(6)], 1.0)
    assert np.allclose([stabilizer_expectation(p, i, 6) for i in range(6)], 0.0)


def test_t1_mean_qgz_monotone_and_qgs_not():
    rng = np.random.default_rng(1)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    p = z_distribution(psi / np.linalg.norm(psi))
    s, z = t1_noise_scan(p, 4, np.linspace(0, 0.5, 51))
    assert is_monotone(z, True) and not is_monotone(s, True) and not is_monotone(s, False)
    assert abs(damp_distribution(p, 4, 0.3).sum() - 1) < 1e-12


def test_arccos_range_trap():
    e, th = best_cost_clipped(-0.5)
    assert abs(e + np.cos(0.5)) < 1e-9 and th == 0.0


def test_weight_analyzer():
    n = 4
    assert not preserves_weight(circuit_unitary(n, qft_gates(n)), n) and len(weight_breakers(n, qft_gates(n))) == 4
    assert not preserves_weight(circuit_unitary(n, ghz_gates(n)), n) and len(weight_breakers(n, ghz_gates(n))) == 4
    assert preserves_weight(circuit_unitary(n, unary_rbs_gates(n, [.3, .7, 1.1])), n)


def test_weight1_quadratic_form():
    rng = np.random.default_rng(2)
    th = rng.uniform(-3, 3, 4); v = rng.normal(size=5); v /= np.linalg.norm(v)
    assert np.max(np.abs(weight1_qgz(weight1_orthogonal(5, th), v) - simulate_qgz_weight1(5, th, v))) < 1e-13


def test_in_sector_error_passes_filter():
    r = in_sector_error(0.3)
    assert abs(r["error_retained"] - 1) < 1e-12 and abs(r["shift_mqg"]) > 100
