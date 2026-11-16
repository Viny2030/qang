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


# ---- una sola base de lectura (§34.13) ----
from qang.qlimits import (path_graph_state, ghz_state_n, diagonal_twin, purity, purity_range_from_z, pure_state_from_z,
                          total_correlation_z, shannon_bits, local_purities, partial_transpose_min_eig, witness_path,
                          witness_value, witness_radius, hoeffding_shots, sufficient_shots_entropy, plugin_cz_bias_uniform,
                          dephased_path3, product_amp_damped, relative_entropy_coherence_z, von_neumann_bits, apply_local,
                          HADAMARD, _I2)


def _rho(psi):
    return np.outer(psi, psi.conj())


def test_single_basis_no_go_graph_vs_product_vs_mixed():
    for n in (2, 3):
        g, pl = path_graph_state(n), plus_state(n)
        p = z_distribution(g)
        assert np.allclose(p, z_distribution(pl)) and np.allclose(p, 1 / 2 ** n)
        assert abs(total_correlation_z(p, n)) < 1e-12 and abs(shannon_bits(p) - n) < 1e-12
        assert np.allclose(local_purities(_rho(pl), n), 1.0)
        assert np.allclose(local_purities(_rho(g), n), 0.5)
    assert abs(purity(np.eye(8) / 8) - 1 / 8) < 1e-12


def test_ct_false_positive_and_pairwise_hidden_dependence():
    p = z_distribution(ghz_state_n(3))
    assert abs(total_correlation_z(p, 3) - 2) < 1e-12 and abs(total_correlation_z(diagonal_twin(p).real.diagonal(), 3) - 2) < 1e-12
    even = np.zeros(8); even[[0, 3, 5, 6]] = 0.25
    assert abs(shannon_bits(even) - 2) < 1e-12 and abs(total_correlation_z(even, 3) - 1) < 1e-12


def test_purity_range_from_z_endpoints():
    p = np.array([0.6, 0.15, 0.2, 0.05])
    lo, hi = purity_range_from_z(p)
    assert abs(purity(diagonal_twin(p)) - lo) < 1e-12 and abs(purity(_rho(pure_state_from_z(p))) - hi) < 1e-12


def test_witness_path3_certifies_graph_only():
    W, g = witness_path(3), path_graph_state(3)
    assert abs(witness_value(W, _rho(g)) + 1) < 1e-12
    assert abs(witness_value(W, _rho(plus_state(3))) - 2) < 1e-12 and abs(witness_value(W, np.eye(8) / 8) - 2) < 1e-12
    assert abs(witness_radius(3, 1000) - 0.293556) < 1e-6 and -1 + witness_radius(3, 1000) < 0


def test_dephasing_trajectory_freezes_z_but_not_witness():
    W = witness_path(3)
    for t in (0.0, 1 / 3, 1.0):
        r = dephased_path3(t)
        assert np.allclose(np.real(np.diag(r)), 1 / 8)
        assert abs(witness_value(W, r) - (-1 + 3 * t)) < 1e-12


def test_local_basis_change_moves_entropy_not_entanglement():
    g = path_graph_state(2)
    b = apply_local(g, 2, {1: HADAMARD})
    p = z_distribution(b)
    assert abs(shannon_bits(p) - 1) < 1e-12 and abs(total_correlation_z(p, 2) - 1) < 1e-12
    assert abs(purity(_rho(b)) - 1) < 1e-12 and partial_transpose_min_eig(_rho(b), 2) < -0.4


def test_coherence_equals_HZ_only_when_pure():
    g = _rho(path_graph_state(2)); m = np.eye(4) / 4
    assert abs(relative_entropy_coherence_z(g) - 2) < 1e-12 and abs(relative_entropy_coherence_z(m)) < 1e-12


def test_amplitude_damping_product_nonmonotone_entropy_and_purity():
    h = [product_amp_damped(x)["H_Z"] for x in (0.0, 0.5, 1.0)]
    pu = [product_amp_damped(x)["P"] for x in (0.0, 0.5, 1.0)]
    assert np.allclose(h, [0, 2, 0]) and np.allclose(pu, [1, 0.25, 1])


def test_shot_budgets_and_bias():
    assert hoeffding_shots(0.05, 0.05) == 2952
    assert [sufficient_shots_entropy(2, .05, .05, "H"), sufficient_shots_entropy(3, .05, .05, "C")] == [133966, 3973513]
    assert abs(plugin_cz_bias_uniform(3, 100) - 4 / (200 * np.log(2))) < 1e-15
