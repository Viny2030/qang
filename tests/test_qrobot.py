import numpy as np
import pytest

from qang.qrobot import (planar_fk, planar_jacobian, two_link_ik, joint_readings, joint_from_readings, joint_variance_z,
                         joint_variance_atan2, joint_crb_two_bases, joint_sigma_qga, shots_for_joint, end_effector_rms,
                         shots_for_end_effector, shots_per_cycle, realtime_joint_sigma, max_control_rate,
                         steady_state_variance, kalman_scalar_track, grover_optimal_iterations, grover_success,
                         grover_simulate, classical_expected_queries, grover_query_ratio, preflight, joint_parameter_shift)

LINKS = (1.0, 0.8)


def test_ik_round_trip_and_unreachable():
    for el in (1, -1):
        p = planar_fk([0.4, 0.9], LINKS)
        s = two_link_ik(p, *LINKS, elbow=el)
        assert np.allclose(planar_fk(s, LINKS), p, atol=1e-10)
    assert two_link_ik([5.0, 0.0], *LINKS) is None


def test_jacobian_matches_finite_differences():
    t = np.array([0.3, -0.7, 1.1])
    L = [1.0, 0.7, 0.4]
    Jn = np.stack([(planar_fk(t + e, L) - planar_fk(t - e, L)) / 2e-6 for e in 1e-6 * np.eye(3)], axis=1)
    assert np.allclose(planar_jacobian(t, L), Jn, atol=1e-6)


def test_joint_readings_sign_trap_and_variances():
    r = joint_readings(-0.7)
    assert abs(joint_from_readings(r["qg_Z"], r["qg_X"]) + 0.7) < 1e-12
    assert abs(np.arccos(r["qg_Z"]) - 0.7) < 1e-12            # qg_Z solo no distingue el signo
    for th in (0.3, 0.7853981633974483, 1.2):
        assert joint_variance_atan2(th, 100) >= joint_crb_two_bases(100) - 1e-15
    assert abs(joint_variance_atan2(np.pi / 4, 100) - joint_crb_two_bases(100)) < 1e-15
    assert abs(joint_variance_z(100) - 0.01) < 1e-15
    assert abs(joint_sigma_qga(1) - 1000 / (2 * np.pi)) < 1e-12


def test_shots_budgets():
    assert shots_for_joint(0.1, z=1.0) == 100
    th = [0.3, 0.9]
    n = shots_for_end_effector(th, LINKS, 0.01)
    assert end_effector_rms(th, LINKS, n) <= 0.01 < end_effector_rms(th, LINKS, n - 1)
    assert shots_per_cycle(100, 1e-3) == 10 and shots_per_cycle(1000, 1e-3, 0.002) == 0
    assert realtime_joint_sigma(1000, 1e-3, 0.002) == float("inf")
    assert abs(max_control_rate(0.01, 1e-3) - 0.1) < 1e-12


def test_preflight_flags_infeasible_loop():
    th = two_link_ik((1.2, 0.9), *LINKS)
    bad = preflight(th, LINKS, 0.02, 100, 1e-3)
    good = preflight(th, LINKS, 0.02, 0.1, 1e-3)
    assert bad["feasible"] == 0.0 and bad["shots_available"] == 10 and bad["shots_required"] == 7225
    assert good["feasible"] == 1.0 and good["ee_rms"] <= 0.02


def test_kalman_steady_state():
    q, r = 1e-4, 1e-2
    prior, post = steady_state_variance(q, r)
    assert abs(prior - post - q) < 1e-15
    assert abs(post - prior * r / (prior + r)) < 1e-15        # punto fijo de la actualización
    _, ps = kalman_scalar_track(np.zeros(300), r, q, p0=1.0)
    assert abs(ps[-1] - post) < 1e-9


def test_grover_formula_matches_simulation_and_scaling():
    for N, M in ((16, 1), (64, 3), (1000, 7)):
        k = grover_optimal_iterations(N, M)
        assert abs(grover_success(N, M, k) - grover_simulate(N, list(range(M)), k)) < 1e-12
    assert grover_optimal_iterations(1024, 1) == 25
    assert classical_expected_queries(1000, 5) == pytest.approx(1001 / 6)
    assert grover_query_ratio(2 ** 20, 1) == pytest.approx(2 / np.pi * np.sqrt(2 ** 20), rel=0.01)


def test_parameter_shift_is_minus_sin():
    for th in (-1.3, 0.2, 2.0):
        assert abs(joint_parameter_shift(th) + np.sin(th)) < 1e-12
