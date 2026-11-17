import numpy as np

from qang.qagent import (born_policy, exploration_floor, score, expected_gradient, gradient_variance, gradient_second_moment,
                         episodes_for_gradient_sign, logit_of, flow_theta, time_to_wrong_probability, logit_vanilla_time,
                         decision_report, neyman_allocation, allocation_variance, uniform_to_neyman_ratio, run_bandit)


def test_policy_fisher_one_and_exact_gradient():
    for t in (0.3, 1.1, 2.7):
        p0, p1 = born_policy(t)
        assert abs(p0 * score(0, t) ** 2 + p1 * score(1, t) ** 2 - 1) < 1e-12
        m = p0 * 0.9 * score(0, t) + p1 * 0.3 * score(1, t)
        assert abs(m - expected_gradient(t, 0.9, 0.3)) < 1e-12
        e2 = p0 * (0.9 * score(0, t)) ** 2 + p1 * (0.3 * score(1, t)) ** 2
        assert abs(e2 - gradient_second_moment(t, 0.9, 0.3)) < 1e-12
        assert abs(gradient_variance(t, 0.9, 0.3) - (e2 - m * m)) < 1e-12


def test_flow_is_natural_gradient_of_logit():
    for t0 in (0.4, 1.3, 2.2):
        assert abs(logit_of(flow_theta(t0, 0.1, 0.5, 7.0)) - (logit_of(t0) + 0.1 * 0.5 * 7.0)) < 1e-9
    assert abs(time_to_wrong_probability(np.pi / 2, 1, 1, 0.01) - np.log(99)) < 1e-12
    assert logit_vanilla_time(0.0, 1, 1, 0.01) > 20 * time_to_wrong_probability(np.pi / 2, 1, 1, 0.01)


def test_pole_trap_and_floor():
    assert episodes_for_gradient_sign(0.0, 0.9, 0.3) == float("inf")
    assert episodes_for_gradient_sign(0.05, 0.9, 0.3) > 50 * episodes_for_gradient_sign(np.pi / 2, 0.9, 0.3)
    p = run_bandit(np.pi - 1e-9, (0.7, 0.3), 0.05, 500, np.random.default_rng(0))
    assert p[-1] == p[0]
    assert abs(np.sin(exploration_floor(0.02) / 2) ** 2 - 0.02) < 1e-15
    q = run_bandit(np.pi / 2, (0.7, 0.3), 0.05, 300, np.random.default_rng(1), p_floor=0.01)
    assert q.min() >= exploration_floor(0.01) - 1e-12 and q.max() <= np.pi - exploration_floor(0.01) + 1e-12


def test_decision_report_and_allocation():
    d = decision_report(np.pi / 2)
    assert d["shots_for_sign"] == float("inf") and abs(d["qg_S"] - 1) < 1e-12
    assert decision_report(0.3)["greedy_action"] == 0.0 and decision_report(2.8)["greedy_action"] == 1.0
    th = [0.2, 1.0, 2.9]
    n = neyman_allocation(th, 900)
    assert abs(n.sum() - 900) < 1e-9
    assert allocation_variance(th, n) <= allocation_variance(th, np.full(3, 300.0)) + 1e-15
    assert abs(uniform_to_neyman_ratio([1.0, 1.0]) - 1.0) < 1e-12 and uniform_to_neyman_ratio(th) > 1.0
