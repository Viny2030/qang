import math

import pytest

from quang.gradients import (
    benchmark,
    inverse_jacobian_clipped,
    inverse_jacobian_raw,
    inverse_jacobian_tikhonov,
    jacobian,
    run_gradient_descent,
    toy_vqe_energy,
    toy_vqe_grad_theta,
)

PI = math.pi


# --------------------------------------------------------------------- #
# Section 4.1's stated limitation: the raw Jacobian vanishes at the poles,
# so its inverse diverges there.
# --------------------------------------------------------------------- #
def test_jacobian_vanishes_at_poles():
    assert jacobian(0.0) == pytest.approx(0.0, abs=1e-9)
    assert jacobian(PI) == pytest.approx(0.0, abs=1e-9)


def test_raw_inverse_jacobian_diverges_at_poles():
    assert math.isinf(inverse_jacobian_raw(0.0))
    assert math.isinf(inverse_jacobian_raw(PI))


def test_raw_inverse_jacobian_grows_near_poles():
    # paper's own worked numbers: 1/sin(theta) at theta=0.001 rad exceeds 1000
    assert abs(inverse_jacobian_raw(0.001)) > 1000.0


def test_raw_inverse_jacobian_matches_exact_away_from_poles():
    assert inverse_jacobian_raw(PI / 2) == pytest.approx(-1.0, abs=1e-9)


# --------------------------------------------------------------------- #
# Regularized inverses: bounded everywhere, converge to the exact value
# away from the poles as eps -> 0.
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("theta", [0.0, 1e-6, PI])
def test_clipped_inverse_jacobian_is_finite_at_poles(theta):
    val = inverse_jacobian_clipped(theta, eps=0.05)
    assert math.isfinite(val)
    assert abs(val) == pytest.approx(1.0 / 0.05, rel=1e-6)


@pytest.mark.parametrize("theta", [0.0, 1e-6, PI])
def test_tikhonov_inverse_jacobian_is_finite_at_poles(theta):
    val = inverse_jacobian_tikhonov(theta, eps=0.05)
    assert math.isfinite(val)


def test_regularizations_converge_to_exact_away_from_poles():
    theta = PI / 2
    exact = inverse_jacobian_raw(theta)
    assert inverse_jacobian_clipped(theta, eps=0.001) == pytest.approx(exact, abs=1e-3)
    assert inverse_jacobian_tikhonov(theta, eps=0.001) == pytest.approx(exact, abs=1e-3)


# --------------------------------------------------------------------- #
# toy VQE loss sanity
# --------------------------------------------------------------------- #
def test_toy_vqe_energy_matches_qg_z_when_hx_zero():
    for theta in (0.1, 1.0, 2.5):
        assert toy_vqe_energy(theta, h_z=1.0, h_x=0.0) == pytest.approx(math.cos(theta), abs=1e-9)


def test_toy_vqe_gradient_matches_finite_difference():
    h_z, h_x = 1.0, 0.3
    theta = 1.234
    eps = 1e-6
    fd = (
        toy_vqe_energy(theta + eps, h_z, h_x) - toy_vqe_energy(theta - eps, h_z, h_x)
    ) / (2 * eps)
    assert toy_vqe_grad_theta(theta, h_z, h_x) == pytest.approx(fd, abs=1e-6)


# --------------------------------------------------------------------- #
# convergence behavior: the whole point of Future Research Direction #3
# --------------------------------------------------------------------- #
def test_theta_space_gd_converges_from_near_pole_start():
    hist = run_gradient_descent("theta", theta0=0.05, lr=0.1, steps=300)
    assert not hist.diverged
    # gradient should shrink close to zero near the optimum
    assert abs(toy_vqe_grad_theta(hist.theta[-1])) < 0.05


def test_raw_qg_space_gd_is_numerically_unstable_from_near_pole_start():
    """This is the concrete demonstration of the paper's Section 4.1 limitation:
    starting close to a pole, the *raw* qg-space update (which uses the exact,
    unbounded -1/sin(theta) conversion factor) takes wildly oversized steps."""
    hist = run_gradient_descent("qg_raw", theta0=0.01, lr=0.1, steps=50)
    step_sizes = [abs(hist.theta[i + 1] - hist.theta[i]) for i in range(len(hist.theta) - 1)]
    assert max(step_sizes) > 1.0  # a single step larger than the entire [0, pi] domain


@pytest.mark.parametrize("space", ["qg_clipped", "qg_tikhonov"])
def test_regularized_qg_space_gd_converges_from_near_pole_start(space):
    # Note: a regularized conversion factor is *bounded* (by ~1/eps), not
    # small -- with an unchanged lr=0.1 it can still comfortably outrun the
    # raw case's single-step size near the pole. What regularization buys
    # you is a *predictable* bound (lr/eps) instead of an unbounded one, so
    # here we use a smaller lr, appropriate to that bound, to see smooth
    # convergence rather than a first-step overshoot across the domain.
    hist = run_gradient_descent(space, theta0=0.01, lr=0.02, steps=400, eps=0.05)
    assert not hist.diverged
    step_sizes = [abs(hist.theta[i + 1] - hist.theta[i]) for i in range(len(hist.theta) - 1)]
    assert max(step_sizes) < 1.0  # no wild single-step jumps, unlike the raw case
    assert abs(toy_vqe_grad_theta(hist.theta[-1])) < 0.1


def test_benchmark_runs_all_four_spaces():
    results = benchmark(theta0=0.05, lr=0.05, steps=100)
    assert set(results.keys()) == {"theta", "qg_raw", "qg_clipped", "qg_tikhonov"}
    for hist in results.values():
        assert len(hist.theta) > 0
