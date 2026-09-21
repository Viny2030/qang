import math

import pytest

from qang.gradients import (
    benchmark,
    inverse_jacobian_clipped,
    inverse_jacobian_raw,
    inverse_jacobian_tikhonov,
    jacobian,
    pole_damping_factor,
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


def test_benchmark_runs_all_five_spaces():
    results = benchmark(theta0=0.05, lr=0.05, steps=100)
    assert set(results.keys()) == {
        "theta",
        "qg_raw",
        "qg_clipped",
        "qg_tikhonov",
        "theta_pole_damped",
    }
    for hist in results.values():
        assert len(hist.theta) > 0


# --------------------------------------------------------------------- #
# theta_pole_damped: a fifth space that stays entirely in theta-space
# (no qg round trip, so no arccos-range trapping), with a
# Levenberg-Marquardt-style damping factor derived from the same
# pole-proximity floor as the regularized Jacobians above.
# --------------------------------------------------------------------- #
def test_pole_damping_factor_is_one_away_from_poles():
    assert pole_damping_factor(PI / 2, eps=0.05) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("theta", [0.0, 1e-6, PI])
def test_pole_damping_factor_floors_at_eps_near_poles(theta):
    assert pole_damping_factor(theta, eps=0.05) == pytest.approx(0.05, abs=1e-9)


def test_pole_damping_factor_matches_sin_between_the_floor_and_one():
    theta = 0.3  # sin(0.3) ~= 0.2955, above the default eps=0.05 floor
    assert pole_damping_factor(theta, eps=0.05) == pytest.approx(abs(math.sin(theta)), abs=1e-9)


def test_theta_pole_damped_matches_plain_theta_at_a_safe_learning_rate():
    """Away from any divergence risk, damping should cost (at most) a
    negligible difference in the converged optimum -- it is not a
    different optimizer, just a step-size modulator that has no effect
    once the raw step is already safe."""
    h_z, h_x = 1.0, -0.3
    h_plain = run_gradient_descent("theta", theta0=0.02, lr=0.3, steps=300, h_z=h_z, h_x=h_x)
    h_damped = run_gradient_descent(
        "theta_pole_damped", theta0=0.02, lr=0.3, steps=300, eps=0.05, h_z=h_z, h_x=h_x
    )
    assert not h_plain.diverged and not h_damped.diverged
    assert h_damped.energy[-1] == pytest.approx(h_plain.energy[-1], abs=1e-6)


def test_theta_pole_damped_survives_an_aggressive_learning_rate_where_plain_fails():
    """The headline finding for this space (see qang.gradients' module
    docstring and examples/pole_damped_gradient_descent_robustness.py for
    the full statistical study, and examples/vqe_h2_qg_vs_theta.py for the
    same effect on a real molecule): at a badly-tuned, too-aggressive
    learning rate, plain theta-space gradient descent overshoots and never
    recovers the true minimum, while theta_pole_damped -- started from the
    exact same near-pole point -- still reaches it."""
    h_z, h_x = 1.0, -0.3
    true_min = -math.hypot(h_z, h_x)
    theta0 = 0.02

    for lr in (2.0, 3.0, 5.0):
        h_plain = run_gradient_descent("theta", theta0=theta0, lr=lr, steps=300, h_z=h_z, h_x=h_x)
        h_damped = run_gradient_descent(
            "theta_pole_damped", theta0=theta0, lr=lr, steps=300, eps=0.05, h_z=h_z, h_x=h_x
        )
        assert abs(h_plain.energy[-1] - true_min) > 1e-3  # plain fails to converge
        assert h_damped.energy[-1] == pytest.approx(true_min, abs=1e-6)  # damped succeeds


def test_theta_pole_damped_is_not_clamped_to_zero_pi_domain():
    """theta_pole_damped deliberately opts out of the [0, pi] domain clamp
    that the other spaces use (see run_gradient_descent's implementation):
    that clamp is what would otherwise reproduce the qg-space variants'
    arccos-range trapping. This checks the escape hatch is actually wired
    up, not just documented."""
    h_z, h_x = 1.0, -0.3
    hist = run_gradient_descent(
        "theta_pole_damped", theta0=0.02, lr=5.0, steps=5, eps=0.05, h_z=h_z, h_x=h_x
    )
    assert any(theta < 0.0 or theta > math.pi for theta in hist.theta)
