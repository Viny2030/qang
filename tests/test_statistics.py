import math

import pytest

from quang.statistics import (
    confidence_interval_theta,
    p0_from_theta,
    propagated_theta_std,
    propagated_theta_variance,
    std_qg_z_shot_noise,
    var_qg_z_shot_noise,
)

PI = math.pi


def test_p0_from_theta_matches_born_rule():
    assert p0_from_theta(0.0) == pytest.approx(1.0, abs=1e-9)
    assert p0_from_theta(PI) == pytest.approx(0.0, abs=1e-9)
    assert p0_from_theta(PI / 2) == pytest.approx(0.5, abs=1e-9)


def test_var_qg_z_shot_noise_matches_binomial_formula():
    theta, n = 1.2, 1000
    p0 = p0_from_theta(theta)
    expected = 4.0 * p0 * (1.0 - p0) / n
    assert var_qg_z_shot_noise(theta, n) == pytest.approx(expected, abs=1e-12)


def test_var_qg_z_shot_noise_equals_sin_squared_over_n():
    """The closed-form identity p0*(1-p0) = sin^2(theta)/4 used throughout the module."""
    for theta in (0.3, 1.0, 1.5, 2.2, 2.9):
        n = 500
        assert var_qg_z_shot_noise(theta, n) == pytest.approx(math.sin(theta) ** 2 / n, abs=1e-9)


def test_var_qg_z_shot_noise_vanishes_at_poles():
    n = 1000
    assert var_qg_z_shot_noise(1e-9, n) == pytest.approx(0.0, abs=1e-6)
    assert var_qg_z_shot_noise(PI - 1e-9, n) == pytest.approx(0.0, abs=1e-6)


def test_n_shots_must_be_positive():
    with pytest.raises(ValueError):
        var_qg_z_shot_noise(1.0, 0)


# --------------------------------------------------------------------- #
# the central (and somewhat surprising) result: away from the poles, the
# propagated theta variance is ~1/n_shots REGARDLESS of theta
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("theta", [0.3, 0.8, PI / 2, 2.0, 2.8])
def test_propagated_theta_variance_is_theta_independent_away_from_poles(theta):
    n = 2000
    assert propagated_theta_variance(theta, n) == pytest.approx(1.0 / n, abs=1e-9)


def test_propagated_theta_std_scales_as_inverse_sqrt_n():
    theta = PI / 2
    for n in (100, 400, 1600):
        assert propagated_theta_std(theta, n) == pytest.approx(1.0 / math.sqrt(n), abs=1e-9)


def test_propagated_theta_variance_is_infinite_exactly_at_a_pole():
    # the delta-method linear approximation breaks down there (0 * inf)
    assert math.isinf(propagated_theta_variance(0.0, 1000))
    assert math.isinf(propagated_theta_variance(PI, 1000))


def test_confidence_interval_theta_is_symmetric_and_shrinks_with_n():
    theta_hat = 1.0
    lo95, hi95 = confidence_interval_theta(theta_hat, n_shots=1000, confidence=0.95)
    assert lo95 < theta_hat < hi95
    assert (theta_hat - lo95) == pytest.approx(hi95 - theta_hat, abs=1e-9)

    lo95_more, hi95_more = confidence_interval_theta(theta_hat, n_shots=10000, confidence=0.95)
    assert (hi95_more - lo95_more) < (hi95 - lo95)


def test_confidence_interval_theta_clips_to_domain():
    lo, hi = confidence_interval_theta(0.01, n_shots=10, confidence=0.99)
    assert lo >= 0.0
    lo2, hi2 = confidence_interval_theta(PI - 0.01, n_shots=10, confidence=0.99)
    assert hi2 <= PI


def test_confidence_interval_rejects_unsupported_confidence():
    with pytest.raises(ValueError):
        confidence_interval_theta(1.0, n_shots=100, confidence=0.5)


# --------------------------------------------------------------------- #
# empirical validation against the Qiskit simulator (skipped if unavailable)
# --------------------------------------------------------------------- #
def test_empirical_theta_std_matches_analytical_away_from_pole():
    qiskit = pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from quang.statistics import empirical_theta_std

    result = empirical_theta_std(theta=PI / 2, n_shots=2000, n_trials=300, seed=42)
    # empirical std should land within ~30% of the analytical 1/sqrt(N) prediction
    assert result.empirical_std == pytest.approx(result.analytical_std, rel=0.3)
    assert result.empirical_mean == pytest.approx(PI / 2, abs=0.05)


def test_empirical_theta_std_breaks_down_very_close_to_a_pole():
    """With few shots and theta very close to a pole, the minority outcome
    becomes a rare event: many trials land EXACTLY on the pole (zero
    error), which is qualitatively different from -- and not predicted by
    -- the Gaussian delta-method formula."""
    qiskit = pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from quang.statistics import empirical_theta_std

    result = empirical_theta_std(theta=0.05, n_shots=50, n_trials=300, seed=7)
    assert result.n_at_pole_boundary > 0
