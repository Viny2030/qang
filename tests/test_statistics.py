import math

import pytest

from qang.statistics import (
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
    from qang.statistics import empirical_theta_std

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
    from qang.statistics import empirical_theta_std

    result = empirical_theta_std(theta=0.05, n_shots=50, n_trials=300, seed=7)
    assert result.n_at_pole_boundary > 0


# --------------------------------------------------------------------- #
# Few-shot estimation: Haar prior = uniform on qg_Z
# --------------------------------------------------------------------- #
def test_haar_random_states_have_uniform_qg_z():
    import numpy as np

    rng = np.random.default_rng(0)
    v = rng.normal(size=(200_000, 2)) + 1j * rng.normal(size=(200_000, 2))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    qz = np.abs(v[:, 0]) ** 2 - np.abs(v[:, 1]) ** 2
    hist, _ = np.histogram(qz, bins=10, range=(-1, 1))
    assert np.all(np.abs(hist / hist.sum() - 0.1) < 0.005)


@pytest.mark.parametrize("a,b", [(1, 1), (50, 2), (3, 48), (26, 26), (51, 1)])
def test_beta_ppf_matches_scipy(a, b):
    scipy_stats = pytest.importorskip("scipy.stats")
    from qang.statistics import beta_cdf, beta_ppf

    for u in (0.025, 0.5, 0.975):
        assert beta_ppf(u, a, b) == pytest.approx(scipy_stats.beta.ppf(u, a, b), abs=1e-9)
    for x in (0.1, 0.5, 0.9, 0.99):
        assert beta_cdf(x, a, b) == pytest.approx(scipy_stats.beta.cdf(x, a, b), abs=1e-10)


def test_bayes_interval_does_not_collapse_when_all_shots_agree():
    from qang.statistics import bayes_qg_estimate, delta_qg_estimate

    d = delta_qg_estimate(50, 50)
    b = bayes_qg_estimate(50, 50)
    assert d.low == d.high == 1.0
    assert 0.99 < b.high <= 1.0
    assert b.low < 0.9
    assert b.low < b.qg_z < b.high
    assert b.qg_z == pytest.approx(2 * 51 / 52 - 1)


def _coverage(p_true, estimator, n_shots=50, trials=4000, seed=1):
    import numpy as np

    rng = np.random.default_rng(seed)
    k = rng.binomial(n_shots, p_true)
    q = 2 * p_true - 1
    hits = 0
    for ki, qi in zip(k, q):
        e = estimator(int(ki), n_shots)
        hits += e.low <= qi <= e.high
    return hits / len(k)


def test_near_a_pole_bayes_and_wilson_keep_coverage_while_delta_collapses():
    import numpy as np
    from qang.statistics import bayes_qg_estimate, delta_qg_estimate, wilson_qg_estimate

    p = np.full(4000, math.cos(0.04) ** 2)  # theta = 0.08 rad
    assert _coverage(p, delta_qg_estimate) < 0.10
    assert _coverage(p, wilson_qg_estimate) > 0.90
    assert _coverage(p, bayes_qg_estimate) > 0.90


def test_on_haar_states_bayes_mean_beats_raw_frequency():
    import numpy as np
    from qang.statistics import bayes_qg_estimate

    rng = np.random.default_rng(2)
    n = 50
    q_true = rng.uniform(-1, 1, 20000)
    k = rng.binomial(n, (1 + q_true) / 2)
    mse_raw = np.mean((2 * k / n - 1 - q_true) ** 2)
    post_mean = np.array([bayes_qg_estimate(kk, n).qg_z for kk in range(n + 1)])
    mse_bayes = np.mean((post_mean[k] - q_true) ** 2)
    assert mse_bayes < mse_raw
    assert 0.02 < 1 - mse_bayes / mse_raw < 0.06
