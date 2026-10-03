"""
qang.statistics — error propagation between probability-space (finite-shot
measurement) and theta-space / qg-space.

This addresses the still-open half of Future Research Direction #2: "prove
the stated bounds, the domain over which qg_Z and qg_S are each invertible
(Section 2.2) [done in qang.core], and error-propagation bounds when
converting between theta-space and qg-space [done HERE]; complement this
with empirical benchmarking of qang-parameterized optimizers on NISQ
hardware and simulators [the empirical half is done here via Qiskit's
AerSimulator]."

The question: you never observe P(|0>) directly. You estimate it from N
measurement shots as p0_hat = counts0 / N, a Binomial(N, p0) / N estimator
with Var(p0_hat) = p0*(1-p0) / N. Since qg_Z_hat = 2*p0_hat - 1, that shot
noise propagates directly into qg_Z_hat, and then -- through the *same*
singular Jacobian studied in qang.gradients for optimization steps --
into theta_hat = arccos(qg_Z_hat).

The closed-form result (delta method, first order)
----------------------------------------------------
    Var(qg_Z_hat)   = 4 * p0*(1-p0) / N = sin^2(theta) / N
                       (since p0*(1-p0) = cos^2(theta/2)*sin^2(theta/2)
                       = sin^2(theta)/4)

    Var(theta_hat)  ~= Var(qg_Z_hat) * (d(theta)/d(qg))^2
                     = [sin^2(theta)/N] * [1/sin^2(theta)]
                     = 1/N

The sin^2(theta) factors cancel EXACTLY: to first order, the propagated
angular uncertainty is CONSTANT (std ~= 1/sqrt(N) rad) across the whole
Bloch sphere, independent of theta -- the shrinking shot-noise variance
near a pole (fewer "wrong-outcome" events to measure) exactly offsets the
growing sensitivity of arccos there (the Section 4.1 singularity). This is
a genuinely different statement from the *optimization*-step story in
qang.gradients, where the (non-vanishing) energy gradient dE/d(theta) does
NOT shrink near the poles the way sqrt(Var(qg_Z_hat)) does here -- so a
qg-space *gradient step* still blows up near a pole even though qg-space
*measurement* uncertainty does not.

The cancellation is only a first-order (delta-method / Gaussian) result: it
requires N large enough that the binomial count of the minority outcome is
itself well approximated by a Gaussian (rule of thumb: N * min(p0, p1) >> 1).
Very close to a pole, for fixed N, that condition fails -- the minority
outcome becomes a rare event, theta_hat is usually exactly the pole itself
(zero error) with an occasional large jump when the rare outcome appears --
and qang.statistics.empirical_theta_std lets you see that breakdown
directly by simulating it, rather than assuming the asymptotic formula
holds everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .core import Qang
from .gradients import inverse_jacobian_raw


# --------------------------------------------------------------------- #
# closed-form (delta-method) propagation
# --------------------------------------------------------------------- #
def p0_from_theta(theta: float) -> float:
    return math.cos(theta / 2.0) ** 2


def var_qg_z_shot_noise(theta: float, n_shots: int) -> float:
    """
    Var(qg_Z_hat) for N i.i.d. Z-basis measurement shots on the state at
    polar angle theta. Exact (not an approximation): Var(qg_Z_hat) =
    4*p0*(1-p0)/N = sin^2(theta)/N.
    """
    if n_shots <= 0:
        raise ValueError(f"n_shots must be positive, got {n_shots}.")
    p0 = p0_from_theta(theta)
    return 4.0 * p0 * (1.0 - p0) / n_shots


def std_qg_z_shot_noise(theta: float, n_shots: int) -> float:
    return math.sqrt(var_qg_z_shot_noise(theta, n_shots))


def propagated_theta_variance(theta: float, n_shots: int) -> float:
    """
    First-order (delta-method) propagated variance of theta_hat =
    arccos(qg_Z_hat), reusing the exact inverse Jacobian from
    qang.gradients (the same -1/sin(theta) that diverges at the poles in
    the *optimization* story). Away from the poles this evaluates to
    (very nearly) 1/n_shots for ANY theta -- see the module docstring for
    why the sin^2(theta) factors cancel.
    """
    var_qg = var_qg_z_shot_noise(theta, n_shots)
    dtheta_dqg = inverse_jacobian_raw(theta)
    if not math.isfinite(dtheta_dqg):
        return float("inf")  # exactly at a pole: linear approximation breaks down (0 * inf)
    return var_qg * dtheta_dqg ** 2


def propagated_theta_std(theta: float, n_shots: int) -> float:
    var = propagated_theta_variance(theta, n_shots)
    return math.sqrt(var) if math.isfinite(var) else float("inf")


def confidence_interval_theta(
    theta_hat: float, n_shots: int, confidence: float = 0.95
) -> tuple:
    """
    A normal-approximation confidence interval for the true theta, centered
    on an observed theta_hat, using the (theta-independent, away from the
    poles) 1/n_shots propagated variance: CI = theta_hat +/- z * sqrt(1/N).
    z=1.96 for confidence=0.95.
    """
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(round(confidence, 2))
    if z is None:
        raise ValueError(f"confidence must be one of {sorted(z_scores)}, got {confidence}.")
    half_width = z / math.sqrt(n_shots)
    lo = max(0.0, theta_hat - half_width)
    hi = min(math.pi, theta_hat + half_width)
    return lo, hi


# --------------------------------------------------------------------- #
# empirical validation via simulated measurement (Qiskit AerSimulator)
# --------------------------------------------------------------------- #
@dataclass
class BootstrapResult:
    theta: float
    n_shots: int
    n_trials: int
    theta_hat_samples: List[float]
    empirical_mean: float
    empirical_std: float
    analytical_std: float
    n_at_pole_boundary: int  # trials where theta_hat landed exactly on a domain boundary (0 or pi)


def empirical_theta_std(
    theta: float,
    n_shots: int,
    n_trials: int = 500,
    seed: Optional[int] = None,
) -> BootstrapResult:
    """
    Simulate n_trials independent experiments, each measuring the qg_Z
    state at polar angle `theta` with `n_shots` shots on Qiskit's
    AerSimulator, reconstructing theta_hat = arccos(qg_Z_hat) each time,
    and returning the empirical mean/std of theta_hat across trials --
    for direct comparison against propagated_theta_std(theta, n_shots).

    Requires Qiskit + qiskit-aer (raises ImportError with an install hint
    if unavailable, same pattern as qang.qiskit_gate).
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qang.statistics.empirical_theta_std requires Qiskit + qiskit-aer. "
            "Install with `pip install qiskit qiskit-aer`."
        ) from exc

    from .qiskit_gate import RQangGate  # local import: keeps qang.statistics importable without qiskit

    qg = Qang.from_angles(theta, mode="polar")
    qc = QuantumCircuit(1, 1)
    qc.append(RQangGate(qg), [0])
    qc.measure(0, 0)

    sim = AerSimulator(seed_simulator=seed)
    qc_t = transpile(qc, sim)

    theta_hats: List[float] = []
    at_boundary = 0
    for trial in range(n_trials):
        trial_seed = None if seed is None else seed + trial
        job = sim.run(qc_t, shots=n_shots, seed_simulator=trial_seed)
        counts = job.result().get_counts()
        p0_hat = counts.get("0", 0) / n_shots
        p1_hat = counts.get("1", 0) / n_shots
        qg_hat = max(-1.0, min(1.0, p0_hat - p1_hat))
        theta_hat = math.acos(qg_hat)
        theta_hats.append(theta_hat)
        if qg_hat <= -1.0 + 1e-12 or qg_hat >= 1.0 - 1e-12:
            at_boundary += 1

    mean_theta = sum(theta_hats) / len(theta_hats)
    var_theta = sum((t - mean_theta) ** 2 for t in theta_hats) / (len(theta_hats) - 1)

    return BootstrapResult(
        theta=theta,
        n_shots=n_shots,
        n_trials=n_trials,
        theta_hat_samples=theta_hats,
        empirical_mean=mean_theta,
        empirical_std=math.sqrt(var_theta),
        analytical_std=propagated_theta_std(theta, n_shots),
        n_at_pole_boundary=at_boundary,
    )


# --------------------------------------------------------------------- #
# Few-shot estimation: Haar prior = uniform prior on qg_Z
# --------------------------------------------------------------------- #
# Under the Haar measure on single-qubit pure states, qg_Z = cos(theta) is
# UNIFORM on [-1, 1] (Archimedes' hat-box theorem: equal-height bands of a
# sphere have equal area). A uniform prior on qg_Z is therefore the
# rotation-invariant prior, and because qg_Z = 2*p0 - 1 is affine it is the
# same as p0 ~ Beta(1, 1). With k0 zeros in N shots the posterior is
# Beta(k0 + 1, N - k0 + 1), and every qg_Z quantity follows by the affine
# map. This fixes exactly the regime where the delta-method interval above
# fails: near a pole, k0 = N is common and the delta interval collapses to
# a single point. tests/test_statistics.py compares coverage against the
# delta method and the standard Wilson interval: near a pole Bayes-Haar and
# Wilson both keep ~92% coverage at N = 50 while the delta method drops
# below 10%; on Haar-random states the posterior mean has ~4% lower mean
# squared error than the raw frequency. Implemented with the standard
# library only (no SciPy), so the qang core stays NumPy-free here.
def _betacf(a: float, b: float, x: float) -> float:
    # Continued fraction for the incomplete beta function (modified Lentz).
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) (the Beta(a, b) CDF)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(u: float, a: float, b: float) -> float:
    """Inverse of beta_cdf by bisection (60 iterations: ~1e-18 in x)."""
    if not 0.0 <= u <= 1.0:
        raise ValueError("u must lie in [0, 1].")
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if beta_cdf(mid, a, b) < u:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class QgEstimate:
    """A qg_Z point estimate with a two-sided interval."""

    qg_z: float
    low: float
    high: float
    method: str


def bayes_qg_estimate(k0: int, n_shots: int, confidence: float = 0.95) -> QgEstimate:
    """Posterior mean and equal-tailed credible interval for qg_Z under the
    Haar (uniform-in-qg_Z) prior, from k0 outcomes '0' in n_shots shots."""
    if n_shots < 1 or not 0 <= k0 <= n_shots:
        raise ValueError("need n_shots >= 1 and 0 <= k0 <= n_shots.")
    a, b = k0 + 1.0, n_shots - k0 + 1.0
    alpha = 1.0 - confidence
    p_lo, p_hi = beta_ppf(alpha / 2.0, a, b), beta_ppf(1.0 - alpha / 2.0, a, b)
    return QgEstimate(
        qg_z=2.0 * a / (a + b) - 1.0,
        low=2.0 * p_lo - 1.0,
        high=2.0 * p_hi - 1.0,
        method="bayes_haar",
    )


def wilson_qg_estimate(k0: int, n_shots: int, z: float = 1.96) -> QgEstimate:
    """Wilson score interval for p0, mapped to qg_Z; the standard
    frequentist fix for the delta method's collapse near the poles."""
    if n_shots < 1 or not 0 <= k0 <= n_shots:
        raise ValueError("need n_shots >= 1 and 0 <= k0 <= n_shots.")
    n = float(n_shots)
    p = k0 / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return QgEstimate(
        qg_z=2.0 * p - 1.0,
        low=2.0 * (centre - half) - 1.0,
        high=2.0 * (centre + half) - 1.0,
        method="wilson",
    )


def delta_qg_estimate(k0: int, n_shots: int, z: float = 1.96) -> QgEstimate:
    """Delta-method (Wald) interval for qg_Z: 2*p0_hat - 1 +- z*sqrt(Var).
    Collapses to a single point when k0 = 0 or k0 = n_shots."""
    if n_shots < 1 or not 0 <= k0 <= n_shots:
        raise ValueError("need n_shots >= 1 and 0 <= k0 <= n_shots.")
    p = k0 / n_shots
    q = 2.0 * p - 1.0
    half = 2.0 * z * math.sqrt(p * (1.0 - p) / n_shots)
    return QgEstimate(qg_z=q, low=q - half, high=q + half, method="delta")


def qg_estimate(k0: int, n_shots: int, method: str = "bayes", confidence: float = 0.95) -> QgEstimate:
    """One entry point for the qg_Z interval estimators (RFC v2, Section 3.4).

    method: "bayes" (posterior under the Haar prior, equal-tailed credible
    interval), "wilson" (score interval) or "delta" (Wald). ``confidence`` sets
    the two-sided level for all three (z = the normal quantile for Wilson and
    delta)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1).")
    m = method.lower()
    if m in ("bayes", "bayes_haar"):
        return bayes_qg_estimate(k0, n_shots, confidence)
    from statistics import NormalDist

    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    if m == "wilson":
        return wilson_qg_estimate(k0, n_shots, z)
    if m in ("delta", "wald"):
        return delta_qg_estimate(k0, n_shots, z)
    raise ValueError(f"unknown method {method!r}; use 'bayes', 'wilson' or 'delta'.")


# --------------------------------------------------------------------- #
# mixed states and multi-qubit registers (RESEARCH_NOTES §45)
# --------------------------------------------------------------------- #
def propagated_theta_variance_mixed(theta: float, r: float, n_shots: int) -> float:
    """Delta-method variance of theta_hat = arccos(qg_Z_hat / r) for a qubit
    whose Bloch vector has known length r (0 < r <= 1) at polar angle theta:

        Var(theta_hat) = (1 - r^2 cos^2 theta) / (N r^2 sin^2 theta)
                       = [1 + (1 - r^2) / (r^2 sin^2 theta)] / N.

    For r = 1 this is the theta-independent 1/N of §5; for any r < 1 the
    shot noise no longer vanishes at the poles (it tends to (1 - r^2)/N)
    while the Jacobian still diverges, so the error blows up there. At the
    equator it equals the quantum Cramer-Rao bound 1/(N r^2)."""
    if not 0.0 < r <= 1.0:
        raise ValueError(f"r must be in (0, 1], got {r}.")
    if n_shots <= 0:
        raise ValueError(f"n_shots must be positive, got {n_shots}.")
    s2 = math.sin(theta) ** 2
    if s2 == 0.0:
        return 0.0 if r == 1.0 else float("inf")
    return (1.0 - (r * math.cos(theta)) ** 2) / (n_shots * r * r * s2)


def theta_qcrb_variance(r: float, n_shots: int) -> float:
    """Quantum Cramer-Rao bound 1/(N r^2) for a polar rotation of a qubit
    with Bloch length r."""
    return 1.0 / (n_shots * r * r)


def _z_values(n_qubits: int):
    import numpy as np

    idx = np.arange(2 ** n_qubits)
    return np.array([1 - 2 * ((idx >> i) & 1) for i in range(n_qubits)], dtype=float)  # (n, 2^n)


def qg_covariance(probs, n_qubits: int, n_shots: int):
    """Per-qubit qg_Z (vector) and the exact covariance matrix of their
    estimators from the SAME n_shots joint Z-basis shots:

        Cov(qg_i_hat, qg_j_hat) = (<Z_i Z_j> - qg_i qg_j) / N.

    probs: the 2^n outcome distribution, bit i of the index = qubit i
    (Qiskit ordering)."""
    import numpy as np

    p = np.asarray(probs, dtype=float)
    z = _z_values(n_qubits)
    q = z @ p
    zz = (z * p) @ z.T
    return q, (zz - np.outer(q, q)) / n_shots


def delta_method_variance(gradient, covariance) -> float:
    """Var(f) ~= grad^T Cov grad for any smooth function of the qg vector."""
    import numpy as np

    g = np.asarray(gradient, dtype=float)
    return float(g @ np.asarray(covariance, dtype=float) @ g)


def register_witness_variance(probs, n_qubits: int, n_shots: int, independent: bool = False) -> float:
    """Variance of the register witness (1/n) sum_i qg_i_hat. independent=True
    gives the naive value that ignores the correlations between qubits
    (the diagonal of the covariance only)."""
    import numpy as np

    _, cov = qg_covariance(probs, n_qubits, n_shots)
    if independent:
        cov = np.diag(np.diag(cov))
    return max(0.0, delta_method_variance(np.full(n_qubits, 1.0 / n_qubits), cov))  # clip rounding below 0


# --------------------------------------------------------------------- #
# qg_S from finite shots: point estimates and intervals (RESEARCH_NOTES §51)
# --------------------------------------------------------------------- #
def _h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def qg_s_from_qg_z_interval(low: float, high: float) -> tuple:
    """Map an interval for qg_Z through the exact identity qg_S = H((1+qg_Z)/2)
    (§7). H is unimodal with its maximum 1 at qg_Z = 0, so the image of
    [low, high] is [min(H(low), H(high)), 1] when the interval contains 0 and
    the ordered endpoint values otherwise. Coverage of the qg_S interval is at
    least that of the qg_Z interval."""
    low, high = max(-1.0, low), min(1.0, high)
    h_lo, h_hi = _h2((1.0 + low) / 2.0), _h2((1.0 + high) / 2.0)
    if low <= 0.0 <= high:
        return min(h_lo, h_hi), 1.0
    return min(h_lo, h_hi), max(h_lo, h_hi)


def qg_s_estimate(k0: int, n_shots: int, method: str = "qg_wilson", z: float = 1.96) -> tuple:
    """(point estimate, low, high) for a single qubit's qg_S from k0 zeros in
    n_shots. method:
      "qg_wilson"  Miller-Madow point estimate; interval = Wilson interval for
                   qg_Z mapped through qg_S = H((1+qg_Z)/2)  (recommended)
      "wald"       plug-in +- z * delta-method standard error (collapses at
                   qg_Z = 0, where dH/dp = 0, and at the poles)
    """
    if n_shots < 1 or not 0 <= k0 <= n_shots:
        raise ValueError("need n_shots >= 1 and 0 <= k0 <= n_shots.")
    p = k0 / n_shots
    plug = _h2(p)
    if method == "wald":
        se = 0.0 if p in (0.0, 1.0) else math.sqrt(p * (1 - p) / n_shots) * abs(math.log2((1 - p) / p))
        return plug, max(0.0, plug - z * se), min(1.0, plug + z * se)
    if method != "qg_wilson":
        raise ValueError("method must be 'qg_wilson' or 'wald'.")
    mm = plug + (1.0 / (2.0 * n_shots * math.log(2)) if 0 < k0 < n_shots else 0.0)
    w = wilson_qg_estimate(k0, n_shots, z)
    lo, hi = qg_s_from_qg_z_interval(w.low, w.high)
    return min(mm, 1.0), lo, hi


# --------------------------------------------------------------------- #
# the squared radius from counts: an unbiased estimator
# --------------------------------------------------------------------- #
def qg2_unbiased(k0: int, n_shots: int) -> float:
    """Unbiased estimate of qg^2 from k0 outcomes '0' in n_shots measurements
    of one axis. The plug-in qg_hat^2 is biased upwards by (1 - qg^2)/N;
    (N qg_hat^2 - 1)/(N - 1) has expectation exactly qg^2. Needs N >= 2; the
    estimate can fall below 0 when qg is near 0."""
    if n_shots < 2 or not 0 <= k0 <= n_shots:
        raise ValueError("need n_shots >= 2 and 0 <= k0 <= n_shots.")
    q = 2.0 * k0 / n_shots - 1.0
    return (n_shots * q * q - 1.0) / (n_shots - 1.0)


def radius2_estimate(counts, unbiased: bool = True) -> tuple:
    """Squared local radius r^2 = qg_X^2 + qg_Y^2 + qg_Z^2 from counts of the
    measured axes, ``counts`` = [(k0, n_shots), ...] (one pair per axis; pass
    only qg_Z in a definite-weight state, where qg_X = qg_Y = 0). Returns
    (estimate, standard error), the error from the delta method,
    Var(qg_hat^2) ~ 4 qg^2 (1 - qg^2) / N, plus the second-order term
    2 (1 - qg^2)^2 / N^2 that dominates near qg = 0."""
    est, var = 0.0, 0.0
    for k0, n in counts:
        q = 2.0 * k0 / n - 1.0
        est += qg2_unbiased(k0, n) if unbiased else q * q
        var += 4.0 * q * q * (1.0 - q * q) / n + 2.0 * (1.0 - q * q) ** 2 / n**2
    return est, math.sqrt(var)
