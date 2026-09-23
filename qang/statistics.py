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
