"""Shot budget vs angle in qga (P1).

Compares the finite-shot budget required for a target precision in two
coordinates of the same one-qubit readout:

- qg_Z (probability-space linear coordinate): Var(qg_Z_hat) = (1-qg_Z^2)/N,
  so N_qg(target) = (1-qg_Z^2) / target_qg^2 depends on the operating point.
- qga angle (Bloch polar angle in thousandths of a turn): Fisher information
  per Z shot is constant, F = 4*pi^2/1e6 qga^-2, so the delta-method shot
  budget N_qga(target) = 1/(F * target_qga^2) = (159.1549.../target_qga)^2
  is state-independent away from the poles.

The constant-qga budget is only a first-order result. Near the poles, with
fixed N, the minority outcome is rare and the delta method breaks down; the
empirical angle error can deviate from 159.15/sqrt(N) because boundary
clipping and rare minority outcomes dominate. This script reports both the
closed-form budgets and the finite-shot Monte Carlo check.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qang.qga import QGA_PER_TURN, FISHER_PER_QGA2, angle_sigma_qga, p0_from_qga
from qang.statistics import var_qg_z_shot_noise


def shot_budget_qg(target_qg: float, theta_qga: float) -> float:
    """Shots for target standard error in qg_Z at a given qga angle."""
    if target_qg <= 0:
        raise ValueError("target_qg must be positive.")
    p0 = p0_from_qga(theta_qga)
    return 4.0 * p0 * (1.0 - p0) / (target_qg * target_qg)


def shot_budget_qga(target_qga: float) -> float:
    """Delta-method shots for target standard error in qga (state-independent)."""
    if target_qga <= 0:
        raise ValueError("target_qga must be positive.")
    return 1.0 / (FISHER_PER_QGA2 * target_qga * target_qga)


def empirical_angle_std_qga(theta_qga: float, n_shots: int, n_trials: int, seed: int = 0) -> float:
    """Monte Carlo std of the qga angle estimate from Z-basis shots."""
    rng = np.random.default_rng(seed)
    p0 = p0_from_qga(theta_qga)
    k0 = rng.binomial(n_shots, p0, size=n_trials)
    p0_hat = k0 / n_shots
    qg_hat = np.clip(2.0 * p0_hat - 1.0, -1.0, 1.0)
    theta_hat_qga = QGA_PER_TURN / math.pi * np.arccos(np.sqrt(np.clip((qg_hat + 1.0) / 2.0, 0.0, 1.0)))
    return float(np.std(theta_hat_qga, ddof=1))


def run_demo() -> dict:
    target_qga = 5.0
    target_qg = 0.02
    angles = [10.0, 50.0, 125.0, 250.0, 400.0, 490.0]  # qga
    n_qga = math.ceil(shot_budget_qga(target_qga))
    rows = []
    for a in angles:
        n_qg = math.ceil(shot_budget_qg(target_qg, a))
        emp = empirical_angle_std_qga(a, n_qga, n_trials=4000, seed=7)
        rows.append(
            {
                "theta_qga": a,
                "shot_budget_qg": n_qg,
                "shot_budget_qga": n_qga,
                "empirical_angle_std_qga": emp,
                "delta_method_angle_std_qga": angle_sigma_qga(n_qga),
            }
        )
    return {
        "target_qga": target_qga,
        "target_qg": target_qg,
        "rows": rows,
    }


if __name__ == "__main__":
    demo = run_demo()
    print(f"target_qga={demo['target_qga']} -> N_qga={demo['rows'][0]['shot_budget_qga']}")
    print(f"target_qg={demo['target_qg']}")
    for row in demo["rows"]:
        print(
            f"theta={row['theta_qga']:6.1f} qga  "
            f"N_qg={row['shot_budget_qg']:5d}  "
            f"N_qga={row['shot_budget_qga']:5d}  "
            f"emp_qga={row['empirical_angle_std_qga']:6.3f}  "
            f"delta_qga={row['delta_method_angle_std_qga']:6.3f}"
        )
