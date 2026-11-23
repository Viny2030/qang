"""tools/verify_exp41_1.py — Exp. 41.1: amplificación P-CTC y balance de Fisher por copia de entrada."""
from __future__ import annotations

import math

import numpy as np

from qang.qtimeloop import pctc_amplification, pctc_fisher, pctc_p_post, pctc_theta_out

A = 0.5
N = 20_000
SEEDS = (21, 22, 23)
THETAS = (math.pi - 0.01, math.pi / 2.0, 0.1)


def main() -> None:
    print("Exp. 41.1 — P-CTC: amplificación por disparo aceptado, ×1/p por copia y Fisher total = 1")

    worst_fisher = 0.0
    rng_grid = np.random.default_rng(41)
    for _ in range(2000):
        a = float(rng_grid.uniform(0.05, 0.95))
        theta = float(rng_grid.uniform(1e-3, math.pi - 1e-3))
        worst_fisher = max(worst_fisher, abs(pctc_fisher(a, theta)[3] - 1.0))
    print(f"  Fisher exacto en 2000 pares (a,θ): max|F_total−1| = {worst_fisher:.2e}")
    assert worst_fisher < 1e-10

    for theta in THETAS:
        p = pctc_p_post(A, theta)
        amp = pctc_amplification(A, theta)
        f_acc, p_check, f_bern, total = pctc_fisher(A, theta)
        theta_out = pctc_theta_out(A, theta)
        q_out = math.cos(theta_out)
        dp = -(1.0 - A * A) * math.sin(theta) / 2.0
        assert abs(p_check - p) < 1e-12
        assert abs(total - 1.0) < 1e-12

        print(f"\n  a={A} θ={theta:.4f} rad: p_post={p:.6f}  amplificación={amp:.6f}")
        print(f"    F_acc={f_acc:.6f}  F_fracaso={f_bern:.6f}  total={total:.12f}")
        print(f"    θ'={theta_out:.6f}  qg_Z'={q_out:.9f}  coste/copia=1/p={1.0 / p:.4f}")

        totals = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            k = int(rng.binomial(N, p))
            p_hat = k / N
            ones = int(rng.binomial(k, (1.0 - q_out) / 2.0))
            q_hat = 1.0 - 2.0 * ones / k
            sigma_p = math.sqrt(p * (1.0 - p) / N)
            sigma_q = math.sqrt(max(0.0, 1.0 - q_out * q_out) / k)
            assert abs(p_hat - p) < 5.0 * sigma_p
            assert abs(q_hat - q_out) < 5.0 * sigma_q

            total_hat = p_hat * f_acc + dp * dp / (p_hat * (1.0 - p_hat))
            totals.append(total_hat)
            if seed == SEEDS[0]:
                print(f"    semilla {seed}: k={k:5d}/{N}  p̂={p_hat:.6f}±{sigma_p:.6f}  "
                      f"q̂'={q_hat:.6f}±{sigma_q:.6f}  overhead={N / k:.4f}  F_total̂={total_hat:.6f}")

        mean_total = float(np.mean(totals))
        spread = float(np.std(totals))
        assert abs(mean_total - 1.0) < 5.0 * max(spread, 1e-12)
        print(f"    Fisher muestreado ({len(SEEDS)} semillas): media={mean_total:.6f}  σ={spread:.6f}")

    theta_pole = math.pi
    assert abs(pctc_amplification(A, theta_pole) - 1.0 / A) < 1e-12
    assert abs(pctc_p_post(A, theta_pole) - A * A) < 1e-12
    print(f"\n  polo sur: amplificación=1/a={1.0 / A:.1f}, p_post=a²={A * A:.2f} → ×{1.0 / (A * A):.0f} disparos")
    print("verify_exp41_1: OK")


if __name__ == "__main__":
    main()
