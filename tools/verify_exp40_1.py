"""tools/verify_exp40_1.py — Exp. 40.1: presupuesto de shots para Y_B y control de regeneración."""
from __future__ import annotations

import math

import numpy as np

from qang.probability import analyze, shot_noise_milliqang, shots_for_resolution, shots_for_sign

Y_B = 8.7e-11
Z = 3.0
Q_DEMO = 0.05
DEMO_SEEDS = (11, 12, 13, 14, 15)
DELTA_P_REGEN = 4.85e-4
SHOTS_REGEN = 2**24


def main() -> None:
    m_b = 1000.0 * Y_B
    n_1s = shots_for_resolution(m_b, math.pi / 2.0)
    n_3s = shots_for_sign(Y_B, z=Z)
    assert math.isclose(n_3s, 9.0 * n_1s, rel_tol=1e-9)
    assert math.isclose(n_3s, (Z / Y_B) ** 2, rel_tol=1e-9)
    print("Exp. 40.1 — presupuesto de shots para la asimetría bariónica")
    print(f"  Y_B={Y_B:.2e} -> qg_Z={Y_B:.2e}, m={m_b:.2e} m-qg")
    print(f"  N(1σ)={n_1s:.4e}; N(3σ)={n_3s:.4e} shots  [=(3/Y_B)^2]")

    sigma_demo = shot_noise_milliqang(math.acos(Q_DEMO), shots_for_sign(Q_DEMO, z=Z))
    assert math.isclose(sigma_demo, 1000.0 * Q_DEMO / Z, rel_tol=1e-12)
    n_demo = 4 * shots_for_sign(Q_DEMO, z=Z)
    m_demo = 1000.0 * Q_DEMO
    sigma_run = shot_noise_milliqang(math.acos(Q_DEMO), n_demo)
    p0 = (1.0 + Q_DEMO) / 2.0
    for seed in DEMO_SEEDS:
        k0 = int(np.random.default_rng(seed).binomial(n_demo, p0))
        reading = analyze(k0, n_demo, target_qg_z=Q_DEMO, z=Z)
        dev = float(reading.deviation_milliqang)
        assert abs(dev) < 3.0 * sigma_run
        assert reading.milliqang[1] > 0.0
        print(f"  demo q={Q_DEMO:.2f}, semilla {seed}: m={reading.milliqang[0]:+7.2f} m-qg "
              f"(dev={dev:+6.2f}, 3σ={3*sigma_run:.2f}); LI3σ={reading.milliqang[1]:+6.2f} > 0")

    dm_regen = -2000.0 * DELTA_P_REGEN
    sigma_regen = 1000.0 / math.sqrt(SHOTS_REGEN)
    z_regen = abs(dm_regen) / sigma_regen
    assert z_regen > Z
    assert abs(dm_regen) > abs(m_b)
    print(f"  control ΔP={DELTA_P_REGEN:.2e} -> Δm={dm_regen:+.2f} m-qg; "
          f"con {SHOTS_REGEN} shots σ={sigma_regen:.4f} m-qg -> {z_regen:.0f}σ "
          f"({abs(dm_regen)/abs(m_b):.1e}× la señal directa)")
    print("verify_exp40_1: OK")


if __name__ == "__main__":
    main()
