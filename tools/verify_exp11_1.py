"""tools/verify_exp11_1.py — Exp. 11.1: punto de operación Ramsey en qg_Z=0."""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from qang.probability import shot_noise_milliqang, shots_for_resolution
from qang.qmetro import RAD_PER_QGA, optimal_T, ramsey_fisher_phi, ramsey_fisher_omega

N = 20_000


def main() -> None:
    print("Exp. 11.1 — Ramsey: I_F máxima en φ=π/2 ⇔ qg_Z=0 (250 qga)")

    phis = jnp.linspace(0.01, math.pi - 0.01, 2001)
    for C, max_i, loss in ((1.0, 1.0, 0.0), (0.8, 0.64, 0.90), (0.5, 0.25, 0.95)):
        values = ramsey_fisher_phi(phis, jnp.asarray(C))
        k = int(jnp.argmax(values))
        peak = float(values[k])
        pole = float(ramsey_fisher_phi(jnp.asarray(0.2), jnp.asarray(C)))
        assert abs(peak - max_i) < 1e-6
        assert C == 1.0 or abs(float(phis[k]) - math.pi / 2.0) < 2e-3
        assert abs(100.0 * (1.0 - pole / max_i) - 100.0 * loss) < 1.0
        print(f"  C={C:.1f}: max I_F={peak:.6f} en φ={float(phis[k]):.4f} rad; "
              f"a φ=0.2 pierde {100.0 * (1.0 - pole / max_i):.1f} %")

    flat = ramsey_fisher_phi(phis, jnp.asarray(1.0))
    assert float(jnp.std(flat)) < 1e-12
    t_star, _ = optimal_T(lambda n, T, T2_: ramsey_fisher_omega(T, T2_) * n, 1.0, 1.0)
    assert abs(t_star - 0.5) < 1e-6
    print(f"  C=1: I_F≡1 (σ={float(jnp.std(flat)):.1e}); T*=T2/2={t_star:.6f}")

    sigma = shot_noise_milliqang(math.pi / 2.0, N)
    assert abs(sigma - 1000.0 / math.sqrt(N)) < 1e-9
    assert abs(1000.0 / (2.0 * math.pi) - 159.15) < 0.01
    assert abs(RAD_PER_QGA - 2.0 * math.pi / 1000.0) < 1e-15
    print(f"  N={N}: σ_m={sigma:.4f} m-qg; σ_qga≈{159.15 / math.sqrt(N):.4f}; "
          f"resolución 1 m-qg → {shots_for_resolution(1.0, math.pi / 2.0)} shots")
    print("verify_exp11_1: OK")


if __name__ == "__main__":
    main()
