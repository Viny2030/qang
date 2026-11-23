"""tools/verify_exp18_1.py — Exp. 18.1: ley P·F=1/4 del decodificador de Yoshida–Kitaev con lectura por disparos."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from qang.qscramble import haar_unitary, yoshida_kitaev

N_TRIALS = 20_000
SEEDS = (11, 12, 13)
N_B, N_D = 3, 2
D_AB = 2 ** (1 + N_B)


def _product_se(p: float, f: float, n: int) -> float:
    return math.sqrt((f * f * p * (1.0 - p) + p * f * (1.0 - f)) / n)


def main() -> None:
    x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    cases = (
        ("Haar #1", haar_unitary(jax.random.PRNGKey(1), D_AB)),
        ("X en A", jnp.kron(x, jnp.eye(2**N_B, dtype=jnp.complex128))),
        ("I", jnp.eye(D_AB, dtype=jnp.complex128)),
    )
    print("Exp. 18.1 — Hayden–Preskill, P·F=1/4 con conteo de disparos")
    for name, u in cases:
        p, f, f_qg = yoshida_kitaev(u, N_B, N_D)
        assert abs(p * f - 0.25) < 1e-12
        assert abs(f - f_qg) < 1e-12
        assert math.isclose((2000.0 * p) * (2000.0 * f), 1e6, rel_tol=1e-12)
        rows = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            k_succ = int(rng.binomial(N_TRIALS, p))
            k_fid = int(rng.binomial(k_succ, f)) if k_succ else 0
            rows.append((k_succ / N_TRIALS, k_fid / k_succ if k_succ else float("nan"), k_succ))
        p_hat = float(np.mean([r[0] for r in rows]))
        f_hat = float(np.mean([r[1] for r in rows]))
        kept = float(np.mean([r[2] for r in rows]))
        se = _product_se(p, f, N_TRIALS * len(SEEDS))
        assert abs(p_hat * f_hat - 0.25) < 5.0 * se
        print(f"  U={name:8s}: P={p:.4f} F={f:.4f} F(qg)={f_qg:.4f} P·F={p*f:.6f} "
              f"| medido P={p_hat:.4f} F={f_hat:.4f} P·F={p_hat*f_hat:.4f}±{se:.4f} "
              f"| aceptados={kept:.0f}/{N_TRIALS} (×{N_TRIALS/kept:.2f}) "
              f"m_F={1000*(2*f-1):+.0f} m-qg")
    print("verify_exp18_1: OK")


if __name__ == "__main__":
    main()
