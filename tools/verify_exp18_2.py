"""tools/verify_exp18_2.py — Exp. 18.2: curva de Page en Rényi-2 frente a qg_S plano (sólo base Z)."""
from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

from qang.qscramble import X, Y, Z, qg_S_mean, renyi2_subsystem, haar_unitary

N = 8
SHOTS = 20_000
SEEDS = (11, 12, 13)
PAULIS = (jnp.eye(2, dtype=jnp.complex128), X, Y, Z)


def _expect(psi: jnp.ndarray, qubits: tuple[int, ...], ops: tuple[jnp.ndarray, ...]) -> float:
    n = int(round(math.log2(len(psi))))
    k = len(qubits)
    t = psi.reshape([2] * n)
    t = jnp.moveaxis(t, qubits, tuple(range(k)))
    op = ops[0] if k == 1 else jnp.kron(*ops)
    t = (op @ t.reshape(2**k, -1)).reshape([2] * n)
    t = jnp.moveaxis(t, tuple(range(k)), qubits)
    return float(jnp.real(jnp.vdot(psi, t.reshape(-1))))


def _purity_from_correlators(psi: jnp.ndarray, k: int) -> float:
    total = 1.0
    for indices in itertools.product(range(4), repeat=k):
        if all(i == 0 for i in indices):
            continue
        value = _expect(psi, tuple(range(k)), tuple(PAULIS[i] for i in indices))
        total += value * value
    return total / 2**k


def main() -> None:
    psi = haar_unitary(jax.random.PRNGKey(5), 2**N)[:, 0]
    s2 = [renyi2_subsystem(psi, N, k) for k in range(1, N)]
    qs = [qg_S_mean(psi, N, k) for k in range(1, N)]
    assert s2[3] > s2[0] and s2[3] > s2[-1]
    assert max(qs) - min(qs) < 0.05
    print("Exp. 18.2 — Rényi-2 ve la curva de Page; qg_S (base Z) no mide entrelazamiento")
    print("  k      : " + " ".join(f"{k:5d}" for k in range(1, N)))
    print("  S2(k)  : " + " ".join(f"{v:5.2f}" for v in s2))
    print("  qg_S(k): " + " ".join(f"{v:5.2f}" for v in qs))

    exact = {}
    for k in (1, 2):
        purity = _purity_from_correlators(psi, k)
        expected = 2.0 ** (-s2[k - 1])
        assert abs(purity - expected) < 1e-12
        exact[k] = (purity, s2[k - 1], 4**k - 1)
        print(f"  k={k}: pureza={purity:.9f} = 2^-S2 ({expected:.9f}); "
              f"{exact[k][2]} correladores no identidad (4^k−1)")

    q = np.array([_expect(psi, (0,), (p,)) for p in (X, Y, Z)])
    purity, s2_exact, n_corr = exact[1]
    var_purity = float(np.sum(q**2 * (1 - q**2) / SHOTS))
    sigma = math.sqrt(var_purity) / (purity * math.log(2.0))
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        noisy = rng.normal(q, np.sqrt((1 - q**2) / SHOTS))
        purity_hat = (1.0 + float(np.sum(noisy**2))) / 2.0
        s2_hat = -math.log2(purity_hat)
        assert abs(s2_hat - s2_exact) < 5.0 * sigma
        print(f"  shot ruido semilla {seed}: S2(k=1)={s2_hat:.4f} vs {s2_exact:.4f} "
              f"(±{sigma:.4f}, {n_corr} bases frente a 1 de qg_S)")
    print("verify_exp18_2: OK")


if __name__ == "__main__":
    main()
