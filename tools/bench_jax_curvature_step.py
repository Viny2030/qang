"""Does the qg_kappa step help an optimizer near the poles? (book, chapter 1, "Magnitudes derivadas").

The treatise proposes damping an optimizer's step with the information curvature qg_kappa. On the coordinate
q = qg_Z that is the natural-gradient step q <- q - lr (1 - q^2) dE/dq, which to first order is plain gradient descent
on the qang angle theta = arccos q. This script minimizes the one-qubit energy E = <a Z - b X> on the meridian phi = 0,
E(q) = a q - b sqrt(1 - q^2), whose ground state q* = -a / sqrt(a^2 + b^2) sits near the pole |1> when b << a, with
three optimizers (the sign of X keeps the minimum on the branch theta in [0, pi] that q reaches; with +b X it lies on
the other branch, out of reach of q alone: the branch sign of chapter 1):

  * plain gradient descent on q (the pathology of chapter 1: dE/dq diverges at the poles);
  * the qg_kappa step on q (``qang.integrations.jax.curvature_damped_update``);
  * plain gradient descent on theta.

For each (b, lr) it runs 41 starting points q0 in [-0.98, 0.98] and reports how many reach |E - E_min| < 1e-8 within
2000 steps and the median number of steps. Prints the table; exits 0. Uses JAX in float64.

    .venv-int/Scripts/python tools/bench_jax_curvature_step.py
"""

import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from qang.integrations import jax as qj  # noqa: E402

A = 1.0
STEPS = 2000
TOL = 1e-8
BOUND = 1.0 - 1e-12


def run(method, b, lr, q0):
    e_q = lambda q: A * q - b * jnp.sqrt(1.0 - q * q)
    e_t = lambda t: A * jnp.cos(t) - b * jnp.sin(t)
    e_min = -jnp.sqrt(A * A + b * b)
    if method == "theta":
        g = jax.grad(e_t)

        def body(t, _):
            return t - lr * g(t), e_t(t) - e_min

        x0 = jnp.arccos(q0)
    else:
        g = jax.grad(e_q)
        if method == "q":
            def body(q, _):
                return jnp.clip(q - lr * g(q), -BOUND, BOUND), e_q(q) - e_min
        else:
            def body(q, _):
                return qj.curvature_damped_update(q, g(q), lr, bound=BOUND), e_q(q) - e_min
        x0 = q0
    _, gaps = jax.lax.scan(body, x0, None, length=STEPS)
    gaps = jnp.abs(gaps)
    hit = gaps < TOL
    return bool(hit.any()), int(jnp.argmax(hit)) if bool(hit.any()) else None


def main():
    starts = jnp.linspace(-0.98, 0.98, 41)
    print(f"{'b':>5} {'lr':>6} | {'método':<22} | converge | pasos (mediana)")
    for b in (0.3, 0.1, 0.03):
        for lr in (0.01, 0.1, 0.5):
            for method, label in (("q", "gradiente en q"), ("kappa", "paso qg_κ en q"), ("theta", "gradiente en θ")):
                res = [run(method, b, lr, float(q0)) for q0 in starts]
                steps = sorted(s for ok, s in res if ok)
                med = steps[len(steps) // 2] if steps else "—"
                print(f"{b:>5} {lr:>6} | {label:<22} | {len(steps):>3}/{len(res)} | {med}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
