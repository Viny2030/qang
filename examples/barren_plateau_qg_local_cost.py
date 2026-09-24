"""
Barren plateaus: a global cost vs the qg local cost, and the classical
control that "dequantizes" the case where qg wins.

Task: train a hardware-efficient ansatz V(p) (L layers: Ry Rz on every
qubit, a CZ chain between layers) so that V(p)|0...0> becomes a product
target |phi> = (x)_q Ry(a_q)|0>. After V we apply the known inverse
(x)_q Ry(-a_q) and measure in the Z basis. Two cost functions that are
both zero exactly at the target:

  * global:   C_G = 1 - P(0...0)                (1 - fidelity)
  * qg local: C_L = (1 - mean qg_Z) / 2 = 1 - (1/n) sum_q P(q = 0)

C_L is the "local cost" of Cerezo et al., Nat. Commun. 12, 1791 (2021);
in the qg language it is simply the mean qg_Z of the register, the same
quantity used as electron-number witness in examples/
chemistry_qg_symmetry_witness.py.

Finding A (gradient variance, 200 random initialisations per n,
derivative w.r.t. the first Ry on qubit 0, n = 2..12):

    shallow L = 2   global    2.7e-2 -> 6.8e-8   (about 2^-1.8 per qubit)
                    qg local  1.8e-2 -> 3.8e-4   (about 1/n^2, polynomial)
    deep L = 4n     global    2.4e-2 -> 3.1e-8
                    qg local  1.6e-2 -> 4.4e-6   (also exponential)

At n = 12 with L = 2 the qg local gradient has 5,600 times the variance
of the global one, i.e. it needs ~5,600 times fewer shots for the same
signal-to-noise. With deep, 2-design-like circuits the qg local cost
does NOT cure the barren plateau (still ~140x better, but exponential).

Finding B (finite shots, L = 2). Fraction of random initialisations
whose whole shot-estimated gradient is exactly zero (no training signal
at all), 100 shots per circuit: global 0 / 0.06 / 0.22 / 0.35 at
n = 6 / 8 / 10 / 12, qg local 0 at every n (1000 shots: global 0.08 and
0.10 at n = 10 and 12). Training at n = 10 with 100 shots (Adam +
parameter shift, 6 seeds): steps to reach fidelity 0.5 have median 11
with qg local (7-16) vs 20 with global (10-68; one seed stalls at
fidelity 0 for 50 steps); final fidelity after 100 steps 0.987-0.990 vs
0.972-0.983. At n <= 8 both costs train equally well: in this range
the advantage is modest; it grows with n (Finding A).

Finding C (classical control: light cone). With depth L each <Z_i>
depends only on the 2L + 1 qubits around i, so mean qg_Z -- and its
exact gradient -- can be computed classically in O(n 2^(2L+1)) time.
It matches the full statevector to 1e-16 at n = 12, and L-BFGS on the
light-cone cost trains n = 50 and n = 100 qubits (300 / 600
parameters) to C_L < 1e-13 (fidelity >= 1 - n C_L > 0.999999999999) in
about 30 s and 100 s on one CPU core, with no quantum computer and no
shots. A statevector of 100 qubits would need 2^100 amplitudes.

Conclusion, consistent with the "QML hype" discussion of the course:
choosing the qg local cost is a real, measurable advantage over the
global cost for the same quantum model (A, B). But the regime in which
it works -- shallow circuits, local cost -- is exactly the regime a
classical light-cone simulation handles (C). Absence of barren plateaus
and classical simulability go together here (Cerezo et al.,
arXiv:2312.09121), so this is an advantage of qg over a bad cost, not
of quantum over classical.
"""

import time

import numpy as np
from scipy.optimize import minimize

# --------------------------------------------------------------------- #
# Batched statevector simulator (qubit q = bit q of the basis index)
# --------------------------------------------------------------------- #


def _apply_1q(psi, mats, q, n):
    """psi: (B, 2^n); mats: (B, 2, 2) or (2, 2)."""
    B = psi.shape[0]
    v = psi.reshape(B, 2 ** (n - 1 - q), 2, 2**q)
    if mats.ndim == 2:
        v = np.einsum("ij,bajc->baic", mats, v)
    else:
        v = np.einsum("bij,bajc->baic", mats, v)
    return v.reshape(B, 2**n)


def _ry(a):
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a / 2), np.sin(a / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(complex)


def _rz(b):
    b = np.asarray(b, dtype=float)
    e = np.exp(-0.5j * b)
    z = np.zeros_like(e)
    return np.stack([np.stack([e, z], -1), np.stack([z, np.conj(e)], -1)], -2)


def _cz_chain_diag(n):
    k = np.arange(2**n)
    d = np.ones(2**n)
    for q in range(n - 1):
        d[((k >> q) & 1) & ((k >> (q + 1)) & 1) == 1] *= -1
    return d


def n_params(n, L):
    return 2 * n * (L + 1)


def output_probabilities(params, target, n, L):
    """Z-basis distribution of (x)Ry(-a) V(p)|0>. params: (B, n_params)."""
    P = np.asarray(params, dtype=float).reshape(-1, L + 1, n, 2)
    B = P.shape[0]
    psi = np.zeros((B, 2**n), dtype=complex)
    psi[:, 0] = 1.0
    cz = _cz_chain_diag(n)
    for layer in range(L + 1):
        if layer > 0 and n > 1:
            psi = psi * cz
        for q in range(n):
            psi = _apply_1q(psi, _ry(P[:, layer, q, 0]), q, n)
            psi = _apply_1q(psi, _rz(P[:, layer, q, 1]), q, n)
    for q in range(n):
        psi = _apply_1q(psi, _ry(-target[q]), q, n)
    return np.abs(psi) ** 2


def _z_signs(n):
    k = np.arange(2**n)
    return np.array([1 - 2 * ((k >> q) & 1) for q in range(n)])  # (n, 2^n)


def global_cost(probs):
    return 1.0 - probs[:, 0]


def qg_local_cost(probs, n):
    """(1 - mean qg_Z) / 2."""
    return (1.0 - (probs @ _z_signs(n).T).mean(axis=1)) / 2.0


COSTS = {"global": lambda p, n: global_cost(p), "qg_local": qg_local_cost}


def random_target(n, seed=123):
    return np.random.default_rng(seed).uniform(0.0, np.pi, n)


# --------------------------------------------------------------------- #
# Finding A: gradient variance
# --------------------------------------------------------------------- #
def gradient_variance(n, L, cost, samples=200, seed=0):
    """Variance over random initialisations of dC/dp_0 (first Ry, qubit 0),
    exact parameter shift."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 2 * np.pi, (samples, n_params(n, L)))
    plus, minus = p.copy(), p.copy()
    plus[:, 0] += np.pi / 2
    minus[:, 0] -= np.pi / 2
    target = random_target(n)
    f = COSTS[cost]
    g = (f(output_probabilities(plus, target, n, L), n) - f(output_probabilities(minus, target, n, L), n)) / 2
    return float(np.var(g))


def decay_rate(ns, variances):
    """Slope of log2 Var vs n (exponential rate) and of log Var vs log n."""
    ns, v = np.asarray(ns, float), np.asarray(variances, float)
    return float(np.polyfit(ns, np.log2(v), 1)[0]), float(np.polyfit(np.log(ns), np.log(v), 1)[0])


# --------------------------------------------------------------------- #
# Finding B: training with finite shots
# --------------------------------------------------------------------- #
def _estimated_cost(probs, n, cost, shots, rng):
    if shots is None:
        return COSTS[cost](probs, n)
    if cost == "global":
        return 1.0 - rng.binomial(shots, np.clip(probs[:, 0], 0, 1)) / shots
    zs = _z_signs(n)
    out = np.empty(probs.shape[0])
    for b, pb in enumerate(probs):
        counts = rng.multinomial(shots, pb / pb.sum())
        out[b] = (1.0 - (zs @ counts).mean() / shots) / 2.0
    return out


def train(n, L, cost, shots=100, steps=100, lr=0.1, seed=0):
    """Adam with parameter-shift gradients estimated from `shots` samples
    per circuit. Returns the exact fidelity with the target after every
    step (length steps + 1, entry 0 = initial state)."""
    rng = np.random.default_rng(seed)
    target = random_target(n)
    m = n_params(n, L)
    p = rng.uniform(0.0, 2 * np.pi, m)
    shifts = np.vstack([np.eye(m), -np.eye(m)]) * (np.pi / 2)
    mom, vel = np.zeros(m), np.zeros(m)
    fid = [float(output_probabilities(p[None], target, n, L)[0, 0])]
    for t in range(1, steps + 1):
        c = _estimated_cost(output_probabilities(p + shifts, target, n, L), n, cost, shots, rng)
        g = (c[:m] - c[m:]) / 2.0
        mom = 0.9 * mom + 0.1 * g
        vel = 0.999 * vel + 0.001 * g * g
        p = p - lr * (mom / (1 - 0.9**t)) / (np.sqrt(vel / (1 - 0.999**t)) + 1e-8)
        fid.append(float(output_probabilities(p[None], target, n, L)[0, 0]))
    return np.array(fid)


def steps_to_fidelity(fid, threshold=0.5):
    hit = np.nonzero(fid >= threshold)[0]
    return int(hit[0]) if hit.size else None


def zero_gradient_fraction(n, L, cost, shots, inits=50, seed=0):
    """Fraction of random initialisations whose whole shot-estimated
    parameter-shift gradient is exactly zero (no training signal)."""
    rng = np.random.default_rng(seed)
    target = random_target(n)
    m = n_params(n, L)
    shifts = np.vstack([np.eye(m), -np.eye(m)]) * (np.pi / 2)
    zero = 0
    for _ in range(inits):
        p = rng.uniform(0.0, 2 * np.pi, m)
        c = _estimated_cost(output_probabilities(p + shifts, target, n, L), n, cost, shots, rng)
        zero += bool(np.all(c[:m] == c[m:]))
    return zero / inits


# --------------------------------------------------------------------- #
# Finding C: classical light-cone evaluation of the qg local cost
# --------------------------------------------------------------------- #
def _cone(i, n, L):
    return max(0, i - L), min(n - 1, i + L)


def mean_qg_z_lightcone(params, target, n, L):
    """Exact mean qg_Z using only (2L+1)-qubit simulations."""
    P = np.asarray(params, dtype=float).reshape(L + 1, n, 2)
    total = 0.0
    for i in range(n):
        lo, hi = _cone(i, n, L)
        w = hi - lo + 1
        probs = output_probabilities(P[:, lo:hi + 1].reshape(1, -1), target[lo:hi + 1], w, L)
        total += float(probs[0] @ _z_signs(w)[i - lo])
    return total / n


def qg_local_cost_and_grad_lightcone(params, target, n, L):
    """C_L and its exact gradient (parameter shift) from light cones."""
    P = np.asarray(params, dtype=float).reshape(L + 1, n, 2)
    zsum, grad = 0.0, np.zeros_like(P)
    for i in range(n):
        lo, hi = _cone(i, n, L)
        w = hi - lo + 1
        sub = P[:, lo:hi + 1].reshape(-1)
        k = sub.size
        batch = np.vstack([sub, sub + np.eye(k) * np.pi / 2, sub - np.eye(k) * np.pi / 2])
        z = output_probabilities(batch, target[lo:hi + 1], w, L) @ _z_signs(w)[i - lo]
        zsum += z[0]
        grad[:, lo:hi + 1] += ((z[1:k + 1] - z[k + 1:]) / 2).reshape(L + 1, w, 2)
    cost = (1.0 - zsum / n) / 2.0
    return float(cost), -grad.reshape(-1) / (2.0 * n)


def train_classical_lightcone(n, L, seed=0, maxiter=500):
    """L-BFGS on the light-cone cost. Returns (C_L, fidelity lower bound
    1 - n C_L, seconds)."""
    rng = np.random.default_rng(seed)
    target = random_target(n)
    t0 = time.perf_counter()
    res = minimize(qg_local_cost_and_grad_lightcone, rng.uniform(0, 2 * np.pi, n_params(n, L)),
                   args=(target, n, L), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-16, "gtol": 1e-12})
    return float(res.fun), float(min(1.0, max(0.0, 1.0 - n * res.fun))), time.perf_counter() - t0


def make_figure(path, ns=(2, 4, 6, 8, 10, 12)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    styles = {"global": ("#c0392b", "global 1 - P(0...0)"), "qg_local": ("#1f6fb2", "qg local (1 - mean qg_Z)/2")}
    for cost, (color, label) in styles.items():
        ax1.semilogy(ns, [gradient_variance(n, 2, cost) for n in ns], "o-", color=color, lw=2, label=f"{label}, L=2")
        deep = [n for n in ns if n <= 10]
        ax1.semilogy(deep, [gradient_variance(n, 4 * n, cost) for n in deep], "s--", color=color, lw=1.5,
                     alpha=0.6, label=f"{label}, L=4n")
        runs = np.array([train(10, 2, cost, seed=s) for s in range(6)])
        ax2.plot(np.median(runs, axis=0), color=color, lw=2, label=label)
        ax2.fill_between(range(runs.shape[1]), runs.min(0), runs.max(0), color=color, alpha=0.15)
    ax1.set_xlabel("qubits n")
    ax1.set_ylabel("Var(dC/dp)")
    ax1.set_title("A. Gradient variance")
    ax1.legend(fontsize=7)
    ax2.set_xlabel("Adam step")
    ax2.set_ylabel("fidelity with target")
    ax2.set_title("B. Training, n = 10, 100 shots (6 seeds)")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    import sys

    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"))
        sys.exit()
    ns = [2, 4, 6, 8, 10, 12]
    print("A: Var(dC/dp0), 200 random inits")
    for label, depth, nn in [("shallow L=2", lambda n: 2, ns), ("deep L=4n", lambda n: 4 * n, ns)]:
        print(f"  {label}")
        for cost in ("global", "qg_local"):
            v = [gradient_variance(n, depth(n), cost) for n in nn]
            exp_rate, pow_rate = decay_rate(nn, v)
            print(f"    {cost:9s} " + " ".join(f"{x:.1e}" for x in v)
                  + f"   log2-slope {exp_rate:+.2f}/qubit, log-log slope {pow_rate:+.2f}")
    print("B1: fraction of inits with an all-zero shot-estimated gradient (L=2)")
    for n in (6, 8, 10, 12):
        row = [f"{c} {s} shots: {zero_gradient_fraction(n, 2, c, s, inits=20 if n == 12 else 50):.2f}"
               for s in (100, 1000) for c in ("global", "qg_local")]
        print(f"  n={n:2d}  " + "  ".join(row))
    print("B2: training n=10, L=2, 100 shots, 6 seeds")
    for c in ("global", "qg_local"):
        runs = [train(10, 2, c, seed=s) for s in range(6)]
        print(f"  {c:9s} steps to F>=0.5: {[steps_to_fidelity(f) for f in runs]}  "
              f"final F: {[round(float(f[-1]), 3) for f in runs]}")
    print("C: classical light cone")
    rng = np.random.default_rng(5)
    n, L = 12, 2
    p, tg = rng.uniform(0, 2 * np.pi, n_params(n, L)), random_target(n)
    full = float((1 - 2 * qg_local_cost(output_probabilities(p[None], tg, n, L), n))[0])
    print(f"  n=12 mean qg_Z full {full:+.15f}  light cone {mean_qg_z_lightcone(p, tg, n, L):+.15f}")
    for n in (12, 50, 100):
        c, fid, sec = train_classical_lightcone(n, 2)
        print(f"  n={n:3d}  C_L {c:.1e}  fidelity >= {fid:.12f}  ({sec:.1f} s, no shots)")
