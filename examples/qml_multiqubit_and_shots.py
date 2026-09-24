"""
Does the qg-encoding result of examples/qml_encoding_qg_vs_angle.py
survive (A) more qubits, (B) two input features, and (C) training with
finite measurement shots? And how does it compare with a plain classical
fit?

Model: n qubits; in each of L layers every qubit loads a feature with
Ry(theta(x)), then a CZ ring entangles the register and a trainable
Rz Ry Rz rotation acts on every qubit (3 n (L + 1) parameters). The output
is <Z_0>. Encodings: qg (theta = arccos x) and angle (theta = alpha x,
alpha = pi/2 or pi). Fits use the best of 6 L-BFGS restarts (80 training
points in 1D, 150 random points in 2D; test MSE on a grid).

Finding A (1D input, 2 qubits, L = 2). The polynomial bias survives:
x^3 - 0.5x is fitted to 1e-13 with qg vs 5e-5 with the best angle scale,
while sin(pi x) and tanh(4x) are better with angle encoding. A random
degree-4 polynomial -- inside the degree budget n * L = 4 -- is NOT fitted
exactly (2.7e-3; still the best of the three): n * L bounds the degree
the circuit can express, but a shallow CZ-ring architecture does not
always reach every polynomial within that bound.

Finding B (2D input, one feature per qubit, 2 qubits, L = 2). x1 * x2 is
exact with qg (2e-16) vs 2.4e-4 with angle pi/2; the mixed polynomial
0.5 T2(x1) + 0.5 x1 x2^2 is best with qg but not exact (2.1e-3 vs 6.4e-3);
sin(pi x1) cos(pi x2) is exact with angle pi (9e-18) and poor with qg
(9e-2); tanh(2(x1 + x2)) is best with angle pi/2. Same pattern as in 1D:
qg wins on polynomial targets, angle on periodic or saturating ones.

Finding C (finite shots). Training x^3 - 0.5x on one qubit with L = 3
using Adam and parameter-shift gradients estimated from N shots per
circuit (1500 steps, lr 0.02, 5 seeds, median test MSE):

    N = infinite: qg 1.8e-5, angle pi/2 4.2e-4
    N = 10,000:   qg 4.5e-5, angle pi/2 4.2e-4
    N = 1,000:    qg 6.2e-5, angle pi/2 5.5e-4

The advantage survives shot noise but shrinks from "exact" (L-BFGS
without noise) to about 10x. A gradient optimizer also does not reach
the machine-precision fits of Finding A in 1500 steps.

Classical control. A least-squares Chebyshev fit of degree 3 to the same
80 points (numpy, no quantum resources) reaches ~6e-33 for x^3 - 0.5x.
These single- and two-qubit models are classically simulable; the qg
result is a statement about which inductive bias a quantum model should
be given, not a quantum advantage over classical regression.
"""

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.optimize import minimize

OPTS = {"ftol": 1e-15, "gtol": 1e-10, "maxiter": 4000}
X1_TRAIN = np.sort(np.random.default_rng(0).uniform(-1.0, 1.0, 80))
X1_TEST = np.linspace(-1.0, 1.0, 201)
X2_TRAIN = np.random.default_rng(1).uniform(-1.0, 1.0, (150, 2))
_g = np.linspace(-1.0, 1.0, 21)
X2_TEST = np.array([[a, b] for a in _g for b in _g])

ENCODINGS = {
    "qg": lambda x: np.arccos(np.clip(x, -1.0, 1.0)),
    "angle_pi/2": lambda x: 0.5 * np.pi * x,
    "angle_pi": lambda x: np.pi * x,
}


def _rz(a):
    return np.array([[np.exp(-0.5j * a), 0.0], [0.0, np.exp(0.5j * a)]])


def _ry(a):
    c, s = np.cos(a / 2), np.sin(a / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _w(p):
    return _rz(p[2]) @ _ry(p[1]) @ _rz(p[0])


def _kron_all(mats):
    out = np.array([[1.0 + 0j]])
    for m in mats[::-1]:  # qubit 0 is the least significant bit
        out = np.kron(out, m)
    return out


def _cz_ring_diag(n):
    d = 2**n
    diag = np.ones(d)
    pairs = [(0, 1)] if n == 2 else ([(i, (i + 1) % n) for i in range(n)] if n > 2 else [])
    for a, b in pairs:
        for k in range(d):
            if (k >> a) & 1 and (k >> b) & 1:
                diag[k] *= -1
    return diag


def model_output(params, enc_angles, n, L):
    """<Z_0> for a batch; enc_angles has shape (batch, n)."""
    m, d = enc_angles.shape[0], 2**n
    P = np.asarray(params).reshape(L + 1, n, 3)
    psi = np.zeros((m, d), dtype=complex)
    psi[:, 0] = 1.0
    psi = psi @ _kron_all([_w(P[0, q]) for q in range(n)]).T
    cz = _cz_ring_diag(n)
    for layer in range(1, L + 1):
        psi = psi.reshape((m,) + (2,) * n)
        for q in range(n):
            ax = n - q
            shape = (m,) + (1,) * (n - 1)
            c = np.cos(enc_angles[:, q] / 2).reshape(shape)
            s = np.sin(enc_angles[:, q] / 2).reshape(shape)
            a, b = np.take(psi, 0, axis=ax), np.take(psi, 1, axis=ax)
            psi = np.stack([c * a - s * b, s * a + c * b], axis=ax)
        psi = psi.reshape(m, d)
        if n > 1:
            psi = psi * cz
        psi = psi @ _kron_all([_w(P[layer, q]) for q in range(n)]).T
    z0 = np.array([1.0 if not (k & 1) else -1.0 for k in range(d)])
    return (np.abs(psi) ** 2) @ z0


def fit(encoding, target, x_train, x_test, n, L, restarts=6, seed=0):
    """Best-of-restarts L-BFGS fit; x arrays have shape (batch, n). Returns test MSE."""
    rng = np.random.default_rng(seed)
    a_tr, a_te = encoding(x_train), encoding(x_test)
    y_tr, y_te = target(x_train), target(x_test)
    best = None
    for _ in range(restarts):
        loss = lambda p: np.mean((model_output(p, a_tr, n, L) - y_tr) ** 2)
        res = minimize(loss, rng.uniform(-np.pi, np.pi, 3 * n * (L + 1)), method="L-BFGS-B", options=OPTS)
        test = np.mean((model_output(res.x, a_te, n, L) - y_te) ** 2)
        if best is None or res.fun < best[0]:
            best = (res.fun, float(test))
    return best[1]


def train_with_shots(encoding, target, L, shots, steps=1500, lr=0.02, seed=0):
    """One qubit, Adam with parameter-shift gradients estimated from
    `shots` samples per circuit evaluation (None = exact). Returns test MSE."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(-np.pi, np.pi, 3 * (L + 1))
    a_tr = encoding(X1_TRAIN)[:, None]
    y = target(X1_TRAIN)

    def estimate(pp):
        f = model_output(pp, a_tr, 1, L)
        if shots is None:
            return f
        k = rng.binomial(shots, (1.0 + np.clip(f, -1.0, 1.0)) / 2.0)
        return 2.0 * k / shots - 1.0

    m, v = np.zeros_like(p), np.zeros_like(p)
    for t in range(1, steps + 1):
        f = estimate(p)
        g = np.empty_like(p)
        for i in range(len(p)):
            e = np.zeros_like(p)
            e[i] = np.pi / 2
            g[i] = np.mean(2.0 * (f - y) * (estimate(p + e) - estimate(p - e)) / 2.0)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        p = p - lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return float(np.mean((model_output(p, encoding(X1_TEST)[:, None], 1, L) - target(X1_TEST)) ** 2))


def classical_chebyshev_fit(target, degree):
    coeffs = C.chebfit(X1_TRAIN, target(X1_TRAIN), degree)
    return float(np.mean((C.chebval(X1_TEST, coeffs) - target(X1_TEST)) ** 2))


def cubic(x):
    return x**3 - 0.5 * x


def _random_degree4():
    c = np.random.default_rng(7).normal(size=5)
    c *= 0.95 / np.max(np.abs(C.chebval(np.linspace(-1, 1, 2001), c)))
    return c


_DEG4 = _random_degree4()

TARGETS_1D = {
    "random degree-4 poly": lambda X: C.chebval(X[:, 0], _DEG4),
    "x^3 - 0.5x": lambda X: cubic(X[:, 0]),
    "sin(pi x)": lambda X: np.sin(np.pi * X[:, 0]),
    "tanh(4x)": lambda X: np.tanh(4 * X[:, 0]),
}
TARGETS_2D = {
    "x1 * x2": lambda X: X[:, 0] * X[:, 1],
    "0.5 T2(x1) + 0.5 x1 x2^2": lambda X: 0.5 * (2 * X[:, 0] ** 2 - 1) + 0.5 * X[:, 0] * X[:, 1] ** 2,
    "sin(pi x1) cos(pi x2)": lambda X: np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1]),
    "tanh(2(x1 + x2))": lambda X: np.tanh(2 * (X[:, 0] + X[:, 1])),
}


if __name__ == "__main__":
    x1tr, x1te = np.repeat(X1_TRAIN[:, None], 2, 1), np.repeat(X1_TEST[:, None], 2, 1)
    print("A: 1D input on 2 qubits, L = 2 (test MSE)")
    for name, t in TARGETS_1D.items():
        print(f"  {name:26s}" + "  ".join(f"{e}: {fit(f, t, x1tr, x1te, 2, 2):.1e}" for e, f in ENCODINGS.items()))
    print("B: 2D input, one feature per qubit, 2 qubits, L = 2")
    for name, t in TARGETS_2D.items():
        print(f"  {name:26s}" + "  ".join(f"{e}: {fit(f, t, X2_TRAIN, X2_TEST, 2, 2):.1e}" for e, f in ENCODINGS.items()))
    print("C: one qubit, L = 3, x^3 - 0.5x, Adam + parameter shift, median of 5 seeds")
    for shots in (None, 10_000, 1_000):
        row = []
        for e in ("qg", "angle_pi/2"):
            vals = [train_with_shots(ENCODINGS[e], cubic, 3, shots, seed=s) for s in range(5)]
            row.append(f"{e}: {np.median(vals):.1e}")
        print(f"  shots={shots!s:>6}  " + "  ".join(row))
    print(f"Classical Chebyshev least squares, degree 3: {classical_chebyshev_fit(cubic, 3):.1e}")
