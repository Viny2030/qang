"""
Differential equations with a quantum model: qg encoding vs angle
encoding, with classical spectral methods as the control.

Differentiable-quantum-circuit (DQC) solver (Kyriienko, Paine & Elfving
2021): a single-qubit data re-uploading circuit (the model of
examples/qml_encoding_qg_vs_angle.py, L layers, 3(L + 1) rotation
parameters plus one output scale s) represents the solution as

    u(x) = u0 + s * (f(x) - f(0)),     f(x) = <Z>     ("floating boundary"),

so the initial condition holds exactly. x in [0, 1] is mapped to
z in [-0.9, 0.9] (away from the arccos poles, where the qg chain rule
d theta/dz = -1/sqrt(1 - z^2) diverges). df/dx is exact: the parameter-
shift rule on each of the L encoding gates gives df/d theta, times
d theta/dx. Training minimizes the mean squared ODE residual
(du/dx - F(x, u))^2 on 40 collocation points (best of 4 L-BFGS restarts);
the score is the maximum error against the exact solution on 201 points.

Encodings: qg (theta = arccos z, polynomial / Chebyshev bias) and angle
(theta = alpha z, Fourier bias; alpha = pi/2 and pi).

Findings (max error):

  problem                          L | qg      | angle pi/2 | angle pi | classical, same degree | classical, same #params
  decay u' = -2u                   2 | 4.4e-3  | 1.0e-2     | 0.72     | 2.7e-2                 | 2e-11
                                   3 | 9.7e-5  | 1.7e-3     | 0.63     | 3.6e-3                 | 1e-15
                                   4 | 3.8e-5  | 2.5e-4     | 0.53     | 3.9e-4                 | 2e-16
  logistic u' = 4u(1 - u)          3 | 1.2e-3  | 6.6e-4     | 0.76     | 1.6e-2                 | 8e-8
  damped oscillation (Kyriienko)   3 | 3.3e-3  | 6.5e-3     | 2.0      | 1.8e-1                 | 1e-11
  u' = pi cos(pi x)                3 | 1.4e-3  | 5.4e-5     | 2.9e-2   | 5.8e-2                 | 2e-12

  * qg wins clearly on exponential decay: 2.3x, 17x and 6.6x better than
    the best angle scale at L = 2, 3, 4 (4x for u' = -4u at L = 3), same
    result with another seed.
  * It does not win in general: angle pi/2 is ~2x better on the logistic
    equation, 25x better on the sine at L = 2-3, and the damped
    oscillation alternates (angle at L = 2 and 4, qg at L = 3).
  * The angle scale matters a lot (alpha = pi fails everywhere here);
    qg has no scale to tune.
  * Classical control: a Chebyshev spectral (collocation) fit with as
    many coefficients as the quantum model has parameters solves every
    problem to 1e-7 or better, usually to machine precision. With the
    same polynomial DEGREE (fewer parameters) the quantum models do
    better, which only reflects their extra parameters. At this size
    classical spectral methods win outright; the result is about which
    encoding a quantum DE solver should use, not about beating classical
    solvers.
"""

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.optimize import least_squares, minimize

Z_LO, Z_HI = -0.9, 0.9
X_COLLOCATION = np.linspace(0.0, 1.0, 40)
X_TEST = np.linspace(0.0, 1.0, 201)
OPTS = {"ftol": 1e-15, "gtol": 1e-10, "maxiter": 3000}


def _rz(a):
    return np.array([[np.exp(-0.5j * a), 0.0], [0.0, np.exp(0.5j * a)]])


def _ry(a):
    c, s = np.cos(a / 2), np.sin(a / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _w(p):
    return _rz(p[2]) @ _ry(p[1]) @ _rz(p[0])


def model_and_theta_derivative(params, theta, n_layers):
    """f = <Z> for a batch of encoding angles, and df/dtheta (exact, by the
    parameter-shift rule applied to each of the n_layers encoding gates)."""
    P = np.asarray(params).reshape(n_layers + 1, 3)
    ws = [_w(P[l]) for l in range(n_layers + 1)]

    def run(shifts):
        psi = np.zeros((len(theta), 2), dtype=complex)
        psi[:, 0] = 1.0
        psi = psi @ ws[0].T
        for l in range(1, n_layers + 1):
            a = theta + shifts[l - 1]
            c, s = np.cos(a / 2), np.sin(a / 2)
            psi = np.stack([c * psi[:, 0] - s * psi[:, 1], s * psi[:, 0] + c * psi[:, 1]], axis=1) @ ws[l].T
        return np.abs(psi[:, 0]) ** 2 - np.abs(psi[:, 1]) ** 2

    f = run([0.0] * n_layers)
    df = np.zeros_like(f)
    for l in range(n_layers):
        plus, minus = [0.0] * n_layers, [0.0] * n_layers
        plus[l], minus[l] = np.pi / 2, -np.pi / 2
        df += 0.5 * (run(plus) - run(minus))
    return f, df


ENCODINGS = {
    "qg": (lambda z: np.arccos(z), lambda z: -1.0 / np.sqrt(1.0 - z**2)),
    "angle_pi/2": (lambda z: 0.5 * np.pi * z, lambda z: 0.5 * np.pi * np.ones_like(z)),
    "angle_pi": (lambda z: np.pi * z, lambda z: np.pi * np.ones_like(z)),
}

PROBLEMS = {
    "decay": (lambda x, u: -2.0 * u, 1.0, lambda x: np.exp(-2.0 * x)),
    "decay_k4": (lambda x, u: -4.0 * u, 1.0, lambda x: np.exp(-4.0 * x)),
    "logistic": (lambda x, u: 4.0 * u * (1.0 - u), 0.2, lambda x: 1.0 / (1.0 + 4.0 * np.exp(-4.0 * x))),
    "damped_oscillation": (lambda x, u: -4.0 * u * (0.2 + np.tan(4.0 * x)), 1.0,
                           lambda x: np.exp(-0.8 * x) * np.cos(4.0 * x)),
    "sine": (lambda x, u: np.pi * np.cos(np.pi * x), 0.0, lambda x: np.sin(np.pi * x)),
}


def solve_quantum(encoding, problem, n_layers, restarts=4, seed=0):
    """Train the DQC model; return the max error on X_TEST."""
    F, u0, exact = PROBLEMS[problem]
    to_theta, dtheta_dz = ENCODINGS[encoding]
    scale = Z_HI - Z_LO
    z = Z_LO + scale * X_COLLOCATION
    theta, dtheta_dx = to_theta(z), dtheta_dz(z) * scale
    theta0 = to_theta(np.array([Z_LO]))

    def u_and_du(p, th, dth):
        s = p[-1]
        f, df = model_and_theta_derivative(p[:-1], th, n_layers)
        f0, _ = model_and_theta_derivative(p[:-1], theta0, n_layers)
        return u0 + s * (f - f0[0]), s * df * dth

    def loss(p):
        u, du = u_and_du(p, theta, dtheta_dx)
        return float(np.mean((du - F(X_COLLOCATION, u)) ** 2))

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        p0 = np.concatenate([rng.uniform(-np.pi, np.pi, 3 * (n_layers + 1)), [1.0]])
        res = minimize(loss, p0, method="L-BFGS-B", options=OPTS)
        if best is None or res.fun < best.fun:
            best = res
    zt = Z_LO + scale * X_TEST
    u, _ = u_and_du(best.x, to_theta(zt), dtheta_dz(zt) * scale)
    return float(np.max(np.abs(u - exact(X_TEST))))


def solve_classical_chebyshev(problem, degree):
    """Chebyshev spectral collocation (least squares on the residual, initial
    condition as a heavily weighted extra equation); max error on X_TEST."""
    F, u0, exact = PROBLEMS[problem]
    t = 2.0 * X_COLLOCATION - 1.0

    def residual(c):
        u = C.chebval(t, c)
        du = 2.0 * C.chebval(t, C.chebder(c))
        return np.concatenate([du - F(X_COLLOCATION, u), [1e3 * (C.chebval(-1.0, c) - u0)]])

    sol = least_squares(residual, np.zeros(degree + 1), xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return float(np.max(np.abs(C.chebval(2.0 * X_TEST - 1.0, sol.x) - exact(X_TEST))))


if __name__ == "__main__":
    for problem in ("decay", "logistic", "damped_oscillation", "sine"):
        print(problem)
        for L in (2, 3, 4):
            q = "  ".join(f"{e}: {solve_quantum(e, problem, L):.1e}" for e in ENCODINGS)
            print(f"  L={L}  {q}  | classical deg {L}: {solve_classical_chebyshev(problem, L):.1e}"
                  f"  deg {3 * (L + 1) + 1} (same #params): {solve_classical_chebyshev(problem, 3 * (L + 1) + 1):.1e}")
