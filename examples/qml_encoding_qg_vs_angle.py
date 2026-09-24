"""
QML data encoding: qg encoding theta = arccos(x) vs plain angle encoding
theta = alpha * x, in a single-qubit data re-uploading regressor.

The model (Perez-Salinas et al. 2020; Schuld, Sweke & Meyer 2021):

    |psi(x)> = W_L Ry(theta(x)) W_{L-1} ... W_1 Ry(theta(x)) W_0 |0>,
    f(x)     = <psi(x)| Z |psi(x)>,

with each W_l a general single-qubit rotation Rz Ry Rz (3 trainable
parameters), so every encoding below gets exactly 3(L + 1) parameters.
Each Ry(theta) contributes frequencies +-1/2, so f is a trigonometric
polynomial of degree L in theta:

    f = sum_{k<=L} a_k cos(k theta) + b_k sin(k theta).

  * Angle encoding, theta = alpha * x: f is a truncated Fourier series in
    x with frequencies k * alpha (Schuld et al. 2021). The scale alpha is a
    hyperparameter that has to match the target.
  * qg encoding, theta = arccos(x), i.e. qg_Z of the encoded state equals
    the feature x itself: cos(k theta) = T_k(x) and sin(k theta) =
    sqrt(1 - x^2) U_{k-1}(x), so f is a Chebyshev polynomial of degree L
    in x (plus sqrt(1 - x^2) times a polynomial of degree L - 1). This is
    the single-qubit case of quantum signal processing (Low & Chuang 2017;
    Martyn et al. 2021) and of Chebyshev feature maps (Kyriienko et al.
    2021). With W_l = identity, f = cos(L arccos x) = T_L(x) exactly.

Findings (80 training points on [-1, 1], 201 test points, best of 8
L-BFGS restarts by training loss, tight tolerances; test MSE):

  Finding A (the qg advantage, structural): with L re-uploadings the qg
  encoding fits random bounded polynomials of degree L to machine
  precision (test MSE 1e-10 or below) and fails with L - 1 (1e-4 to
  1e-2). For x^3 - 0.5 x at L = 3, qg reaches 2e-15, while the best
  fixed-scale angle encoding reaches 1.5e-4 and angle encoding with a
  trainable scale and offset per layer (2L extra parameters) reaches
  3e-5. The layer count is the polynomial degree, which gives a direct
  rule for sizing the circuit when the target is (close to) a low-degree
  polynomial of a bounded feature -- expectation values, smooth physical
  responses, solutions of differential equations.

  Finding B (no universal winner): for targets that are not low-degree
  polynomials the Fourier bias of angle encoding is better. sin(pi x) is
  exact at L = 1 with alpha = pi, while qg only reaches 1.5e-4 at L = 4.
  On tanh(4x), the Runge function and |x| at L = 4, angle encoding with
  a trainable scale beats qg by 1.5 to 4.5 orders of magnitude (Runge:
  7e-6 vs 7e-2), and the best fixed scale beats it by 0.8 to 1.5. The
  price of angle encoding is the scale: with the wrong one (alpha = 1 on
  cos(2 pi x)) it fails completely (0.34 at L = 4).

The honest summary: qg encoding is not better in general. It swaps the
Fourier inductive bias for a polynomial one, with no scale to tune.
"""

import numpy as np
from scipy.optimize import minimize

X_TRAIN = np.sort(np.random.default_rng(0).uniform(-1.0, 1.0, 80))
X_TEST = np.linspace(-1.0, 1.0, 201)
_OPTS = {"ftol": 1e-15, "gtol": 1e-10, "maxiter": 3000}


def _rz(a):
    return np.array([[np.exp(-0.5j * a), 0.0], [0.0, np.exp(0.5j * a)]])


def _ry(a):
    c, s = np.cos(a / 2), np.sin(a / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _w(p):
    return _rz(p[2]) @ _ry(p[1]) @ _rz(p[0])


def reupload_output(params, enc_angles, n_layers):
    """<Z> of the re-uploading circuit for a batch of encoding angles."""
    p = np.asarray(params).reshape(n_layers + 1, 3)
    psi = np.zeros((len(enc_angles), 2), dtype=complex)
    psi[:, 0] = 1.0
    psi = psi @ _w(p[0]).T
    c, s = np.cos(enc_angles / 2), np.sin(enc_angles / 2)
    for layer in range(1, n_layers + 1):
        a = c * psi[:, 0] - s * psi[:, 1]
        b = s * psi[:, 0] + c * psi[:, 1]
        psi = np.stack([a, b], axis=1) @ _w(p[layer]).T
    return np.abs(psi[:, 0]) ** 2 - np.abs(psi[:, 1]) ** 2


ENCODINGS = {
    "qg": lambda x: np.arccos(np.clip(x, -1.0, 1.0)),
    "angle_1": lambda x: x,
    "angle_pi/2": lambda x: 0.5 * np.pi * x,
    "angle_pi": lambda x: np.pi * x,
}

TARGETS = {
    "x^3 - 0.5x": lambda x: x**3 - 0.5 * x,
    "tanh(4x)": lambda x: np.tanh(4 * x),
    "Runge": lambda x: 2.0 / (1.0 + 25.0 * x**2) - 1.0,
    "|x|": lambda x: 2.0 * np.abs(x) - 1.0,
    "sin(pi x)": lambda x: np.sin(np.pi * x),
    "cos(2 pi x)": lambda x: np.cos(2 * np.pi * x),
}


def fit(encoding, target, n_layers, restarts=8, seed=0):
    """Best-of-restarts L-BFGS fit; returns (train MSE, test MSE)."""
    rng = np.random.default_rng(seed)
    a_tr, a_te = encoding(X_TRAIN), encoding(X_TEST)
    y_tr, y_te = target(X_TRAIN), target(X_TEST)
    best = None
    for _ in range(restarts):
        p0 = rng.uniform(-np.pi, np.pi, 3 * (n_layers + 1))
        loss = lambda p: np.mean((reupload_output(p, a_tr, n_layers) - y_tr) ** 2)
        res = minimize(loss, p0, method="L-BFGS-B", options=_OPTS)
        test = np.mean((reupload_output(res.x, a_te, n_layers) - y_te) ** 2)
        if best is None or res.fun < best[0]:
            best = (float(res.fun), float(test))
    return best


def reupload_output_trainable_scale(params, x, n_layers):
    """Angle encoding with a trainable scale and offset per layer,
    theta_l = w_l * x + b_l (2 extra parameters per layer)."""
    p = np.asarray(params)
    rot = p[: 3 * (n_layers + 1)].reshape(n_layers + 1, 3)
    sc = p[3 * (n_layers + 1):].reshape(n_layers, 2)
    psi = np.zeros((len(x), 2), dtype=complex)
    psi[:, 0] = 1.0
    psi = psi @ _w(rot[0]).T
    for layer in range(1, n_layers + 1):
        th = sc[layer - 1, 0] * x + sc[layer - 1, 1]
        c, s = np.cos(th / 2), np.sin(th / 2)
        a = c * psi[:, 0] - s * psi[:, 1]
        b = s * psi[:, 0] + c * psi[:, 1]
        psi = np.stack([a, b], axis=1) @ _w(rot[layer]).T
    return np.abs(psi[:, 0]) ** 2 - np.abs(psi[:, 1]) ** 2


def fit_trainable_scale(target, n_layers, restarts=8, seed=0):
    rng = np.random.default_rng(seed)
    y_tr, y_te = target(X_TRAIN), target(X_TEST)
    best = None
    for _ in range(restarts):
        p0 = np.concatenate([
            rng.uniform(-np.pi, np.pi, 3 * (n_layers + 1)),
            rng.uniform(0.5, 3.0, 2 * n_layers) * np.tile([1.0, 0.3], n_layers),
        ])
        loss = lambda p: np.mean((reupload_output_trainable_scale(p, X_TRAIN, n_layers) - y_tr) ** 2)
        res = minimize(loss, p0, method="L-BFGS-B", options=_OPTS)
        test = np.mean((reupload_output_trainable_scale(res.x, X_TEST, n_layers) - y_te) ** 2)
        if best is None or res.fun < best[0]:
            best = (float(res.fun), float(test))
    return best


def random_bounded_chebyshev(degree, rng, bound=0.95):
    """A random polynomial of exact degree `degree`, scaled so that
    max |p(x)| on [-1, 1] equals `bound`."""
    from numpy.polynomial import chebyshev as C

    coeffs = rng.normal(size=degree + 1)
    grid = np.linspace(-1, 1, 2001)
    coeffs *= bound / np.max(np.abs(C.chebval(grid, coeffs)))
    return lambda x, c=coeffs: C.chebval(x, c)


if __name__ == "__main__":
    print("Test MSE, best of 8 restarts (3(L+1) parameters for every encoding)\n")
    for name, target in TARGETS.items():
        print(name)
        for n_layers in (1, 2, 3, 4):
            row = "  ".join(f"{e}: {fit(enc, target, n_layers)[1]:.1e}" for e, enc in ENCODINGS.items())
            tr = fit_trainable_scale(target, n_layers)[1]
            print(f"  L={n_layers}  {row}  trainable-scale angle: {tr:.1e}")
    print("\nFinding A: random bounded polynomials of degree d, qg encoding")
    rng = np.random.default_rng(5)
    for d in (2, 3, 4, 5):
        tgt = random_bounded_chebyshev(d, rng)
        print(f"  degree {d}: L = d -> {fit(ENCODINGS['qg'], tgt, d, restarts=12)[1]:.1e}   "
              f"L = d-1 -> {fit(ENCODINGS['qg'], tgt, d - 1, restarts=12)[1]:.1e}")
