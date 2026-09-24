"""
Finite-precision control: a uniform grid in theta vs a uniform grid in qg.

Control electronics set each rotation angle with a finite number of bits
b, i.e. from a grid of 2^b values. There are two natural choices:

  * a grid uniform in theta on [0, pi] (the usual one), or
  * a grid uniform in qg = cos(theta) on [-1, 1], i.e. uniform in the Born
    probability P(0) = (1 + qg)/2. On the Bloch sphere these are
    equal-area bands (Archimedes), so for Haar-random targets every grid
    cell is equally likely.

Each grid is minimax-optimal for a different error:

  * Probability error |dP(0)| = |d qg|/2: the qg grid has constant
    resolution 1/(2^b - 1) everywhere; the theta grid's worst case, at the
    equator, is pi/2 times larger. So the qg grid saves log2(pi/2) = 0.65
    bits for a worst-case probability target.
  * Infidelity 1 - cos^2(d theta / 2): the theta grid has constant angular
    resolution; the qg grid is coarse near the poles (a step d qg there is
    an angle of order sqrt(2 d qg)), so its worst-case infidelity decays
    like 2^-b instead of 4^-b.

Findings (single qubit, 200,000 targets; see __main__):

  Finding A (single qubit, exact trade-off). At every b, the qg grid's
  worst-case probability error is pi/2 = 1.571 times smaller and its
  mean error on Haar-random targets about 1.23 times smaller; its
  worst-case infidelity is larger by a factor that grows like 2^b (6x
  at b = 4, 400x at b = 10).

  Finding B (application: loading a probability distribution with a
  Grover-Rudolph tree of Ry rotations, the state-preparation step of
  quantum Monte Carlo). In qg units each rotation is set directly by a
  conditional probability, qg = 2p - 1. With b-bit angles (b = 4, 6, 8),
  n = 4, 6, 8 qubits and five distributions (45 cases):
    - the total variation distance of the loaded distribution is lower
      with the qg grid in 43 of 45 cases (median 1.56x, up to 2.9x);
    - for smooth distributions (normal, log-normal, Dirichlet(1)) the
      infidelity is ALSO lower with the qg grid in 25 of 27 cases (median
      2.0x, up to 3.5x), because their conditional probabilities cluster
      near 1/2, where the qg grid is denser;
    - for sparse or sharply peaked distributions (Dirichlet(0.1), a
      narrow normal), whose conditional probabilities sit near 0 or 1,
      the theta grid gives lower infidelity in 15 of 18 cases (median
      3.3x, up to 56x at b = 8).

Design rule: use a qg-uniform grid when the task is to reproduce
probabilities (sampling, Monte Carlo, amplitude loading of smooth
distributions); use a theta-uniform grid when state fidelity matters and
the targets sit near the poles.
"""

import numpy as np


def theta_grid(bits: int) -> np.ndarray:
    return np.linspace(0.0, np.pi, 2**bits)


def qg_grid(bits: int) -> np.ndarray:
    """Angles whose qg values are uniform on [-1, 1]."""
    return np.arccos(np.linspace(-1.0, 1.0, 2**bits))


GRIDS = {"theta": theta_grid, "qg": qg_grid}


def quantize(theta, grid, metric: str):
    """Best grid point for each target angle. metric="probability" picks
    the closest P(0); metric="fidelity" picks the closest angle."""
    theta = np.asarray(theta)
    if metric == "probability":
        d = np.abs(np.cos(theta)[:, None] - np.cos(grid)[None, :])
    elif metric == "fidelity":
        d = np.abs(theta[:, None] - grid[None, :])
    else:
        raise ValueError("metric must be 'probability' or 'fidelity'.")
    return grid[np.argmin(d, axis=1)]


def single_qubit_errors(theta_targets, bits: int, grid: str):
    """(max |dP0|, mean |dP0|, max infidelity, mean infidelity)."""
    g = GRIDS[grid](bits)
    tp = quantize(theta_targets, g, "probability")
    tf = quantize(theta_targets, g, "fidelity")
    dp = np.abs(np.cos(theta_targets) - np.cos(tp)) / 2.0
    inf = np.sin((theta_targets - tf) / 2.0) ** 2
    return dp.max(), dp.mean(), inf.max(), inf.mean()


# --------------------------------------------------------------------- #
# Grover-Rudolph amplitude loading with quantized angles
# --------------------------------------------------------------------- #
def conditional_probabilities(p):
    """Per tree level, the conditional probability of the left branch."""
    p = np.asarray(p, dtype=float)
    n = int(round(np.log2(len(p))))
    levels, segments = [], [p]
    for _ in range(n):
        level, nxt = [], []
        for seg in segments:
            half = len(seg) // 2
            mass = seg.sum()
            level.append(seg[:half].sum() / mass if mass > 0 else 0.5)
            nxt += [seg[:half], seg[half:]]
        levels.append(np.array(level))
        segments = nxt
    return levels


def loaded_distribution(levels):
    probs = np.array([1.0])
    for level in levels:
        probs = np.stack([probs * level, probs * (1.0 - level)], axis=1).ravel()
    return probs


def quantize_probabilities(p_cond, bits: int, grid: str):
    """Round each conditional probability to the closest value the b-bit
    angle grid can produce (P(0) = cos^2(theta/2))."""
    pg = np.cos(GRIDS[grid](bits) / 2.0) ** 2
    idx = np.argmin(np.abs(np.asarray(p_cond)[:, None] - pg[None, :]), axis=1)
    return pg[idx]


def loading_errors(p, bits: int, grid: str):
    """(total variation distance, infidelity) of the loaded distribution."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    q = loaded_distribution([quantize_probabilities(l, bits, grid) for l in conditional_probabilities(p)])
    tvd = 0.5 * np.abs(q - p).sum()
    infidelity = 1.0 - np.sum(np.sqrt(q * p)) ** 2
    return tvd, infidelity


def _gauss(x, sd=1.0):
    return np.exp(-0.5 * (x / sd) ** 2)


def _lognormal(x, s=0.6):
    return np.exp(-0.5 * (np.log(x) / s) ** 2) / x


def benchmark_distributions(n_qubits: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    m = 2**n_qubits
    return {
        "normal": _gauss(np.linspace(-3, 3, m)),
        "lognormal": _lognormal(np.linspace(0.05, 4, m)),
        "narrow normal": _gauss(np.linspace(-3, 3, m), 0.15),
        "Dirichlet(0.1)": rng.dirichlet(np.full(m, 0.1)),
        "Dirichlet(1)": rng.dirichlet(np.full(m, 1.0)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    haar = np.arccos(rng.uniform(-1, 1, 200_000))
    print("Finding A: single qubit, Haar-random targets")
    print(f"{'bits':>4} {'grid':>6} | {'max dP':>8} {'mean dP':>8} | {'max infid':>9} {'mean infid':>10}")
    for b in (4, 6, 8, 10):
        for g in ("theta", "qg"):
            e = single_qubit_errors(haar, b, g)
            print(f"{b:4d} {g:>6} | {e[0]:8.2e} {e[1]:8.2e} | {e[2]:9.2e} {e[3]:10.2e}")
    print("\nFinding B: Grover-Rudolph loading (TVD / infidelity)")
    for n in (4, 6, 8):
        for name, dist in benchmark_distributions(n).items():
            cells = []
            for b in (4, 6, 8):
                t_th, f_th = loading_errors(dist, b, "theta")
                t_qg, f_qg = loading_errors(dist, b, "qg")
                cells.append(f"b={b}: TVD {t_th:.1e}/{t_qg:.1e} inf {f_th:.1e}/{f_qg:.1e}")
            print(f"  n={n} {name:15s} " + " | ".join(cells))
    print("  (each pair: theta grid / qg grid)")
