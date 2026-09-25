"""
Is the §25 entropy gap a witness of quantum coherence? Thermal states of
the transverse-field Ising ring, exact and with finite shots.

§25 found that for a 6-qubit Ising ring (H = -J sum Z_i Z_{i+1} - h sum Z_i
- g sum X_i, J = h = 1) the Z-basis entropy written in qg units,
Sum_i qg_S,i - qg_correlation = H(Z-basis outcomes), equals the
thermodynamic entropy S(rho) exactly when g = 0 and exceeds it when
g > 0. That gap is a known quantity: it is the relative entropy of
coherence in the Z basis [Baumgratz, Cramer, Plenio, PRL 113, 140401
(2014)],

    C(rho) = H(diag rho) - S(rho) >= 0,  = 0 iff rho is diagonal in Z.

So the gap is an exact coherence measure, but it needs S(rho), which Z
counts cannot give. Two lower bounds that ARE measurable:

  * qubit bound (qg): partial trace is an incoherent operation, so
    C(rho) >= C(rho_i) for every qubit. For one qubit with Bloch vector
    (qg_X, qg_Y, qg_Z), |r| = sqrt(qg_X^2 + qg_Y^2 + qg_Z^2) and
        C(rho_i) = H((1 + qg_Z)/2) - H((1 + |r|)/2) = qg_S(qg_Z) - qg_S(|r|).
    It needs only the single-qubit qg in two bases (X and Z here; Y = 0
    for this real Hamiltonian). With translation invariance all sites
    give the same value and can be averaged. Also: qg_X != 0 alone
    certifies coherence, because a diagonal state has <X_i> = 0.
  * sum bound (qg): C is superadditive over qubits,
        C(rho) - Sum_i C(rho_i) = T(rho) - qg_correlation >= 0,
    where T is the quantum total correlation and qg_correlation the
    Z-outcome total correlation (local dephasing cannot increase total
    correlation). So C(rho) >= Sum_i [qg_S(qg_Z,i) - qg_S(|r_i|)], still
    from single-qubit qg in two bases.
  * basis bound: S(rho) <= H(outcomes in ANY basis), so
        C(rho) >= H_Z - H_X,
    the difference of the joint Shannon entropies of the Z- and X-basis
    counts (2^n outcomes each).

Findings (6-qubit ring, J = h = 1):

  * The sum bound from single-qubit qg captures 92-100% of the exact
    Z-basis coherence over beta in [0.25, 3], g in [0.25, 2] (e.g.
    0.354 of 0.356 bits at beta = 1.5, g = 0.5; 2.94 of 3.10 at
    beta = 3, g = 2). The remainder is T - qg_correlation, the part of
    the correlations that the Z basis does not see.
  * The single-qubit bound alone gets only ~16% (it is the sum bound
    divided by n here, by translation invariance).
  * The basis bound H_Z - H_X is useless for these states: it is
    negative almost everywhere (down to -6 bits) because the X-basis
    outcomes are nearly uniform; it turns positive only for a strong
    field (g > ~2.1, e.g. +0.13 of 0.89 at beta = 0.25, g = 2).
  * Finite shots: the qg bounds need few shots (sum bound 0.26 ± 0.11
    at 100 shots, 0.32 ± 0.04 at 1000, exact 0.354, beta = 1.5,
    g = 0.5; small downward bias at 100 shots). At g = 0 (no coherence)
    they read ~0.001 bits and the |qg_X| test at 3 sigma fires in
    0.3-0.5% of runs, the nominal rate.
  * Where it fails: for a graph (cluster) state every reduced qubit is
    I/2, so the qg bounds are 0 while C = n bits -- the same Z-basis
    blind spot as in the first preprint. The tightness above is a
    property of these thermal states, not a general law.

The superadditivity of the relative entropy of coherence is known in
the coherence literature; what qg adds is that the bound is two
single-qubit qg values per site, with the §25 decomposition naming the
missing term.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from thermal_states_qg_tanh import _site_op, gibbs, ising_ring, von_neumann_bits  # noqa: E402

N_SITES = 6
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.diag([1.0, -1.0])
HAD = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2)
BETAS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
GS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)


def shannon_bits(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def binary_entropy(p):
    return shannon_bits([p, 1 - p])


def _hadamard_all(n):
    out = np.array([[1.0]])
    for _ in range(n):
        out = np.kron(out, HAD)
    return out


def thermal_state(beta, g, n=N_SITES, j=1.0, h=1.0):
    return gibbs(ising_ring(n, j, h, g), beta)


def exact_coherence(rho):
    """Relative entropy of coherence in Z = the §25 entropy gap."""
    return shannon_bits(np.real(np.diag(rho))) - von_neumann_bits(rho)


def bloch_site0(rho, n=N_SITES):
    return (float(np.real(np.trace(_site_op(X, 0, n) @ rho))), float(np.real(np.trace(_site_op(Z, 0, n) @ rho))))


def qubit_bound_from_qg(qg_x, qg_z, qg_y=0.0):
    r = min(math.sqrt(qg_x**2 + qg_y**2 + qg_z**2), 1.0)
    return binary_entropy((1 + qg_z) / 2) - binary_entropy((1 + r) / 2)


def z_and_x_distributions(rho, n=N_SITES):
    hh = _hadamard_all(n)
    return np.clip(np.real(np.diag(rho)), 0, None), np.clip(np.real(np.diag(hh @ rho @ hh)), 0, None)


def exact_row(beta, g):
    rho = thermal_state(beta, g)
    pz, px = z_and_x_distributions(rho)
    qx, qz = bloch_site0(rho)
    return {"C": exact_coherence(rho), "qubit": qubit_bound_from_qg(qx, qz),
            "sum": N_SITES * qubit_bound_from_qg(qx, qz),
            "basis": shannon_bits(pz) - shannon_bits(px), "qg_x": qx, "qg_z": qz, "S": von_neumann_bits(rho)}


def graph_state_counterexample(n=N_SITES):
    """Ring cluster state: (exact C, qg sum bound) = (n, 0)."""
    psi = np.ones(2**n) / math.sqrt(2**n)
    idx = np.arange(2**n)
    for i in range(n):
        j = (i + 1) % n
        psi = psi * np.where(((idx >> i) & 1) & ((idx >> j) & 1), -1.0, 1.0)
    rho = np.outer(psi, psi)
    qx, qz = bloch_site0(rho, n)
    return exact_coherence(rho), n * qubit_bound_from_qg(qx, qz)


# --------------------------------------------------------------------- #
# finite shots
# --------------------------------------------------------------------- #
def miller_madow(counts):
    n = counts.sum()
    k = int((counts > 0).sum())
    return shannon_bits(counts / n) + (k - 1) / (2 * n * math.log(2))


def sampled_bounds(beta, g, shots, rng):
    """Both bounds from `shots` Z-basis and `shots` X-basis measurements."""
    rho = thermal_state(beta, g)
    pz, px = z_and_x_distributions(rho)
    cz = rng.multinomial(shots, pz / pz.sum())
    cx = rng.multinomial(shots, px / px.sum())
    bits = (np.arange(2**N_SITES)[:, None] >> np.arange(N_SITES)) & 1
    sign = 1 - 2 * bits  # per-site +-1
    qz = float((cz @ sign).mean() / shots)  # site-averaged qg_Z
    qx = float((cx @ sign).mean() / shots)  # site-averaged qg_X
    return {"qubit": qubit_bound_from_qg(qx, qz), "sum": N_SITES * qubit_bound_from_qg(qx, qz), "basis_plugin": shannon_bits(cz / shots) - shannon_bits(cx / shots),
            "basis_mm": miller_madow(cz) - miller_madow(cx), "qg_x": qx}


def shot_table(shots_list=(100, 1000, 10000), cases=((1.5, 0.5), (1.5, 1.5), (0.5, 1.0)), reps=200, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for beta, g in cases:
        ex = exact_row(beta, g)
        for shots in shots_list:
            rows = [sampled_bounds(beta, g, shots, rng) for _ in range(reps)]
            out[(beta, g, shots)] = {
                "exact": ex,
                **{k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
                   for k in ("qubit", "sum", "basis_plugin", "basis_mm")},
            }
    return out


def false_positive_rate(shots, beta=1.5, reps=400, seed=1, z_sigma=3.0):
    """g = 0 (no coherence): how often does each estimated bound exceed
    z_sigma times its own shot-noise standard error? The qubit bound is
    tested through |qg_X| (standard error sqrt(1/(n shots))); the basis
    bound through its bootstrap-free delta-method error ~ sqrt(2 Var/shots)."""
    rng = np.random.default_rng(seed)
    fp_qx, fp_basis = 0, 0
    rho = thermal_state(beta, 0.0)
    pz, px = z_and_x_distributions(rho)
    var_z = float((pz * np.log2(np.clip(pz, 1e-300, None)) ** 2).sum() - shannon_bits(pz) ** 2)
    var_x = float((px * np.log2(np.clip(px, 1e-300, None)) ** 2).sum() - shannon_bits(px) ** 2)
    se_basis = math.sqrt((var_z + var_x) / shots)
    for _ in range(reps):
        r = sampled_bounds(beta, 0.0, shots, rng)
        if abs(r["qg_x"]) > z_sigma / math.sqrt(N_SITES * shots):
            fp_qx += 1
        if r["basis_mm"] > z_sigma * se_basis:
            fp_basis += 1
    return fp_qx / reps, fp_basis / reps


def make_figure(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gs = np.linspace(0, 2.5, 26)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    for ax, beta in zip(axes, (0.5, 1.5, 3.0)):
        rows = [exact_row(beta, g) for g in gs]
        ax.plot(gs, [r["C"] for r in rows], "k-", lw=2, label="exact coherence = §25 entropy gap")
        ax.plot(gs, [r["sum"] for r in rows], color="#1f6fb2", lw=2, ls="--", label="sum of qubit bounds (qg_X, qg_Z)")
        ax.plot(gs, [r["qubit"] for r in rows], color="#9ecae1", lw=2, label="single-qubit bound")
        ax.plot(gs, [max(r["basis"], 0) for r in rows], color="#e0a030", lw=2, label="basis bound H_Z - H_X")
        ax.set_title(f"beta = {beta}", fontsize=9)
        ax.set_xlabel("transverse field g")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("coherence in Z (bits)")
    axes[0].legend(fontsize=7)
    fig.suptitle("6-qubit Ising ring, J = h = 1: Z-basis coherence and its measurable lower bounds", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    print("Exact: C (= §25 gap) / qg sum bound / basis bound, bits")
    print("beta\\g " + "".join(f"{g:>20}" for g in GS))
    for beta in BETAS:
        cells = []
        for g in GS:
            r = exact_row(beta, g)
            cells.append(f"{r['C']:.3f}/{r['sum']:.3f}/{r['basis']:+.2f}")
        print(f"{beta:5} " + "".join(f"{c:>20}" for c in cells))
    print("\nFinite shots (mean ± std over 200 repetitions):")
    for (beta, g, shots), r in shot_table().items():
        print(f"  beta {beta}, g {g}, {shots:5d} shots: exact C {r['exact']['C']:.3f}; sum bound "
              f"{r['sum'][0]:.3f}±{r['sum'][1]:.3f} (exact {r['exact']['sum']:.3f}); basis bound plug-in "
              f"{r['basis_plugin'][0]:+.3f}±{r['basis_plugin'][1]:.3f}, Miller-Madow {r['basis_mm'][0]:+.3f}"
              f"±{r['basis_mm'][1]:.3f} (exact {r['exact']['basis']:+.3f})")
    print("\nFalse positives at g = 0 (3 sigma), beta = 1.5:")
    for shots in (100, 1000, 10000):
        a, b = false_positive_rate(shots)
        print(f"  {shots:5d} shots: |qg_X| test {a:.3f}, basis bound {b:.3f}")
    c, b = graph_state_counterexample()
    print(f"\nRing cluster state: exact C {c:.3f} bits, qg sum bound {b:.3f}")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"))
