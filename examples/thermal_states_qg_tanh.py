"""
Thermal states in qg units: qg_Z = tanh(beta h).

A qubit with Hamiltonian H = -h Z in equilibrium at temperature T
(k_B = 1, beta = 1/T) is the Gibbs state rho = exp(-beta H) / Z, and its
polar qang is

    qg_Z = Tr(rho Z) = tanh(beta h).

This script checks what that identity buys, in four parts. It was
listed as "conceptual only, no advantage expected", and the results
confirm that: everything below is either an exact rewriting of
textbook thermodynamics in qg, or a comparison between two ways of
running the same qg estimate. Nothing here beats a classical method.

A. Exact thermodynamics of one qubit, written in qg alone
   (checked against exp(-beta H) computed numerically):

     energy          U   = -h qg
     entropy         S   = qg_S(qg)          (Z-basis entropy = von Neumann,
                                               because rho is diagonal in Z)
     free energy     F   = -T ln(2 / sqrt(1 - qg^2))
     heat capacity   C   = (beta h)^2 (1 - qg^2) = artanh(qg)^2 (1 - qg^2)
     preparation     theta = arccos(qg) = pi/2 - gd(beta h)   (gd = Gudermannian)

   Nernst (third law): as T -> 0, qg -> 1 and qg_S, C -> 0.

B. Thermometry. A Z measurement is an energy measurement, so it is the
   optimal measurement for a Gibbs qubit; per shot the Fisher
   information on T is F_T = C / T^2, and the best relative precision
   sqrt(N) std(T)/T >= 1/sqrt(C) is reached when C is maximal:

     d C / d(beta h) = 0   <=>   qg * artanh(qg) = 1
                         =>   qg* = 0.8336, beta h = 1.1997,
                              beta * gap = 2.3994

   the peak of the Schottky anomaly (gap ~ 2.4 k_B T), i.e. the known
   optimal operating point of a two-level thermometer (Correa et al.,
   PRL 114, 220405, 2015), here as a one-line condition on qg.

   Few shots (exact enumeration over the binomial, no Monte Carlo):
   plugging the raw frequency into T = h / artanh(qg) fails whenever
   qg_hat <= 0 (T = infinity or negative) or qg_hat = 1 (T = 0). The
   posterior median of qg under the Haar prior (uniform in qg, §15.3),
   restricted to qg > 0 (positive temperature), maps to the posterior
   median of T exactly, because T(qg) is monotone. Jeffreys' prior with
   the same restriction is the standard control.

C. Circuits and hardware. A Gibbs qubit is half of a 2-qubit pure state:
   Ry(theta) on q0, CX q0 -> q1, with cos(theta) = tanh(beta h). And the
   residual excited-state population p1 of an idle qubit is a
   temperature: qg = 1 - 2 p1 gives T_eff = h f / (2 k_B artanh(qg)).

D. Where the tanh law stops being exact: an Ising ring of 6 qubits.
   With only ZZ couplings (classical Ising, rho diagonal in Z) the
   thermodynamic entropy is exactly  sum_i qg_S_i - qg_correlation,
   the qg quantities of §7.2. A transverse field (coherence) breaks the
   identity: the Z-basis quantities only give an upper bound.

Findings (h = 1, k_B = 1; run with --figure for the plot):

  A. All identities hold to 5e-12 over beta h in [0.01, 8].
  B. Few-shot thermometry. Failure = T_hat is 0 or infinite; error =
     median |T_hat/T - 1| (failures count as infinite error), exact
     over the binomial distribution of k0:

      true qg        shots | failure: raw  Haar>0  Jeffr>0 | error: raw  Haar>0  Jeffr>0 | CR std/T
      0.8336 (opt.)    10  |          0.42  0       0      |        0.73  0.37    0.37    |  0.48
                       30  |          0.07  0       0      |        0.28  0.19    0.25    |  0.28
                      100  |          0     0       0      |        0.09  0.10    0.10    |  0.15
      0.99             10  |          0.95  0       0      |        inf   0.94    0.39    |  0.85
                       30  |          0.86  0       0      |        inf   0.40    0.08    |  0.49
                      100  |          0.61  0       0      |        inf   0.06    0.13    |  0.27

     Near the ground state the plug-in estimate says T = 0 in 61% of
     runs even with 100 shots; the restricted posteriors never fail.
     Neither prior wins everywhere (Jeffreys is better at 10-30 shots
     near qg = 1, Haar at 30 shots near qg*): the prior and the
     positivity restriction do the work, not qg. By 100 shots at qg*
     all three agree. (A median absolute error can sit below the
     Cramer-Rao bound, which is on the standard deviation.)
  C. The purification circuit reproduces tanh(beta h) within shot
     noise (20,000 shots: 0.2022 vs 0.1974, 0.7642 vs 0.7616, 0.9958
     vs 0.9951). A 5 GHz qubit with 1% residual excited population has
     qg_Z = 0.98 and T_eff = 52 mK.
  D. Classical Ising ring (J = h = 1): sum qg_S - qg_correlation equals
     the Gibbs entropy to 1e-13 at every temperature. The mean-field
     law qg = tanh(beta (h + 2 J qg)) overshoots the exact per-site qg
     by up to 0.095 (at beta = 0.38); the free-spin tanh(beta h) is far
     below both. With a transverse field g the Z-basis entropy
     overestimates the true one, and the gap grows with g and as T
     falls: at beta = 1.5, 0.36 bits for g = 0.5 and 2.1 bits for
     g = 1.5, while the true entropy is ~0.01 bits.

Honest summary: qg = tanh(beta h) is a clean coordinate for a thermal
qubit: every thermodynamic quantity and the optimal-thermometer condition
are one-line functions of it, and for diagonal (classical) Gibbs states
the qg entropy decomposition is exact. It is not a quantum advantage and
the few-shot gain comes from the prior, as in §15.3.
"""

import math
import sys

import numpy as np

from qang.core import qg_s_from_qg_z
from qang.multiqubit import qg_correlation, per_qubit_qg_z
from qang.statistics import beta_cdf, beta_ppf

H_FIELD = 1.0  # h, in units of k_B * (temperature unit)

# --------------------------------------------------------------------- #
# A. single-qubit thermodynamics in qg
# --------------------------------------------------------------------- #
Z = np.diag([1.0, -1.0])
X = np.array([[0.0, 1.0], [1.0, 0.0]])


def gibbs(hamiltonian: np.ndarray, beta: float) -> np.ndarray:
    """exp(-beta H) / Tr(...) via eigendecomposition (H Hermitian)."""
    w, v = np.linalg.eigh(hamiltonian)
    e = np.exp(-beta * (w - w.min()))
    return (v * (e / e.sum())) @ v.conj().T


def von_neumann_bits(rho: np.ndarray) -> float:
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-15]
    return float(-(w * np.log2(w)).sum())


def qg_thermal(beta: float, h: float = H_FIELD) -> float:
    return math.tanh(beta * h)


def beta_from_qg(qg: float, h: float = H_FIELD) -> float:
    return math.atanh(qg) / h


def energy_from_qg(qg: float, h: float = H_FIELD) -> float:
    return -h * qg


def entropy_from_qg(qg: float) -> float:
    """Thermodynamic (von Neumann) entropy in bits: exactly qg_S."""
    return qg_s_from_qg_z(qg)


def free_energy_from_qg(qg: float, h: float = H_FIELD) -> float:
    return -math.log(2.0 / math.sqrt(1.0 - qg * qg)) / beta_from_qg(qg, h)


def heat_capacity_from_qg(qg: float) -> float:
    """C / k_B = (beta h)^2 (1 - qg^2); independent of h once written in qg."""
    return math.atanh(qg) ** 2 * (1.0 - qg * qg)


def gudermannian(x: float) -> float:
    return math.atan(math.sinh(x))


def preparation_angle(beta: float, h: float = H_FIELD) -> float:
    """Ry angle whose reduced state (after the purifying CX) is the Gibbs state."""
    return math.pi / 2.0 - gudermannian(beta * h)


def check_identities(betas=np.linspace(0.01, 8.0, 200), h=H_FIELD) -> float:
    """Largest deviation between the qg formulas and direct numerics."""
    worst = 0.0
    ham = -h * Z
    for b in betas:
        rho = gibbs(ham, b)
        qg = float(np.real(np.trace(rho @ Z)))
        z_part = 2.0 * math.cosh(b * h)
        u = float(np.real(np.trace(rho @ ham)))
        var_e = float(np.real(np.trace(rho @ ham @ ham))) - u * u
        exact = {
            "qg": math.tanh(b * h),
            "U": u,
            "S": von_neumann_bits(rho),
            "F": -math.log(z_part) / b,
            "C": b * b * var_e,
            "theta": math.acos(qg),
        }
        mine = {
            "qg": qg_thermal(b, h),
            "U": energy_from_qg(qg, h),
            "S": entropy_from_qg(qg),
            "F": free_energy_from_qg(qg, h),
            "C": heat_capacity_from_qg(qg),
            "theta": preparation_angle(b, h),
        }
        for k in exact:
            worst = max(worst, abs(exact[k] - mine[k]))
    return worst


# --------------------------------------------------------------------- #
# B. thermometry
# --------------------------------------------------------------------- #
def optimal_probe_qg(tol: float = 1e-14) -> float:
    """Root of qg * artanh(qg) = 1 on (0, 1): maximal heat capacity."""
    lo, hi = 0.5, 1.0 - 1e-15
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if mid * math.atanh(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def relative_cramer_rao(qg: float, n_shots: int) -> float:
    """Lower bound on std(T_hat)/T from N Z-shots on a Gibbs qubit."""
    return 1.0 / math.sqrt(n_shots * heat_capacity_from_qg(qg))


def _t_from_qg(qg: float, h: float = H_FIELD) -> float:
    if qg >= 1.0:
        return 0.0
    if qg <= 0.0:
        return math.inf
    return h / math.atanh(qg)


def estimate_temperature(k0: int, n_shots: int, method: str, h: float = H_FIELD) -> float:
    """T_hat from k0 outcomes '0' in n_shots Z-shots.

    raw       plug-in frequency qg = 2 k0/N - 1  (may give 0 or infinity)
    haar      posterior median, uniform prior on qg (= Haar), restricted to qg > 0
    jeffreys  posterior median, Jeffreys prior Beta(1/2, 1/2), restricted to qg > 0
    """
    if method == "raw":
        return _t_from_qg(2.0 * k0 / n_shots - 1.0, h)
    a0 = {"haar": 1.0, "jeffreys": 0.5}[method]
    a, b = k0 + a0, n_shots - k0 + a0
    # Median of the posterior restricted to p0 > 1/2 (qg > 0). Written with
    # the mirrored Beta(b, a) for 1 - p0, so the tail mass never underflows
    # to 1 - 1: P(1 - p0 < 1 - p_med) = P(1 - p0 < 1/2) / 2.
    tail = beta_cdf(0.5, b, a)
    p_med = 1.0 - beta_ppf(0.5 * tail, b, a)
    p_med = min(max(p_med, 0.5 + 1e-12), 1.0 - 1e-15)
    return _t_from_qg(2.0 * p_med - 1.0, h)


def thermometry_table(qg_true: float, n_shots: int, methods=("raw", "haar", "jeffreys")) -> dict:
    """Exact (binomial-enumerated) failure rate and median relative error."""
    p0 = 0.5 * (1.0 + qg_true)
    t_true = _t_from_qg(qg_true)
    ks = np.arange(n_shots + 1)
    log_pmf = np.array([math.lgamma(n_shots + 1) - math.lgamma(k + 1) - math.lgamma(n_shots - k + 1)
                        for k in ks])
    with np.errstate(divide="ignore"):
        log_pmf = log_pmf + ks * np.log(p0) + (n_shots - ks) * np.log1p(-p0)
    pmf = np.exp(log_pmf)
    pmf /= pmf.sum()
    out = {}
    for m in methods:
        t_hat = np.array([estimate_temperature(int(k), n_shots, m) for k in ks])
        fail = (t_hat == 0.0) | ~np.isfinite(t_hat)
        rel = np.where(fail, np.inf, np.abs(t_hat / t_true - 1.0))
        order = np.argsort(rel)
        cdf = np.cumsum(pmf[order])
        median = float(rel[order][np.searchsorted(cdf, 0.5)])
        out[m] = {"failure": float(pmf[fail].sum()), "median_rel_error": median}
    out["cramer_rao"] = relative_cramer_rao(qg_true, n_shots)
    return out


# --------------------------------------------------------------------- #
# C. circuits and hardware
# --------------------------------------------------------------------- #
def thermal_purification_circuit(beta: float, h: float = H_FIELD):
    """2-qubit pure state whose q0 marginal is the Gibbs state of -h Z."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 1)
    qc.ry(preparation_angle(beta, h), 0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    return qc


def sampled_qg(beta: float, shots: int = 20000, seed: int = 7, h: float = H_FIELD) -> float:
    from qiskit_aer import AerSimulator

    counts = AerSimulator(seed_simulator=seed).run(thermal_purification_circuit(beta, h),
                                                   shots=shots).result().get_counts()
    return (counts.get("0", 0) - counts.get("1", 0)) / shots


PLANCK_OVER_KB_K_PER_GHZ = 6.62607015e-34 * 1e9 / 1.380649e-23  # h*f/k_B in kelvin for f = 1 GHz


def effective_temperature_mK(qg: float, f_ghz: float) -> float:
    """Qubit temperature from its idle qg_Z, for a transition at f_ghz.
    Here the gap is h f = 2 h_field, so T = h f / (2 k_B artanh(qg))."""
    if qg >= 1.0:
        return 0.0
    if qg <= 0.0:
        return math.inf
    return 1e3 * PLANCK_OVER_KB_K_PER_GHZ * f_ghz / (2.0 * math.atanh(qg))


# --------------------------------------------------------------------- #
# D. Ising ring: where tanh and the qg entropy decomposition stop being exact
# --------------------------------------------------------------------- #
def _site_op(op: np.ndarray, i: int, n: int) -> np.ndarray:
    out = np.array([[1.0]])
    for k in range(n):
        out = np.kron(out, op if k == i else np.eye(2))
    return out


def ising_ring(n: int, j: float, h: float, g: float = 0.0) -> np.ndarray:
    """H = -J sum Z_i Z_{i+1} - h sum Z_i - g sum X_i on a ring."""
    zs = [_site_op(Z, i, n) for i in range(n)]
    ham = -h * sum(zs) - j * sum(zs[i] @ zs[(i + 1) % n] for i in range(n))
    if g:
        ham = ham - g * sum(_site_op(X, i, n) for i in range(n))
    return ham


def mean_field_qg(beta: float, j: float, h: float, coordination: int = 2) -> float:
    """Curie-Weiss self-consistency qg = tanh(beta (h + z J qg)), by iteration."""
    qg = 1.0
    for _ in range(10000):
        new = math.tanh(beta * (h + coordination * j * qg))
        if abs(new - qg) < 1e-14:
            break
        qg = new
    return qg


def ising_row(beta: float, n: int = 6, j: float = 1.0, h: float = H_FIELD, g: float = 0.0) -> dict:
    rho = gibbs(ising_ring(n, j, h, g), beta)
    qg_site = float(np.mean(per_qubit_qg_z(rho, n)))
    s_true = von_neumann_bits(rho)
    s_qg = n * qg_s_from_qg_z(qg_site) - qg_correlation(rho, n)
    return {
        "qg_exact": qg_site,
        "qg_free": math.tanh(beta * h),
        "qg_mean_field": mean_field_qg(beta, j, h),
        "S_true": s_true,
        "S_qg": s_qg,
        "gap_bits": s_qg - s_true,
    }


# --------------------------------------------------------------------- #
def make_figure(path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    q = np.linspace(0.001, 0.999, 400)
    c = np.array([heat_capacity_from_qg(x) for x in q])
    s = np.array([entropy_from_qg(x) for x in q])
    qs = optimal_probe_qg()
    ax[0].plot(q, s, color="#1f6fb2", label="entropy S = qg_S (bits)")
    ax[0].plot(q, c, color="#c0392b", label="heat capacity C/k_B")
    ax[0].axvline(qs, color="k", ls=":", lw=1)
    ax[0].text(qs - 0.02, 0.9, f"optimal thermometer\nqg·artanh(qg)=1\nqg*={qs:.4f}", ha="right", fontsize=8)
    ax[0].set_xlabel("qg_Z = tanh(beta h)   (T = infinity  ->  T = 0)")
    ax[0].set_title("A/B. one qubit: thermodynamics in qg")
    ax[0].legend(fontsize=8, loc="center left")

    shots = [5, 10, 20, 30, 50, 100, 200, 500]
    colors = {"raw": "#8c8c8c", "haar": "#1f6fb2", "jeffreys": "#e0a030"}
    clip = 1.5
    for qg_true, ls in ((qs, "-"), (0.99, "--")):
        rows = [thermometry_table(qg_true, n) for n in shots]
        for m in ("raw", "haar", "jeffreys"):
            ax[1].plot(shots, [min(r[m]["median_rel_error"], clip) for r in rows], ls, color=colors[m],
                       marker="o", ms=3, label=f"{m}, qg={qg_true:.3g}")
        ax[1].plot(shots, [r["cramer_rao"] for r in rows], ls, color="k", lw=0.8,
                   label=f"Cramer-Rao std/T, qg={qg_true:.3g}")
    ax[1].axhline(clip, color="#8c8c8c", lw=0.5)
    ax[1].text(5, clip * 0.93, "raw: T = 0 in most runs", fontsize=7, va="top")
    ax[1].set_xscale("log")
    ax[1].set_ylim(0, clip * 1.05)
    ax[1].set_xlabel("shots")
    ax[1].set_ylabel("median |T_hat / T - 1|")
    ax[1].set_title("B. few-shot thermometry (exact binomial)")
    ax[1].legend(fontsize=6.5, ncol=2)

    betas = np.linspace(0.05, 2.5, 40)
    rows = [ising_row(b) for b in betas]
    ax[2].plot(betas, [r["qg_exact"] for r in rows], color="k", label="exact (6-ring, J=h=1)")
    ax[2].plot(betas, [r["qg_mean_field"] for r in rows], "--", color="#c0392b", label="mean field tanh")
    ax[2].plot(betas, [r["qg_free"] for r in rows], ":", color="#8c8c8c", label="free spin tanh(beta h)")
    for g, col in ((0.5, "#2e8b57"), (1.5, "#1f6fb2")):
        gap = [ising_row(b, g=g)["gap_bits"] for b in betas]
        ax[2].plot(betas, np.array(gap) / 6, color=col, lw=1, label=f"entropy gap / n, g={g}")
    ax[2].set_xlabel("beta")
    ax[2].set_title("D. 6-qubit Ising ring: qg per site, entropy gap")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    print(f"A. max deviation of the qg identities: {check_identities():.2e}")
    qs = optimal_probe_qg()
    print(f"B. optimal probe qg* = {qs:.4f}, beta h = {math.atanh(qs):.4f}, beta*gap = {2 * math.atanh(qs):.4f}")
    for qg_true in (qs, 0.99):
        print(f"   true qg = {qg_true:.4f}")
        for n in (10, 30, 100):
            r = thermometry_table(qg_true, n)
            print(f"   N={n:4d} fail " + " ".join(f"{m}={r[m]['failure']:.3f}" for m in ("raw", "haar", "jeffreys"))
                  + " | median rel err " + " ".join(f"{m}={r[m]['median_rel_error']:.3f}"
                                                   for m in ("raw", "haar", "jeffreys"))
                  + f" | CR bound {r['cramer_rao']:.3f}")
    try:
        for b in (0.2, 1.0, 3.0):
            print(f"C. beta={b}: tanh={math.tanh(b):.4f} sampled={sampled_qg(b):.4f}")
    except ImportError:
        print("C. (qiskit-aer not installed: skipping the circuit check)")
    print(f"C. 5 GHz qubit, p1 = 1%: qg = 0.98, T_eff = {effective_temperature_mK(0.98, 5.0):.1f} mK")
    for g in (0.0, 0.5, 1.5):
        for b in (0.3, 0.6, 1.5):
            r = ising_row(b, g=g)
            print(f"D. g={g} beta={b}: qg exact {r['qg_exact']:.4f} mean-field {r['qg_mean_field']:.4f} "
                  f"free {r['qg_free']:.4f} | S true {r['S_true']:.4f} S_qg {r['S_qg']:.4f} gap {r['gap_bits']:+.2e}")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"))
