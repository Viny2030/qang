"""
The I-eta plane of RBM neural-network quantum states, written in qg.

The review by Singh, Bhatia, Saggi, Sajjan and Kais (Academia Quantum 3,
2026, Sec. 3.2 and Fig. 4) studies an RBM learner through its Ising-type
Hamiltonian H(X) = sum a_i s_i + sum b_j h_j + sum W_ij s_i h_j and the
thermal state P(v, h) ~ exp(-H). For each visible-hidden pair it reports

    eta = Cov(s_k, h_m) = <s_k h_m> - qg_v qg_h        (read from the OTOC)
    I   = mutual information between s_k and h_m,

and shows that (I, eta) lies between

    LB(eta) = 2 - 2 l((1+eta)/4) - 2 l((1-eta)/4),
    UB(eta) = l(1/2 + sqrt(1-|eta|)/2) + l(1/2 - sqrt(1-|eta|)/2),  l(x) = -x log2 x.

Trained RBMs (transverse-field Ising drivers) sit on LB for every size and
field ratio g, which the review reads as a learning principle: the network
uses the least mutual information compatible with the covariance.

qg reading. A pair of +-1 spins is fixed by three numbers, the two
marginals qg_v = <s_k>, qg_h = <h_m> and eta:

    p(s, t) = [1 + s qg_v + t qg_h + s t (eta + qg_v qg_h)] / 4,
    I = qg_S(qg_v) + qg_S(qg_h) - H(p),

with qg_S the binary entropy of (1 + qg)/2 (§7). Then

  * LB(eta) is exactly the case qg_v = qg_h = 0 and is the minimum of I
    over the marginals at fixed eta;
  * UB(eta) is the perfectly correlated pair with |qg_v| = |qg_h| =
    sqrt(1 - |eta|);
  * to leading order in eta and the marginals, I - LB ~= eta^2 (qg_v^2 +
    qg_h^2) / (2 ln 2) (5 % low at eta = 0.1, 12 % at eta = 0.3).

The transverse-field Ising driver H = -B sum X_i - J sum Z_i Z_{i+1} has a
Z2 symmetry (Z -> -Z), so its ground state has <Z_i> = 0 and an RBM that
respects it has a = b = 0 and zero marginals. Saturation of LB is then a
consequence of the symmetry, not a learning principle. Prediction, which
can fail: add a longitudinal field -h_z sum Z_i; the points must leave LB
by the amount set by the measured marginals.

Test. RBM wavefunction psi(v) = exp(a.v) prod_j 2 cosh(b_j + W_j.v)
(Carleo-Troyer, real and positive, enough for this stoquastic driver),
n = 6 visible spins, alpha = 1, periodic chain, energy minimised exactly
(full enumeration, no sampling), 3 seeds per setting.

Findings (python examples/rbm_mutual_information_qg.py):

  Closed forms (checked on a grid of marginals): at fixed eta the minimum
  of I over (qg_v, qg_h) is LB(eta), reached at qg_v = qg_h = 0; UB(eta) is
  I at |qg_v| = |qg_h| = sqrt(1 - eta) (eta = 0.1, 0.3, 0.5, 0.8, equal to
  1e-4).

  Trained RBMs (median over 3 seeds; 36 visible-hidden pairs each):

       g   h_z | fidelity | |qg_v|  |qg_h|  |eta| | mean I-LB  max I-LB
     1.0   0   |  1.0000  | 0.0001  0.0016  0.27  |  3e-7      1e-5
     2.0   0   |  1.0000  | 0.0000  0.0034  0.15  |  7e-7      2e-5
     1.0   0.1 |  1.0000  | 0.39    0.45    0.16  |  1.6e-2    0.10
     1.0   0.3 |  1.0000  | 0.52    0.42    0.12  |  1.4e-2    0.17
     2.0   0.1 |  1.0000  | 0.07    0.09    0.15  |  5.2e-4    5e-3
     2.0   0.3 |  1.0000  | 0.18    0.19    0.13  |  2.7e-3    3e-2
     0.5   0   |  0.50    | 0.81    0.77    0.05  |  1.1e-2    0.12   (random start)
     0.5   0   |  1.0000  | 0.0002  0.0001  0.49  |  1.5e-8    1e-7   (a = b = 0 start)

  * With the Z2-symmetric driver the trained RBMs sit on LB (gap below
    2e-5 bits) because their marginals vanish, as the review found.
  * The prediction holds: a longitudinal field of 0.1-0.3 moves the same
    learner off LB by 5e-4 to 2e-2 bits on average (up to 0.17 bits for a
    pair), in step with the measured marginals; the small-bias formula
    eta^2 (qg_v^2 + qg_h^2)/(2 ln 2) matches the gap at g = 2 (5.3e-4 vs
    5.2e-4, 2.6e-3 vs 2.7e-3) and underestimates it by up to 2x when the
    marginals are large.
  * In the ordered phase (g = 0.5) the same target admits two RBMs: from a
    random start training breaks the symmetry (fidelity 0.50 with the
    symmetric ground state, energy 3e-3 above it, marginals 0.8) and the
    points leave LB; from a = b = 0 it finds the symmetric ground state
    (fidelity 1.0000) and the points sit on LB. Same physics, different
    position: the position is set by the marginals, not by a learning
    principle. (One of three random starts ends in a fully polarized local
    minimum, 0.38 above the ground energy.)

Honest scope. A real, positive RBM with n = 6 and exact enumeration; the
review uses larger N, stochastic reconfiguration and Monte Carlo sampling,
which we have not reproduced. The claim is limited to this: LB saturation
is what zero marginals imply, and a symmetry-breaking field or a
symmetry-broken learner removes it by the amount the marginals predict.
"""

import itertools
import math

import numpy as np
from scipy.optimize import minimize

N_VIS = 6
ALPHA = 1
SEEDS = (0, 1, 2)
G_VALUES = (0.5, 1.0, 2.0)
HZ_VALUES = (0.0, 0.1, 0.3)


# --------------------------------------------------------------------- #
# closed forms
# --------------------------------------------------------------------- #
def ell(x):
    return 0.0 if x <= 0 else -x * math.log2(x)


def qg_s(q):
    return ell((1 + q) / 2) + ell((1 - q) / 2)


def lower_bound(eta):
    return 2 - 2 * ell((1 + eta) / 4) - 2 * ell((1 - eta) / 4)


def upper_bound(eta):
    r = math.sqrt(1 - abs(eta))
    return ell(0.5 + r / 2) + ell(0.5 - r / 2)


def pair_distribution(qg_v, qg_h, eta):
    c = eta + qg_v * qg_h
    return np.array([(1 + s * qg_v + t * qg_h + s * t * c) / 4 for s in (1, -1) for t in (1, -1)])


def mutual_information(qg_v, qg_h, eta):
    p = pair_distribution(qg_v, qg_h, eta)
    if p.min() < -1e-12:
        raise ValueError("not a valid pair of spins")
    return qg_s(qg_v) + qg_s(qg_h) - sum(ell(x) for x in p)


def small_bias_gap(qg_v, qg_h, eta):
    return eta**2 * (qg_v**2 + qg_h**2) / (2 * math.log(2))


# --------------------------------------------------------------------- #
# driver and RBM
# --------------------------------------------------------------------- #
_V = np.array(list(itertools.product((1, -1), repeat=N_VIS)), dtype=float)  # (2^n, n), s = +1 is |0>


def driver_matrix(g, hz, n=N_VIS, J=1.0):
    """-B sum X - J sum ZZ (periodic) - hz sum Z, with B = g J."""
    dim = 2**n
    v = _V if n == N_VIS else np.array(list(itertools.product((1, -1), repeat=n)), dtype=float)
    diag = -J * np.sum(v * np.roll(v, -1, axis=1), axis=1) - hz * v.sum(axis=1)
    H = np.diag(diag)
    for i in range(n):
        flip = np.arange(dim) ^ (1 << (n - 1 - i))
        H[np.arange(dim), flip] += -g * J
    return H


def unpack(x, n=N_VIS, m=ALPHA * N_VIS):
    return x[:n], x[n:n + m], x[n + m:].reshape(n, m)


def log_psi(x):
    a, b, W = unpack(x)
    return _V @ a + np.sum(np.log(2 * np.cosh(b + _V @ W)), axis=1)


def energy(x, H):
    lp = log_psi(x)
    psi = np.exp(lp - lp.max())
    return float(psi @ H @ psi / (psi @ psi))


def train(g, hz, seed, init_scale=0.05, symmetric=False):
    """symmetric=True starts from a = b = 0 (a Z2-symmetric RBM); with h_z = 0
    the gradient in a, b vanishes there by symmetry, so it stays symmetric."""
    H = driver_matrix(g, hz)
    rng = np.random.default_rng(seed)
    m = ALPHA * N_VIS
    x0 = init_scale * rng.standard_normal(N_VIS + m + N_VIS * m)
    if symmetric:
        x0[:N_VIS + m] = 0.0
    res = minimize(energy, x0, args=(H,), method="L-BFGS-B", options={"maxiter": 3000})
    w, vecs = np.linalg.eigh(H)
    lp = log_psi(res.x)
    psi = np.exp(lp - lp.max())
    psi /= np.linalg.norm(psi)
    fid = float(abs(psi @ vecs[:, 0]) ** 2)
    return res.x, {"energy_error": float(res.fun - w[0]), "fidelity": fid}


def pair_statistics(x):
    """(qg_v, qg_h, eta, I) for every visible-hidden pair of P(v, h) ~ exp(a.v + b.h + v W h)."""
    a, b, W = unpack(x)
    m = b.size
    Hs = np.array(list(itertools.product((1, -1), repeat=m)), dtype=float)
    logp = (_V @ a)[:, None] + (Hs @ b)[None, :] + _V @ W @ Hs.T
    P = np.exp(logp - logp.max())
    P /= P.sum()
    qv = P.sum(axis=1) @ _V  # (n,)
    qh = P.sum(axis=0) @ Hs  # (m,)
    svh = _V.T @ P @ Hs  # (n, m) = <s_k h_m>
    out = []
    for k in range(N_VIS):
        for j in range(m):
            eta = svh[k, j] - qv[k] * qh[j]
            out.append((float(qv[k]), float(qh[j]), float(eta), mutual_information(qv[k], qh[j], eta)))
    return out


def study(g_values=G_VALUES, hz_values=HZ_VALUES, seeds=SEEDS, symmetric=False):
    rows = {}
    for g in g_values:
        for hz in hz_values:
            gaps, rel, qv, qh, etas, fids, errs, pred = [], [], [], [], [], [], [], []
            for s in seeds:
                x, fit = train(g, hz, s, symmetric=symmetric)
                fids.append(fit["fidelity"])
                errs.append(fit["energy_error"])
                for v_, h_, e_, i_ in pair_statistics(x):
                    gap = i_ - lower_bound(e_)
                    gaps.append(gap)
                    pred.append(small_bias_gap(v_, h_, e_))
                    qv.append(abs(v_))
                    qh.append(abs(h_))
                    etas.append(abs(e_))
            gaps, pred = np.array(gaps), np.array(pred)
            rows[(g, hz)] = {
                "fidelity": float(np.median(fids)), "energy_error": float(np.median(errs)),
                "mean_abs_qg_v": float(np.mean(qv)), "mean_abs_qg_h": float(np.mean(qh)),
                "mean_abs_eta": float(np.mean(etas)), "max_gap": float(gaps.max()), "mean_gap": float(gaps.mean()),
                "mean_gap_small_bias": float(pred.mean()),
            }
    return rows


def make_figure(path, g=1.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    e = np.linspace(0, 1, 200)
    for ax in (ax1, ax2):
        ax.plot(e, [upper_bound(t) for t in e], "--", color="#e0a030", lw=1.5, label="UB: |qg| = sqrt(1 - eta), locked")
        ax.plot(e, [lower_bound(t) for t in e], "-", color="#1f6fb2", lw=1.5, label="LB: qg_v = qg_h = 0")
        ax.set_xlabel("eta = Cov(s, h)")
        ax.grid(alpha=0.3)
    for qg in (0.3, 0.6, 0.9):
        ok = [t for t in e if t <= 1 - qg**2 + 1e-12]
        ok = [t for t in ok if pair_distribution(qg, qg, t).min() >= 0]
        ax1.plot(ok, [mutual_information(qg, qg, t) for t in ok], color="#8c8c8c", lw=1)
        ax1.text(ok[-1], mutual_information(qg, qg, ok[-1]), f" qg = {qg}", fontsize=7, color="#555555")
    ax1.set_ylabel("mutual information I (bits)")
    ax1.set_title("The region is swept by the marginals (grey: qg_v = qg_h)", fontsize=9)
    ax1.legend(fontsize=7, loc="upper left")
    cols = {0.0: "#1f6fb2", 0.1: "#e0a030", 0.3: "#8c2d04"}
    for hz in HZ_VALUES:
        x, _ = train(g, hz, 0)
        pts = pair_statistics(x)
        ax2.scatter([abs(p[2]) for p in pts], [p[3] for p in pts], s=18, color=cols[hz], label=f"trained RBM, h_z = {hz}")
    ax2.set_xlim(0, max(0.5, ax2.get_xlim()[1]))
    ax2.set_ylim(0, None)
    ax2.set_title(f"Trained RBMs, TFIM g = {g}, n = {N_VIS}: the field moves them off LB", fontsize=9)
    ax2.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    import sys

    print("Closed forms: min over marginals equals LB; UB at |qg| = sqrt(1 - eta)")
    for eta in (0.1, 0.3, 0.5, 0.8):
        r = math.sqrt(1 - eta)
        print(f"  eta {eta}: LB {lower_bound(eta):.4f} = I(0, 0) {mutual_information(0, 0, eta):.4f}; "
              f"UB {upper_bound(eta):.4f} = I(r, r) {mutual_information(r, r, eta):.4f}")
    rows = study()
    rows_sym = study(g_values=(0.5,), hz_values=(0.0,), symmetric=True)
    rows.update({("0.5 sym", 0.0): rows_sym[(0.5, 0.0)]})
    print(f"\n{'g':>7} {'h_z':>4} | {'fid':>6} {'dE':>8} | {'|qg_v|':>7} {'|qg_h|':>7} {'|eta|':>6} | "
          f"{'mean I-LB':>10} {'max I-LB':>9} {'small-bias':>10}")
    for (g, hz), r in rows.items():
        print(f"{str(g):>7} {hz:4.1f} | {r['fidelity']:6.4f} {r['energy_error']:8.1e} | {r['mean_abs_qg_v']:7.4f} "
              f"{r['mean_abs_qg_h']:7.4f} {r['mean_abs_eta']:6.3f} | {r['mean_gap']:10.2e} {r['max_gap']:9.2e} "
              f"{r['mean_gap_small_bias']:10.2e}")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"))
