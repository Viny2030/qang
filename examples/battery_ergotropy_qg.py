"""
Qubit quantum batteries in qg: how much work a qubit stores, how long it
keeps it under T1 and T2, how to certify it from shots, and how much of a
register's charge is locally extractable.

A qubit with H = omega |1><1| (|0> ground, qg_Z = +1) and Bloch vector
(qg_X, qg_Y, qg_Z), r = |vector|, stores energy E = omega (1 - qg_Z)/2.
Its ergotropy (the work extractable by a unitary) is the energy above
the passive state with the same spectrum (1 +- r)/2:

    W      = (omega/2) (r - qg_Z)
    W_inc  = omega max(0, -qg_Z)          (the dephased state's ergotropy)
    W_coh  = (omega/2) (r - |qg_Z|)       (the part carried by coherence)

A. Closed forms, checked against the general eigenvalue formula.
B. Storage. A qubit charged at polar angle theta (qg_Z0 = cos theta)
   decays with T1 and T2: qg_Z(t) = 1 - (1 - cos theta) e^{-t/T1},
   r_perp(t) = sin theta e^{-t/T2}. The fully inverted battery
   (theta = pi) has no coherence, and its ergotropy dies at t = T1 ln 2
   (population inversion lost). A tilted battery keeps coherent ergotropy
   longer. Which angle to charge to, for a storage time t, is a rule in
   T2/T1 alone.
C. Certification. W needs all three axes and is not linear in the state
   (r), so the plug-in estimate from finite shots is biased upwards near
   passive states. A lower confidence bound built from a lower bound on r
   and an upper bound on qg_Z (as for the certified randomness of §41) is
   compared with the plug-in estimate.
D. Registers. For n qubits, global ergotropy (any unitary on the whole
   register) against the sum of local ergotropies (one qubit at a time).
   A Dicke state D(n, k) is pure, so all its energy k omega is globally
   extractable, while every qubit alone has qg_Z = 1 - 2k/n and no
   coherence, so for k <= n/2 nothing is locally extractable: the register
   mean qg (the §20 witness) sets the local part. A product state with the
   same energy is fully locally extractable. Both are aged under local T1
   and T2.

omega = 1 throughout; times in units of T1.

Findings (python examples/battery_ergotropy_qg.py):

  A. The closed forms match the general eigenvalue formula to 3e-16 on
     200 random states, and the storage formula matches an integrated
     Lindblad equation (0.088130 both).

  B. Storage. Best charging angle and ergotropy after storage time t
     (W_best / W_inverted / W_equator, units of omega):

       T2/T1  crossover t_c   t = 0.5 T1              t = 1 T1               t = 2 T1
       2      0.288 T1        0.70 pi .297/.213/.240   0.58 pi .129/0/.122    0.52 pi .038/0/.038
       1      0.405 T1        0.78 pi .234/.213/.165   0.59 pi .054/0/.050    0.53 pi .005/0/.005
       0.5    0.618 T1        1.00 pi .213/.213/.073   0.60 pi .008/0/.007    ~0
       0.2    0.692 T1        1.00 pi .213/.213/.004   ~0                     ~0

     * Rule: charge fully (theta = pi) if the battery will be used before
       t_c; otherwise tilt it, to about 0.6 pi at t = T1. t_c solves
       x^(2 T1/T2 - 1) = 2 (2x - 1), x = e^{-t/T1} (small-tilt analysis,
       equal to the numerical optimum to 1e-3): ln(4/3) T1 when T2 = 2 T1,
       ln(3/2) T1 when T2 = T1, and it tends to T1 ln 2, the death of the
       inverted battery, as T2 shrinks.
     * After T1 ln 2 the inverted battery stores nothing; a tilted one
       still stores 0.13 omega at t = T1 when T2 = 2 T1 (and 0.05 when
       T2 = T1). Coherence buys storage time only if T2 is comparable to
       T1: at T2 = 0.2 T1 it buys nothing. Trapped ions (T1 -> infinity)
       should always charge fully.

  C. Certification (shots per axis):

       state                truth    plug-in (P over truth)   3-sigma lower bound (P unsafe)
       inverted, t = 0.5    0.213    0.237 (0.60) / 0.215 (0.53)   0.005 / 0.086  (0 / 0)
       inverted, t = 0.8    0        0.042 (0.99) / 0.005 (1.00)   0 / 0          (0 / 0)
       tilted,  t = 1       0.054    0.061 (0.58) / 0.054 (0.52)   0 / 0
                            (100 / 1000 shots)

     * The plug-in estimate overestimates in more than half of the runs and
       reports positive ergotropy for a passive state in every run: r is
       estimated upwards by the shot noise of the three axes.
     * The lower bound is never unsafe but is expensive: with 1000 shots
       per axis it certifies 0.086 of 0.213 omega, and nothing for a
       nearly discharged battery. Certifying small charges needs many shots
       (the pole of qg at r -> |qg_Z| again).

  D. Register n = 4, k = 2 excitations, T2 = T1 (global ergotropy / sum of
     local ergotropies; mean qg_Z is the same for both states):

       t / T1          0             0.1           0.3           0.7           1.5
       Dicke D(4,2)    2.00 / 0      1.49 / 0      0.71 / 0      0.05 / 0      0 / 0
       product         2.00 / 2.00   1.63 / 1.63   1.05 / 1.05   0.41 / 0.41   0.06 / 0.06
       mean qg_Z       0             +0.095        +0.259        +0.503        +0.777

     * The Dicke battery's charge is entirely locked: no single-qubit
       operation extracts any of it, and every qubit looks passive
       (qg_Z = 1 - 2k/n = 0, no coherence). The register mean qg cannot
       tell it from the product battery with the same energy; the sum of
       local ergotropies can.
     * The locked charge is also more fragile under local noise: 0.71 vs
       1.05 omega at t = 0.3 T1, 0.05 vs 0.41 at 0.7 T1.

Honest scope. The formulas are standard single-qubit ergotropy written in
qg; locked (locally passive, globally charged) batteries and coherence-
assisted storage are known in the quantum-battery literature. The
contributions are the T2/T1 charging rule with its closed-form crossover,
the certification cost, and the qg reading of which charge is locally
extractable. Markovian T1/T2 noise, one register size.
"""

import math

import numpy as np

T2_RATIOS = (2.0, 1.0, 0.5, 0.2)


# --------------------------------------------------------------------- #
# A. closed forms
# --------------------------------------------------------------------- #
def ergotropy(qx, qy, qz):
    r = math.sqrt(qx * qx + qy * qy + qz * qz)
    return 0.5 * (r - qz)


def ergotropy_incoherent(qz):
    return max(0.0, -qz)


def ergotropy_coherent(qx, qy, qz):
    r = math.sqrt(qx * qx + qy * qy + qz * qz)
    return 0.5 * (r - abs(qz))


def ergotropy_general(rho, energies):
    """Ergotropy of rho for a Hamiltonian diagonal in the computational basis."""
    e = np.sort(np.asarray(energies, dtype=float))
    lam = np.sort(np.real(np.linalg.eigvalsh(rho)))[::-1]
    energy = float(np.real(np.trace(rho @ np.diag(energies))))
    return energy - float(np.sum(lam * e))


def rho_from_bloch(qx, qy, qz):
    return 0.5 * np.array([[1 + qz, qx - 1j * qy], [qx + 1j * qy, 1 - qz]])


# --------------------------------------------------------------------- #
# B. storage under T1 and T2
# --------------------------------------------------------------------- #
def aged(theta, t, t2_over_t1):
    """(r_perp, qg_Z) after storage time t (units of T1)."""
    qz = 1 - (1 - math.cos(theta)) * math.exp(-t)
    rp = math.sin(theta) * math.exp(-t / t2_over_t1)
    return rp, qz


def stored_ergotropy(theta, t, t2_over_t1):
    rp, qz = aged(theta, t, t2_over_t1)
    return ergotropy(rp, 0.0, qz)


_THETAS = np.linspace(0, math.pi, 1801)


def best_angle(t, t2_over_t1):
    w = np.array([stored_ergotropy(th, t, t2_over_t1) for th in _THETAS])
    i = int(np.argmax(w))
    return float(_THETAS[i]), float(w[i])


def crossover_time(t2_over_t1, tol=1e-3):
    """First storage time at which a tilted battery beats the inverted one."""
    lo, hi = 0.0, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        th, _ = best_angle(mid, t2_over_t1)
        if th < math.pi - tol:
            hi = mid
        else:
            lo = mid
    return hi


def crossover_time_analytic(t2_over_t1):
    """Small-tilt condition for leaving theta = pi: with x = e^{-t},
    x^(2 T1/T2 - 1) = 2 (2x - 1); closed forms t_c = ln(4/3) for T2 = 2 T1
    and ln(3/2) for T2 = T1."""
    from scipy.optimize import brentq

    a = 2.0 / t2_over_t1 - 1.0
    f = lambda x: x**a - 2 * (2 * x - 1)  # noqa: E731
    x = brentq(f, 0.5 + 1e-12, 1.0 - 1e-12)
    return -math.log(x)


def storage_table(times=(0.25, 0.5, 1.0, 2.0)):
    out = {}
    for ratio in T2_RATIOS:
        row = {"t_c": crossover_time(ratio), "t_c_analytic": crossover_time_analytic(ratio)}
        for t in times:
            th, w = best_angle(t, ratio)
            row[t] = {"theta_opt": th, "w_opt": w, "w_inverted": stored_ergotropy(math.pi, t, ratio),
                      "w_equator": stored_ergotropy(math.pi / 2, t, ratio)}
        out[ratio] = row
    return out


def lindblad_check(theta, t, t2_over_t1, steps=4000):
    """Integrate the qubit master equation (amplitude damping + pure dephasing)."""
    g1 = 1.0
    gphi = 1.0 / t2_over_t1 - 0.5
    rho = rho_from_bloch(math.sin(theta), 0.0, math.cos(theta)).astype(complex)
    sm = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1| : lowers |1> -> |0>
    sz = np.diag([1.0, -1.0]).astype(complex)
    dt = t / steps

    def d(r):
        out = g1 * (sm @ r @ sm.conj().T - 0.5 * (sm.conj().T @ sm @ r + r @ sm.conj().T @ sm))
        out += 0.5 * gphi * (sz @ r @ sz - r)
        return out

    for _ in range(steps):
        k1 = d(rho)
        k2 = d(rho + 0.5 * dt * k1)
        k3 = d(rho + 0.5 * dt * k2)
        k4 = d(rho + dt * k3)
        rho = rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ergotropy_general(rho, [0.0, 1.0])


# --------------------------------------------------------------------- #
# C. certification from shots
# --------------------------------------------------------------------- #
def certify(bloch, shots, trials=4000, seed=0, sigmas=3.0):
    """Plug-in ergotropy and a lower confidence bound from `shots` shots per axis."""
    rng = np.random.default_rng(seed)
    qx, qy, qz = bloch
    truth = ergotropy(qx, qy, qz)
    est = {a: 2 * rng.binomial(shots, (1 + q) / 2, size=trials) / shots - 1 for a, q in zip("xyz", bloch)}
    r_hat = np.sqrt(est["x"] ** 2 + est["y"] ** 2 + est["z"] ** 2)
    plug = 0.5 * (np.minimum(r_hat, 1.0) - est["z"])
    se = 1 / math.sqrt(shots)
    r_lo = np.maximum(0.0, r_hat - sigmas * se * math.sqrt(3))
    qz_hi = np.minimum(1.0, est["z"] + sigmas * se)
    lcb = np.maximum(0.0, 0.5 * (r_lo - qz_hi))
    return {"truth": truth, "plug_mean": float(plug.mean()), "plug_over": float(np.mean(plug > truth + 1e-12)),
            "lcb_mean": float(lcb.mean()), "lcb_unsafe": float(np.mean(lcb > truth + 1e-12))}


# --------------------------------------------------------------------- #
# D. registers
# --------------------------------------------------------------------- #
def dicke(n, k):
    v = np.zeros(2**n)
    for i in range(2**n):
        if bin(i).count("1") == k:
            v[i] = 1
    v /= np.linalg.norm(v)
    return np.outer(v, v)


def product_equal_energy(n, k):
    th = math.acos(1 - 2 * k / n)
    one = np.array([math.cos(th / 2), math.sin(th / 2)])
    v = one
    for _ in range(n - 1):
        v = np.kron(v, one)
    return np.outer(v, v)


def _local_channel(rho, n, t, t2_over_t1):
    """Amplitude damping (gamma = 1 - e^-t) and pure dephasing on every qubit."""
    gamma = 1 - math.exp(-t)
    lam = math.exp(-t / t2_over_t1) / math.sqrt(1 - gamma) if gamma < 1 else 0.0  # extra dephasing factor
    K_ad = [np.array([[1, 0], [0, math.sqrt(1 - gamma)]]), np.array([[0, math.sqrt(gamma)], [0, 0]])]
    p = (1 - lam) / 2
    K_ph = [math.sqrt(1 - p) * np.eye(2), math.sqrt(p) * np.diag([1.0, -1.0])]
    for q in range(n):
        for Ks in (K_ad, K_ph):
            new = np.zeros_like(rho)
            for K in Ks:
                full = np.kron(np.kron(np.eye(2 ** (n - 1 - q)), K), np.eye(2**q))
                new = new + full @ rho @ full.conj().T
            rho = new
    return rho


def _energies(n):
    return np.array([bin(i).count("1") for i in range(2**n)], dtype=float)


def _local_bloch(rho, n, q):
    t = rho.reshape([2] * (2 * n))
    keep = n - 1 - q
    axes = [a for a in range(n) if a != keep]
    red = t
    for a in sorted(axes, reverse=True):
        red = np.trace(red, axis1=a, axis2=a + red.ndim // 2)
    red = red.reshape(2, 2)
    qz = float(np.real(red[0, 0] - red[1, 1]))
    qx = float(2 * np.real(red[1, 0]))
    qy = float(2 * np.imag(red[1, 0]))
    return qx, qy, qz


def register_ergotropy(rho, n):
    glob = ergotropy_general(rho, _energies(n))
    loc = sum(ergotropy(*_local_bloch(rho, n, q)) for q in range(n))
    mean_qg = float(np.mean([_local_bloch(rho, n, q)[2] for q in range(n)]))
    return glob, loc, mean_qg


def register_table(n=4, k=2, times=(0.0, 0.1, 0.3, 0.7, 1.5), t2_over_t1=1.0):
    out = {}
    for name, rho0 in (("Dicke", dicke(n, k)), ("product", product_equal_energy(n, k))):
        out[name] = {t: register_ergotropy(_local_channel(rho0, n, t, t2_over_t1), n) for t in times}
    return out


def make_figure(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ts = np.linspace(0, 3, 121)
    cols = {2.0: "#1f6fb2", 1.0: "#e0a030", 0.5: "#8c2d04"}
    for ratio, c in cols.items():
        ax1.plot(ts, [stored_ergotropy(math.pi, t, ratio) for t in ts], color=c, ls=":", lw=1.5)
        ax1.plot(ts, [best_angle(t, ratio)[1] for t in ts], color=c, lw=2, label=f"best angle, T2 = {ratio} T1")
    ax1.plot([], [], "k:", label="fully inverted (theta = pi)")
    ax1.axvline(math.log(2), color="#999999", lw=0.8)
    ax1.set_xlabel("storage time t / T1")
    ax1.set_ylabel("ergotropy (units of omega)")
    ax1.set_title("Qubit battery: inversion dies at T1 ln 2, coherence keeps charge", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    tt = np.linspace(0, 2, 41)
    for name, rho0, c in (("Dicke D(4,2)", dicke(4, 2), "#8c2d04"), ("product, same energy", product_equal_energy(4, 2), "#1f6fb2")):
        vals = [register_ergotropy(_local_channel(rho0, 4, t, 1.0), 4) for t in tt]
        ax2.plot(tt, [v[0] for v in vals], color=c, lw=2, label=f"{name}: global")
        ax2.plot(tt, [v[1] for v in vals], color=c, lw=1.5, ls="--", label=f"{name}: sum of local")
    ax2.set_xlabel("storage time t / T1 (T2 = T1)")
    ax2.set_title("4-qubit register, 2 excitations: locked vs local charge", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(200):
        v = rng.normal(size=3)
        v *= rng.uniform() ** (1 / 3) / np.linalg.norm(v)
        err = max(err, abs(ergotropy(*v) - ergotropy_general(rho_from_bloch(*v), [0.0, 1.0])))
        err = max(err, abs(ergotropy(*v) - ergotropy_incoherent(v[2]) - ergotropy_coherent(*v)))
    print(f"A. closed forms vs general formula, 200 random states: max deviation {err:.1e}")
    print(f"   Lindblad check (theta = 2.2, t = 0.8, T2 = T1): closed {stored_ergotropy(2.2, 0.8, 1.0):.6f}, "
          f"integrated {lindblad_check(2.2, 0.8, 1.0):.6f}")
    st = storage_table()
    print("\nB. storage: best charging angle and ergotropy (W_opt / W_inverted / W_equator)")
    for ratio, row in st.items():
        cells = "  ".join(f"t={t}: {row[t]['theta_opt'] / math.pi:.2f}pi {row[t]['w_opt']:.3f}/{row[t]['w_inverted']:.3f}"
                          f"/{row[t]['w_equator']:.3f}" for t in (0.25, 0.5, 1.0, 2.0))
        print(f"  T2 = {ratio} T1: crossover t_c = {row['t_c']:.3f} T1 (analytic {row['t_c_analytic']:.3f}) | {cells}")
    print("\nC. certification (100 and 1000 shots per axis): truth | plug-in mean (P over) | 3-sigma LCB mean (P unsafe)")
    for label, (th, t, ratio) in (("inverted, t = 0.5", (math.pi, 0.5, 1.0)), ("inverted, t = 0.8", (math.pi, 0.8, 1.0)),
                                  ("best, t = 1", (best_angle(1.0, 1.0)[0], 1.0, 1.0)), ("equator, t = 2", (math.pi / 2, 2.0, 1.0))):
        rp, qz = aged(th, t, ratio)
        for shots in (100, 1000):
            c = certify((rp, 0.0, qz), shots)
            print(f"  {label:18s} N={shots:5d}: {c['truth']:.4f} | {c['plug_mean']:.4f} ({c['plug_over']:.2f}) | "
                  f"{c['lcb_mean']:.4f} ({c['lcb_unsafe']:.2f})")
    reg = register_table()
    print("\nD. register n = 4, k = 2, T2 = T1: (global ergotropy, sum of local ergotropies, mean qg_Z)")
    for name, rows in reg.items():
        print(f"  {name:8s} " + "  ".join(f"t={t}: ({g:.3f}, {l:.3f}, {m:+.3f})" for t, (g, l, m) in rows.items()))
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"))
