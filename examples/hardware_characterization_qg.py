"""
Hardware characterization in qg: T1, T2 (T_phi), effective temperature
and asymmetric readout error from one heralded sweep, vs the standard
calibration suite.

The confound. A real qubit idles in a thermal state with residual
excited population p (qg_eq = 1 - 2p = tanh(hf / 2 k_B T_eff), §25), and
its readout flips 0 -> 1 with probability e01 and 1 -> 0 with e10. In qg
units the readout is an affine map of the true qg_Z:

    measured qg = a + b qg,   a = e10 - e01,   b = 1 - e01 - e10.

The standard suite calibrates readout by preparing "|0>" (really the
thermal state) and "|1>" (X on it). Thermal population is then counted
as readout error: e01 is overestimated by ~p (1 - e01 - e10), and after
readout correction the qubit looks perfectly cold (p = 0, T_eff = 0).
Z-basis experiments alone cannot separate a, b and qg_eq: they only see
a + b qg_eq and b qg_eq.

The qg protocol. Measure once (herald), then apply I, X or Ry(pi/2),
wait t, rotate back and measure again. For ideal QND readout the pair
(r1, r2) of +-1 outcomes at t = 0 gives, in closed form,

    m   = E[r1]            = a + b qg_eq
    V   = E[r1 r2](I) - m^2 = b^2 (1 - qg_eq^2)   (readout covariance)
    C_X = E[r1 r2](X)      = a^2 - b^2
    u = b qg_eq = (m^2 - V - C_X) / (2m),  b = sqrt(V + u^2),  a = m - u,

so e01, e10 and qg_eq (hence T_eff) separate. The delays then give T1
(after X) and T2 (after Ry(pi/2), i.e. the decay of qg_X), and a joint
maximum-likelihood fit of all joint counts returns (a, b, qg_eq, T1, T2).
Same number of circuits and shots as the standard suite; each qg circuit
has one extra (mid-circuit) measurement.

The simulator is an exact single-qubit model (thermal start, generalized
amplitude damping to qg_eq with T1, dephasing with T2, perfect gates,
asymmetric readout, QND measurement) sampled with a multinomial;
tests/ cross-check it against a Qiskit Aer circuit.

Findings (T1 = 100 us, T2 = 70 us, e01 = 0.015, e10 = 0.04, 5 GHz
qubit; 16 circuits x 2000 shots for each protocol; median ± std over
100 repetitions):

  true p (T_eff)   | e01: standard / qg      | T_eff (mK): standard / qg
  0     (0 mK)     | 0.0150 / 0.0149         |  0 / 11 ± 14
  0.01  (52 mK)    | 0.0245 / 0.0151         |  0 / 52.3 ± 1.0
  0.03  (69 mK)    | 0.0415 / 0.0148         |  0 / 69.0 ± 0.8
  0.08  (98 mK)    | 0.0915 / 0.0152         |  0 / 98.2 ± 0.9

  * The standard suite attributes the whole thermal population to
    readout error (e01 overestimated by exactly p·b, e10 by a similar
    amount) and reports a perfectly cold qubit at every temperature.
  * The qg heralded sweep is unbiased for e01, e10, p and T_eff; T_eff
    is resolved to ~1 mK from 52 mK upwards (at p = 0 it can only give
    an upper bound).
  * T1 and T2 are unbiased in both. The qg joint fit is also ~1.5x more
    precise (T1 ± 1.6-2.0 vs ± 2.6-3.6 us, T2 ± 2.1-2.5 vs ± 3.4-4.9 us,
    T_phi ± 5-6 vs ± 8-12 us). Part of that comes from the joint
    maximum-likelihood fit rather than from the herald itself.
  * Robustness: if the herald readout is not QND (|1> relaxes to |0>
    with probability 0.05 during it), the qg estimates move by ~0.001
    (e01 0.0163, p 0.0289 at true 0.015, 0.03).

Scope: the model assumes perfect gates and a QND herald. The method
needs mid-circuit measurement, which IBM devices support and some other
platforms do not. Separating thermal population from readout by repeated
measurement is known practice; what qg adds is the three-line closed
form (m, V, C_X) and T_eff read directly from qg_eq = tanh(hf/2k_BT).
"""

import numpy as np
from scipy.optimize import curve_fit, minimize

H_PLANCK, K_B = 6.62607015e-34, 1.380649e-23
FREQ_HZ = 5.0e9
DELAYS_US = (0.0, 10.0, 25.0, 50.0, 100.0, 150.0, 250.0, 400.0)
TRUE = {"T1": 100.0, "T2": 70.0, "e01": 0.015, "e10": 0.04}  # microseconds, probabilities


def t_eff_from_qg(qg, freq=FREQ_HZ):
    """T_eff = h f / (2 k_B artanh qg); 0 for qg >= 1, inf for qg <= 0."""
    qg = np.asarray(qg, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(qg >= 1, 0.0, np.where(qg <= 0, np.inf,
                                               H_PLANCK * freq / (2 * K_B * np.arctanh(np.clip(qg, 1e-15, 1 - 1e-15)))))


def readout_affine(e01, e10):
    return e10 - e01, 1.0 - e01 - e10


def readout_errors(a, b):
    return (1.0 - b - a) / 2.0, (1.0 - b + a) / 2.0


# --------------------------------------------------------------------- #
# Exact model
# --------------------------------------------------------------------- #
def _z_after(z0, x0, op, t, qg_eq, T1, T2):
    """Bloch (x, z) -> final measured-axis value after op, delay, rotation back."""
    if op == "I":
        x, z = x0, z0
    elif op == "X":
        x, z = x0, -z0
    elif op == "Y90":  # Ry(pi/2): z -> x
        x, z = z0, -x0
    else:
        raise ValueError(op)
    z = qg_eq + (z - qg_eq) * np.exp(-t / T1)
    x = x * np.exp(-t / T2)
    return x if op == "Y90" else z  # Ry(-pi/2) before measuring maps x -> z


def joint_probabilities(op, t, a, b, qg_eq, T1, T2, herald=True, meas_relax=0.0):
    """Probabilities of the outcomes. With herald: array over (r1, r2) in
    order (+1,+1), (+1,-1), (-1,+1), (-1,-1); without: (+1, -1) for a
    single measurement of the thermal state after op and delay.
    meas_relax: probability that |1> relaxes to |0> during the herald
    readout (a non-QND stress test)."""
    mu = lambda s: a + b * s  # noqa: E731  mean outcome given true s = +-1
    if not herald:
        zf = _z_after(qg_eq, 0.0, op, t, qg_eq, T1, T2)
        p = (1 + a + b * zf) / 2
        return np.array([p, 1 - p])
    out = np.zeros(4)
    for s, ps in ((+1, (1 + qg_eq) / 2), (-1, (1 - qg_eq) / 2)):
        p_r1 = {+1: (1 + mu(s)) / 2, -1: (1 - mu(s)) / 2}
        post = [(s, 1.0)] if s == +1 else [(-1, 1.0 - meas_relax), (+1, meas_relax)]
        for s_post, w in post:
            zf = _z_after(float(s_post), 0.0, op, t, qg_eq, T1, T2)
            p2 = (1 + a + b * zf) / 2
            for i, r1 in enumerate((+1, -1)):
                out[2 * i] += ps * w * p_r1[r1] * p2
                out[2 * i + 1] += ps * w * p_r1[r1] * (1 - p2)
    return out


def truth(p_th, **overrides):
    d = dict(TRUE, **overrides)
    a, b = readout_affine(d["e01"], d["e10"])
    return {"a": a, "b": b, "qg_eq": 1 - 2 * p_th, "T1": d["T1"], "T2": d["T2"]}


# --------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------- #
def standard_circuits(delays=DELAYS_US):
    return [("I", 0.0)] + [("X", t) for t in delays] + [("Y90", t) for t in delays if t > 0]


def qg_circuits(delays=DELAYS_US):
    return [("I", 0.0)] + [("X", t) for t in delays] + [("Y90", t) for t in delays if t > 0]


def sample(protocol, params, shots, rng, meas_relax=0.0):
    circuits = standard_circuits() if protocol == "standard" else qg_circuits()
    herald = protocol == "qg"
    out = []
    for op, t in circuits:
        p = joint_probabilities(op, t, params["a"], params["b"], params["qg_eq"], params["T1"], params["T2"],
                                herald=herald, meas_relax=meas_relax)
        out.append((op, t, rng.multinomial(shots, p / p.sum())))
    return out


def fit_standard(data):
    """The usual suite: assignment matrix from I / X at t = 0, then
    readout-corrected exponential fits for T1 and T2; the equilibrium
    excited population is read from the corrected |0> preparation."""
    d = {(op, t): c for op, t, c in data}
    n0, n1 = d[("I", 0.0)], d[("X", 0.0)]
    e01 = n0[1] / n0.sum()           # read 1 after "prepare 0"
    e10 = n1[0] / n1.sum()           # read 0 after "prepare 1"
    a, b = readout_affine(e01, e10)
    corr = lambda c: ((c[0] - c[1]) / c.sum() - a) / b  # noqa: E731  corrected qg_Z
    q_eq = float(np.clip(corr(n0), -1, 1))
    tx = np.array(sorted(t for op, t in d if op == "X"))
    zx = np.array([corr(d[("X", t)]) for t in tx])
    (A, T1, B), _ = curve_fit(lambda t, A, T1, B: B - A * np.exp(-t / T1), tx, zx, p0=(2, 80, 1), maxfev=20000)
    ty = np.array(sorted(t for op, t in d if op == "Y90"))
    zy = np.array([corr(d[("Y90", t)]) for t in ty])
    (Ay, T2), _ = curve_fit(lambda t, A, T2: A * np.exp(-t / T2), ty, zy, p0=(max(zy[0], 0.1), 60), maxfev=20000)
    return {"e01": float(e01), "e10": float(e10), "qg_eq": q_eq, "T1": float(T1), "T2": float(T2)}


def closed_form_t0(data):
    """(a, b, qg_eq) from the t = 0 heralded pairs (formulas in the docstring)."""
    d = {(op, t): c for op, t, c in data}
    cI, cX = d[("I", 0.0)], d[("X", 0.0)]
    sign = np.array([1, -1, -1, 1])       # r1 * r2 for (++, +-, -+, --)
    r1 = np.array([1, 1, -1, -1])
    m = (r1 @ cI + r1 @ cX) / (cI.sum() + cX.sum())
    V = sign @ cI / cI.sum() - m**2
    CX = sign @ cX / cX.sum()
    u = (m**2 - V - CX) / (2 * m)
    b = np.sqrt(max(V + u**2, 1e-12))
    return m - u, b, float(np.clip(u / b, -1, 1))


def fit_qg(data, meas_relax=0.0):
    """Joint maximum likelihood of (a, b, qg_eq, T1, T2) over all heralded
    joint counts, started from the closed form and simple fits."""
    a0, b0, q0 = closed_form_t0(data)
    x0 = np.array([a0, b0, q0, np.log(90.0), np.log(60.0)])  # times as log(us)

    def nll(x):
        a, b, q, l1, l2 = x
        T1, T2 = np.exp(l1), np.exp(l2)
        s = 0.0
        for op, t, c in data:
            p = joint_probabilities(op, t, a, b, q, T1, T2, herald=True, meas_relax=meas_relax)
            s -= c @ np.log(np.clip(p, 1e-300, None))
        return s

    res = minimize(nll, x0, method="L-BFGS-B",
                   bounds=[(-0.5, 0.5), (0.3, 1.0), (0.0, 1.0), (0.0, 8.0), (0.0, 8.0)],
                   options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 2000})
    res = minimize(nll, res.x, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
    a, b, q = res.x[:3]
    T1, T2 = np.exp(res.x[3:])
    e01, e10 = readout_errors(a, b)
    return {"e01": float(e01), "e10": float(e10), "qg_eq": float(q), "T1": float(T1), "T2": float(T2)}


def t_phi(T1, T2):
    inv = 1.0 / T2 - 1.0 / (2.0 * T1)
    return np.inf if inv <= 0 else 1.0 / inv


def benchmark(p_th, shots=2000, reps=100, seed=0, meas_relax=0.0):
    """Bias and spread of every parameter for both protocols."""
    rng = np.random.default_rng(seed)
    tr = truth(p_th)
    e01, e10 = readout_errors(tr["a"], tr["b"])
    true = {"e01": e01, "e10": e10, "p_th": p_th, "T1": tr["T1"], "T2": tr["T2"],
            "T_phi": t_phi(tr["T1"], tr["T2"]), "T_eff_mK": 1e3 * float(t_eff_from_qg(tr["qg_eq"]))}
    est = {"standard": [], "qg": []}
    for _ in range(reps):
        for proto in est:
            f = (fit_standard if proto == "standard" else fit_qg)(sample(proto, tr, shots, rng, meas_relax))
            f["p_th"] = (1 - f["qg_eq"]) / 2
            f["T_phi"] = t_phi(f["T1"], f["T2"])
            f["T_eff_mK"] = 1e3 * float(t_eff_from_qg(f["qg_eq"]))
            est[proto].append(f)
    keys = ("e01", "e10", "p_th", "T_eff_mK", "T1", "T2", "T_phi")
    summary = {}
    for proto, rows in est.items():
        summary[proto] = {}
        for k in keys:
            v = np.array([r[k] for r in rows], dtype=float)
            summary[proto][k] = (float(np.median(v)), float(np.std(v[np.isfinite(v)])) if np.isfinite(v).any() else np.inf)
    return true, summary


P_TH_SWEEP = (0.0, 0.01, 0.03, 0.08)


def make_figure(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"standard": "#8c8c8c", "qg": "#1f6fb2"}
    lab = {"standard": "standard suite", "qg": "qg heralded sweep"}
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    panels = [("e01", "readout error e01 (0->1)"), ("T_eff_mK", "effective temperature (mK)"),
              ("T1", "T1 (us)"), ("T2", "T2 (us)")]
    x = np.array(P_TH_SWEEP)
    for ax, (k, title) in zip(axes, panels):
        ax.plot(x, [results[p][0][k] for p in P_TH_SWEEP], "k-", lw=1.5, label="true")
        for j, proto in enumerate(("standard", "qg")):
            med = [results[p][1][proto][k][0] for p in P_TH_SWEEP]
            sd = [results[p][1][proto][k][1] for p in P_TH_SWEEP]
            ax.errorbar(x + (j - 0.5) * 0.002, med, yerr=sd, fmt="o", color=col[proto], ms=6, capsize=3,
                        label=lab[proto], mec="white")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("true excited population p")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("One qubit: thermal population p vs readout error. Median ± std of 100 repetitions, "
                 "16 circuits x 2000 shots each", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    keys = ("e01", "e10", "p_th", "T_eff_mK", "T1", "T2", "T_phi")
    results = {}
    for p_th in P_TH_SWEEP:
        true, s = results[p_th] = benchmark(p_th)
        print(f"\np_th = {p_th} (T_eff {true['T_eff_mK']:.0f} mK); median ± std over 100 repetitions, "
              f"2000 shots x {len(qg_circuits())} circuits each")
        print(f"{'':10s}" + "".join(f"{k:>16s}" for k in keys))
        print(f"{'true':10s}" + "".join(f"{true[k]:16.4g}" for k in keys))
        for proto in ("standard", "qg"):
            print(f"{proto:10s}" + "".join(f"{s[proto][k][0]:9.4g}±{s[proto][k][1]:<6.2g}" for k in keys))
    print("\nstress test: non-QND readout (|1> relaxes to |0> during the herald with prob 0.05), p_th = 0.03")
    true, s = benchmark(0.03, meas_relax=0.05)
    for proto in ("standard", "qg"):
        print(f"{proto:10s}" + "".join(f"{k} {s[proto][k][0]:.4g}  " for k in keys))
    import sys

    if "--figure" in sys.argv:
        make_figure(results, __file__.replace(".py", ".png"))
