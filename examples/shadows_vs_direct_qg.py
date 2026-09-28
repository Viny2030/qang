"""
Classical shadows vs direct measurement for the quantities qang uses.

The cost question a referee can ask of every qg result: with the same
number of shots, would randomized Pauli measurements (classical shadows,
Huang, Kueng, Preskill 2020) estimate the qg of a register better than
measuring the bases qang measures? Everything below is an exact
per-shot variance (no sampling noise), computed from the density matrix
of the state by enumerating all 3^n local Pauli settings, plus one Monte
Carlo check. The figure of merit is the shot ratio

    R = (shots shadows need) / (shots direct needs)  for the same error,

i.e. the ratio of per-shot variances (worst target, when a task has many).

Shadow estimator of a Pauli string P of weight k: 3^k * prod(outcomes)
when every qubit of P was measured in P's basis, else 0; per-shot
variance 3^k - <P>^2. Direct: measure the bases themselves; per-shot
variance 1 - <P>^2 if P is in every setting, and the setting's share of
the shots otherwise.

States (n = 6): the noisy XXZ Trotter state of §60 (4 steps, 3 CNOT,
p2 = 0.01, gamma = 0.005) and the noisy 3-layer LiH ansatz state of §50.

Tasks:
  T1 all n qg_Z                         direct: one Z setting
  T2 register-mean qg_Z (the §20 witness)
  T3 all C(n,2) ZZ correlators
  T4 all 3n single-qubit qg (X, Y, Z)    direct: 3 settings X.., Y.., Z..
  T5 all 9 C(n,2) two-qubit correlators  direct: 18 settings of the L18
                                         orthogonal array
  T6 filtered imbalance of §60 (keep N = n/2)  shadows: ratio estimator
  T7 LiH energy                          direct: qubit-wise commuting
                                         groups, optimal shot allocation

Findings (python examples/shadows_vs_direct_qg.py):

  Shot ratio R = shadows / direct for the same error (worst target;
  R > 1 means direct measurement is cheaper):

    task                                   XXZ     LiH
    T1 all n qg_Z                          3.0     4.3
    T2 register-mean qg_Z                  5.8     8.1
    T3 all C(n,2) ZZ                       9.0    12.6
    T4 all 3n single-qubit qg              1.0     1.0
    T5 all 9 C(n,2) 2-qubit correlators    1.0     1.0   (direct: 18 L18 settings)
    T6 filtered imbalance (§60)           41             (kept 0.60)
    T7 LiH energy (62 terms)                      4.0    (direct: 21 QWC groups)

  Monte Carlo check (T1, qubit 0, XXZ, 3000 x 2000 shots): sampled ratio
  3.17 against the exact 3.01.

  * For everything qang measures in Z (T1-T3) direct measurement wins by
    3^k for weight-k strings, more when the values are near +-1 (LiH,
    whose occupied orbitals have qg_Z close to -1: 4.3x and 12.6x).
  * The register-mean witness gains more than 3x (5.8-8.1x): measured
    directly, its per-shot variance is small because the conserved
    quantity makes the qubits anti-correlated; a shadow estimate sees
    each qubit's Z only in a third of the shots and loses that.
  * The shot-level filter does not exist for shadows: only 3^-6 = 0.14 %
    of random-basis shots have every qubit in Z. A shadow estimate of the
    filtered value (ratio of <Pi_N O> and <Pi_N>) is unbiased but needs
    41x the shots of post-selection.
  * When all local Paulis of a given weight are wanted (T4, T5), shadows
    and a designed direct scheme tie: 3 settings X..., Y..., Z... for
    weight 1, and the 18-run L18 orthogonal array for weight 2 give each
    Pauli the same 1/3 or 1/9 hit rate as random bases, and per target
    direct is never worse ((1 - q^2)/share <= 3^k - q^2). A greedy
    covering array (15 settings, unequal coverage) lost to shadows
    (R = 0.6): the tie needs a balanced design.
  * For a Hamiltonian (T7) qubit-wise-commuting grouping with optimal shot
    allocation needs 4x fewer shots than plain shadows, in line with the
    literature (derandomized and locally-biased shadows close part of
    that gap: Huang, Kueng, Preskill 2021; Hadfield et al. 2022).

  What this answers and what not. For the quantities of the qg framework
  (Z-basis qg, register-mean witness, shot-level filter) direct
  measurement is 3-41x cheaper than classical shadows; for "all local
  Paulis" it ties with a balanced design. Shadows keep their real
  advantages, not tested here: estimating observables chosen after the
  measurement, very many observables on large n without designing
  settings, and nonlinear quantities such as purities from randomized
  measurements. The results are for n = 6 and two states; the 3^k and
  3^-n factors are general.
"""

import itertools
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

N = 6
BASES = "XYZ"
_H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
_SDG_H = _H @ np.diag([1, -1j])  # maps Y eigenbasis to Z
ROT = {"X": _H, "Y": _SDG_H, "Z": np.eye(2)}


# --------------------------------------------------------------------- #
# outcome distributions for every local Pauli setting
# --------------------------------------------------------------------- #
def setting_probs(rho, setting):
    """Outcome distribution when qubit i is measured in setting[i]
    (bit i of the index = qubit i)."""
    U = np.array([[1.0]])
    for q in range(N - 1, -1, -1):
        U = np.kron(U, ROT[setting[q]])
    p = np.real(np.einsum("ij,jk,ik->i", U, rho, U.conj()))
    return np.clip(p, 0, None) / p.clip(0).sum()


def all_settings(rho):
    settings = ["".join(s) for s in itertools.product(BASES, repeat=N)]
    return settings, np.array([setting_probs(rho, s) for s in settings])


_IDX = np.arange(1 << N)
_BITS = (_IDX[:, None] >> np.arange(N)) & 1
_SIGN = 1 - 2 * _BITS  # outcome +-1 per qubit


def pauli_expect(rho, label):
    """<P> for label like {q: 'X', ...}."""
    s = ["Z"] * N
    for q, b in label.items():
        s[q] = b
    p = setting_probs(rho, "".join(s))
    val = np.prod(_SIGN[:, list(label)], axis=1) if label else np.ones(1 << N)
    return float(p @ val)


# --------------------------------------------------------------------- #
# per-shot variances
# --------------------------------------------------------------------- #
def shadow_var_pauli(exp, k):
    return 3**k - exp**2


def direct_var_pauli(exp, share=1.0):
    return (1 - exp**2) / share


_L18 = ("11111111 11222222 11333333 12112233 12223311 12331122 13121323 13232131 13313212 "
        "21133221 21211332 21322113 22123132 22231213 22312321 23132312 23213123 23321231")


def orthogonal_array(n=N):
    """18 settings from the Taguchi L18 array (its seven 3-level columns
    form an orthogonal array of strength 2): every pair of qubits sees
    each of the 9 basis pairs in exactly 2 settings, i.e. 1/9 of the
    shots -- the same hit rate as random Pauli shadows."""
    rows = [[int(c) - 1 for c in r][1:] for r in _L18.split()]
    assert n <= 7
    return ["".join(BASES[r[q]] for q in range(n)) for r in rows]


def greedy_covering(n=N, tries=200, seed=0):
    """Greedy settings until every pair of qubits has seen all 9 basis
    pairs at least once (unbalanced coverage, for comparison with L18)."""
    rng = np.random.default_rng(seed)
    need = {(i, j, a, b) for i, j in itertools.combinations(range(n), 2) for a in BASES for b in BASES}
    out = []
    while need:
        best, bestc = None, -1
        for _ in range(tries):
            st = "".join(rng.choice(list(BASES), n))
            c = sum((i, j, st[i], st[j]) in need for i, j in itertools.combinations(range(n), 2))
            if c > bestc:
                best, bestc = st, c
        out.append(best)
        need -= {(i, j, best[i], best[j]) for i, j in itertools.combinations(range(n), 2)}
    return out


def tasks_single(rho, t5_settings=None):
    """T1-T5 worst-target shot ratio shadows/direct, and details."""
    res = {}
    qz = [pauli_expect(rho, {q: "Z"}) for q in range(N)]
    # T1: all qg_Z
    res["T1 all qg_Z"] = max(shadow_var_pauli(e, 1) for e in qz) / max(direct_var_pauli(e) for e in qz)
    # T2: register mean -- exact variance of the mean of z_i per shot
    Zs = _SIGN.mean(axis=1)
    pz = setting_probs(rho, "Z" * N)
    vd = float(pz @ Zs**2 - (pz @ Zs) ** 2)
    settings, P = all_settings(rho)
    est = np.zeros((len(settings), 1 << N))
    for si, s in enumerate(settings):
        mask = np.array([c == "Z" for c in s])
        est[si] = 3 * (_SIGN * mask).mean(axis=1)
    m2 = (P * est**2).sum(axis=1).mean()
    vs = float(m2 - (pz @ Zs) ** 2)
    res["T2 register-mean qg_Z"] = vs / vd
    # T3: all ZZ
    zz = [pauli_expect(rho, {i: "Z", j: "Z"}) for i, j in itertools.combinations(range(N), 2)]
    res["T3 all ZZ"] = max(shadow_var_pauli(e, 2) for e in zz) / max(direct_var_pauli(e) for e in zz)
    # T4: all single-qubit qg
    ones = [pauli_expect(rho, {q: b}) for q in range(N) for b in BASES]
    res["T4 all 3n single-qubit qg"] = max(shadow_var_pauli(e, 1) for e in ones) / max(
        direct_var_pauli(e, 1 / 3) for e in ones)
    # T5: all two-qubit correlators with a covering array
    ca = orthogonal_array(N) if t5_settings is None else t5_settings
    worst_s = worst_d = 0.0
    for i, j in itertools.combinations(range(N), 2):
        for a in BASES:
            for b in BASES:
                e = pauli_expect(rho, {i: a, j: b})
                share = sum(1 for s in ca if s[i] == a and s[j] == b) / len(ca)
                worst_s = max(worst_s, shadow_var_pauli(e, 2))
                worst_d = max(worst_d, direct_var_pauli(e, share))
    res["T5 all 2-qubit correlators"] = worst_s / worst_d
    res["_settings_T5"] = len(ca)
    return res


def filtered_ratio(rho, obs_diag, keep):
    """T6: per-shot variance (delta method) of the filtered estimate
    <keep*O>/<keep> for direct post-selection and for shadows."""
    pz = setting_probs(rho, "Z" * N)
    kept = float(pz[keep].sum())
    R = float((pz[keep] * obs_diag[keep]).sum() / kept)
    v_direct = float((pz[keep] * (obs_diag[keep] - R) ** 2).sum() / kept) / kept
    settings, P = all_settings(rho)
    # snapshot value of a diagonal operator D: sum_x D(x) prod_i f_i(x_i),
    # f = 2 / -1 (qubit in Z, outcome equal / different), 1/2 (qubit in X or Y)
    a = np.zeros((len(settings), 1 << N))
    b = np.zeros_like(a)
    for si, s in enumerate(settings):
        zmask = np.array([c == "Z" for c in s])
        # F[outcome, x] = prod_i f_i
        eq = (_BITS[:, None, :] == _BITS[None, :, :])
        f = np.where(zmask[None, None, :], np.where(eq, 2.0, -1.0), 0.5)
        F = f.prod(axis=2)
        a[si] = F @ (obs_diag * keep)
        b[si] = F @ keep.astype(float)
    w = P / len(settings)
    Ea, Eb = (w * a).sum(), (w * b).sum()
    Vaa = (w * a * a).sum() - Ea**2
    Vbb = (w * b * b).sum() - Eb**2
    Vab = (w * a * b).sum() - Ea * Eb
    Rs = Ea / Eb
    v_shadow = (Vaa - 2 * Rs * Vab + Rs**2 * Vbb) / Eb**2
    return {"kept": kept, "filtered": R, "shadow_mean": Rs, "v_direct": v_direct, "v_shadow": float(v_shadow),
            "ratio": float(v_shadow / v_direct), "all_Z_shadow_shots": 3.0**-N}


def energy_ratio(rho, ham):
    """T7: per-shot variance of the energy, qubit-wise-commuting groups with
    optimal allocation vs shadows."""
    labels = ham.paulis.to_labels()
    coeffs = np.real(ham.coeffs)
    terms = []
    for lab, c in zip(labels, coeffs):
        d = {q: ch for q, ch in enumerate(reversed(lab)) if ch != "I"}
        terms.append((d, c))
    # direct: greedy qubit-wise commuting grouping
    groups = []
    for d, c in sorted(terms, key=lambda t: -abs(t[1])):
        if not d:
            continue
        for g in groups:
            if all(g["basis"].get(q, b) == b for q, b in d.items()):
                g["basis"].update(d)
                g["terms"].append((d, c))
                break
        else:
            groups.append({"basis": dict(d), "terms": [(d, c)]})
    sd = 0.0
    for g in groups:
        s = ["Z"] * N
        for q, b in g["basis"].items():
            s[q] = b
        p = setting_probs(rho, "".join(s))
        val = sum(c * np.prod(_SIGN[:, list(d)], axis=1) for d, c in g["terms"])
        sd += math.sqrt(max(float(p @ val**2 - (p @ val) ** 2), 0.0))
    v_direct = sd**2
    settings, P = all_settings(rho)
    est = np.zeros((len(settings), 1 << N))
    for si, s in enumerate(settings):
        for d, c in terms:
            if not d:
                continue
            if all(s[q] == b for q, b in d.items()):
                est[si] += c * 3 ** len(d) * np.prod(_SIGN[:, list(d)], axis=1)
    w = P / len(settings)
    mean = (w * est).sum()
    v_shadow = float((w * est**2).sum() - mean**2)
    return {"groups": len(groups), "terms": len(terms), "v_direct": v_direct, "v_shadow": v_shadow,
            "ratio": v_shadow / v_direct}


def monte_carlo_check(rho, shots=2000, reps=3000, seed=5):
    """Sampled check of T1 (qubit 0) for both methods."""
    rng = np.random.default_rng(seed)
    pz = setting_probs(rho, "Z" * N)
    p0 = float(pz[_SIGN[:, 0] == 1].sum())  # marginal of qubit 0 in Z (same in any setting of the others)
    d_est, s_est = [], []
    for _ in range(reps):
        x = rng.choice(1 << N, size=shots, p=pz)
        d_est.append(_SIGN[x, 0].mean())
        inz = rng.random(shots) < 1 / 3
        z0 = np.where(rng.random(shots) < p0, 1, -1)
        s_est.append((3 * z0 * inz).mean())
    return float(np.var(s_est) / np.var(d_est))


# --------------------------------------------------------------------- #
# states
# --------------------------------------------------------------------- #
def xxz_state(steps=4, dt=0.25, p2=0.01, gamma=0.005):
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    import xxz_trotter_filter_qg as X

    qc = X.trotter_circuit(N, steps, dt, measure=False)
    sim = AerSimulator(method="density_matrix", noise_model=X.noise_model(p2, gamma, 0.0))
    t = transpile(qc, basis_gates=["cx", "rz", "ry", "x"], optimization_level=0)
    t.save_density_matrix()
    return np.asarray(sim.run(t).result().data()["density_matrix"])


def lih_state():
    import filter_scaling_lih_qg as F
    import lih_parity_verification_qg as V

    ham, circ, n, _ = F._molecule("LiH", 3)
    assert n == N
    return V._noisy_density(circ, "all_to_all"), ham


def main():
    import xxz_trotter_filter_qg as X

    rows = {}
    rho_x = xxz_state()
    rho_l, ham = lih_state()
    for name, rho in (("XXZ (§60)", rho_x), ("LiH (§50)", rho_l)):
        rows[name] = tasks_single(rho)
    imb, Nn = X._tables(N)
    t6 = filtered_ratio(rho_x, imb, Nn == N // 2)
    t7 = energy_ratio(rho_l, ham)
    print("Shot ratio R = shadows / direct for the same error (worst target):")
    for task in [k for k in rows["XXZ (§60)"] if not k.startswith("_")]:
        print(f"  {task:30s}  XXZ {rows['XXZ (§60)'][task]:6.2f}   LiH {rows['LiH (§50)'][task]:6.2f}")
    print(f"  (direct settings for T5: {rows['XXZ (§60)']['_settings_T5']} settings, orthogonal array L18)")
    print(f"  T6 filtered imbalance        XXZ {t6['ratio']:6.1f}   kept {t6['kept']:.3f}, "
          f"estimate {t6['filtered']:+.4f} (shadow mean {t6['shadow_mean']:+.4f}); "
          f"all-Z shadow shots {t6['all_Z_shadow_shots']:.4f}")
    print(f"  T7 LiH energy                 R = {t7['ratio']:.2f}  ({t7['terms']} terms, {t7['groups']} groups)")
    gc = greedy_covering()
    rg = tasks_single(rho_x, gc)["T5 all 2-qubit correlators"]
    print(f"  T5 with a greedy covering array instead ({len(gc)} settings, unbalanced): XXZ R = {rg:.2f}")
    mc = monte_carlo_check(rho_x)
    print(f"\nMonte Carlo check, T1 qubit 0 on XXZ: sampled variance ratio {mc:.2f}")
    return rows, t6, t7, mc


if __name__ == "__main__":
    main()
