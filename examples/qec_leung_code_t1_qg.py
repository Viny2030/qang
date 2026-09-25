"""
A code built for T1: the 4-qubit Leung code vs repetition codes, and a
qg witness that decides when to use it.

§27 showed that neither 3-qubit repetition code corrects amplitude
damping: the real choice there was "phase code or no code". The 4-qubit
code of Leung, Nielsen, Chuang and Yamamoto [PRA 56, 2567 (1997)],

    |0_L> = (|0000> + |1111>)/sqrt 2,   |1_L> = (|0011> + |1100>)/sqrt 2,

is an approximate code for amplitude damping: it corrects the no-jump
distortion and any single jump to first order, so its logical infidelity
is O(gamma^2) instead of O(gamma). It does NOT correct phase flips (Z_1
and Z_3 act identically on the code up to a logical Z).

Setting (as §27): one round of amplitude damping gamma and then
dephasing p on every data qubit, perfect recovery; score = logical
average infidelity, exact from Kraus operators. Recovery for the Leung
code: the channel-adapted "polar" recovery for the error set {no jump,
jump on qubit j}: for each error E_j the code image E_j V is mapped back
by the partial isometry V W_j^dagger, W_j = E_j V (V^dag E_j^dag E_j V)^(-1/2)
(the five images are mutually orthogonal); the rest of the space is
sent to |0_L>. The transpose-channel (Petz) recovery is computed as a
cross-check. A Qiskit Aer density-matrix circuit reproduces the numbers.

Witness and policy: exactly §27's two idle experiments read in qg,
qg_Z(|1>) = -1 + 2 gamma and qg_X(|+>) = sqrt(1 - gamma)(1 - 2p), give
gamma_hat and p_hat; the policy picks the option with the lowest
predicted infidelity among {no code, phase code, Leung code}.

Findings:

  * Pure T1: the Leung code is the only option that corrects it. Its
    infidelity is ~0.92 gamma^2 (9.2e-7, 9.2e-5, 8.2e-4 at gamma =
    0.001, 0.01, 0.03) against gamma/3 for no code: 36x better at
    gamma = 0.01, 360x at 0.001. It beats no code up to gamma = 0.44.
    The Petz recovery is ~1.3x worse than the channel-adapted one.
  * It is fragile to dephasing, which it cannot correct: it beats no
    code only for p < gamma/4 (boundary p/gamma = 0.249, 0.247, 0.237
    at gamma = 0.002, 0.01, 0.04). At gamma = 0.02 the advantage goes
    from 18x (p = 0) to 1.9x (p = 0.002) to none (p = 0.005).
  * In device terms (one idle round, gamma = t/T1, p = t/2T_phi) this
    is a rule on T2/T1 alone:  T2 > T1 -> Leung code;
    0.4 T1 < T2 < T1 -> no code;  T2 < 0.4 T1 -> phase-flip code.
    (p = gamma/4 <=> T_phi = 2 T1 <=> T2 = T1;  p = gamma <=> T2 = 0.4 T1.)
    A typical transmon with T1 = 100 us, T2 = 70 us (§31) sits in the
    "no code" band.
  * Policy (300 instances, gamma and p log-uniform in [1e-3, 5e-2]): the
    oracle picks Leung in 21% of cases. The qg witness policy with the
    Leung option has regret +6.4% (right 79%, 1000 shots), vs +14.7% for
    the §27 policy (no code / phase code only), +64-66% for "always
    none/phase" and +220% for "always Leung". Regret +30.7%, +0.7%,
    +0.1% at 100, 10,000 and 100,000 shots.

Honest scope: code-capacity model (noiseless encoding, syndrome
extraction and recovery), one round. The Leung code is known; the
contribution is the decision rule written in qg (and equivalently in
T2/T1), and its cost-benefit against the simpler options.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qec_repetition_code_choice_qg import (  # noqa: E402
    estimate_noise,
    kraus_t1_tphi,
    logical_infidelity as repetition_infidelity,
    sample_witness,
)

OPTIONS = ("none", "phase", "leung")


def leung_encoder() -> np.ndarray:
    v = np.zeros((16, 2))
    v[0b0000, 0] = v[0b1111, 0] = 1 / math.sqrt(2)
    v[0b0011, 1] = v[0b1100, 1] = 1 / math.sqrt(2)
    return v


def _kron(*ms):
    out = np.array([[1.0]])
    for m in ms:
        out = np.kron(out, m)
    return out


def _noise_kraus(gamma, p, n=4):
    k1 = kraus_t1_tphi(gamma, p)
    ops = [np.array([[1.0]])]
    for _ in range(n):
        ops = [np.kron(o, k) for o in ops for k in k1]
    return ops


def _inv_sqrt_psd(m, tol=1e-12):
    w, u = np.linalg.eigh(m)
    inv = np.where(w > tol, 1 / np.sqrt(np.clip(w, tol, None)), 0.0)
    return (u * inv) @ u.conj().T


def _complete(kraus, v):
    """Add Kraus operators sending the unused subspace to |0_L>."""
    s = sum(k.conj().T @ k for k in kraus)
    rest = np.eye(s.shape[0]) - s
    w, u = np.linalg.eigh((rest + rest.conj().T) / 2)
    for val, vec in zip(w, u.T):
        if val > 1e-10:
            kraus.append(math.sqrt(val) * np.outer(v[:, 0], vec.conj()))
    return kraus


def polar_recovery(gamma_design: float):
    """Channel-adapted recovery for {no jump, one jump on qubit j} of
    amplitude damping with strength gamma_design."""
    v = leung_encoder()
    a0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1 - gamma_design)]])
    a1 = np.array([[0.0, math.sqrt(gamma_design)], [0.0, 0.0]])
    errors = [_kron(a0, a0, a0, a0)]
    for j in range(4):
        ms = [a0] * 4
        ms[j] = a1
        errors.append(_kron(*ms))
    kraus = []
    for e in errors:
        ev = e @ v
        w = ev @ _inv_sqrt_psd(ev.conj().T @ ev)
        kraus.append(v @ w.conj().T)
    return _complete(kraus, v)


def petz_recovery(gamma: float, p: float):
    """Transpose-channel recovery R(x) = P E^dag N(P)^(-1/2) x N(P)^(-1/2) E P."""
    v = leung_encoder()
    proj = v @ v.T
    noise = _noise_kraus(gamma, p)
    npj = sum(e @ proj @ e.conj().T for e in noise)
    s = _inv_sqrt_psd(npj)
    kraus = [proj @ e.conj().T @ s for e in noise]
    kraus = [k for k in kraus if np.abs(k).max() > 1e-14]
    return _complete(kraus, v)


def logical_channel_fe(recovery, gamma, p):
    """Entanglement fidelity of decode o recovery o noise o encode."""
    v = leung_encoder()
    noise = _noise_kraus(gamma, p)
    fe = 0.0
    for r in recovery:
        rv = v.conj().T @ r
        for e in noise:
            fe += abs(np.trace(rv @ e @ v)) ** 2
    return fe / 4.0


def leung_infidelity(gamma: float, p: float, recovery="polar") -> float:
    rec = polar_recovery(max(gamma, 1e-9)) if recovery == "polar" else petz_recovery(gamma, p)
    fe = logical_channel_fe(rec, gamma, p)
    return float(1.0 - (2.0 * fe + 1.0) / 3.0)


def infidelity(kind: str, gamma: float, p: float) -> float:
    if kind == "leung":
        return leung_infidelity(gamma, p)
    return repetition_infidelity(kind, gamma, p)


def gamma_scaling(gammas=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1)):
    """Pure T1: (gamma, none, bit, phase, leung polar, leung Petz)."""
    return [(g, infidelity("none", g, 0), repetition_infidelity("bit", g, 0), infidelity("phase", g, 0),
             leung_infidelity(g, 0), leung_infidelity(g, 0, "petz")) for g in gammas]


def crossover_gamma(lo=1e-4, hi=0.9):
    """Pure T1: gamma above which the Leung code is worse than no code."""
    f = lambda g: leung_infidelity(g, 0) - infidelity("none", g, 0)  # noqa: E731
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def leung_boundary_ratio(gamma, lo=1e-7, hi=0.5):
    """p/gamma at which the Leung code and no code cost the same."""
    f = lambda p: leung_infidelity(gamma, p) - infidelity("none", gamma, p)  # noqa: E731
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi) / gamma


def rates_from_times(t_over_t1, t2_over_t1):
    """(gamma, p) of one idle round of length t for given T1, T2:
    gamma = 1 - exp(-t/T1), 1 - 2p = exp(-t/T_phi), 1/T_phi = 1/T2 - 1/(2 T1)."""
    gamma = 1 - math.exp(-t_over_t1)
    inv_tphi = 1 / t2_over_t1 - 0.5  # in units of 1/T1
    p = 0.5 * (1 - math.exp(-t_over_t1 * inv_tphi))
    return gamma, p


def best_by_t2_ratio(t_over_t1=0.01, ratios=(0.3, 0.35, 0.45, 0.8, 0.95, 1.05, 1.2, 1.6, 1.9)):
    return [(r, min(OPTIONS, key=lambda k: infidelity(k, *rates_from_times(t_over_t1, r)))) for r in ratios]


def best_option_map(gammas, ps):
    return [[min(OPTIONS, key=lambda k: infidelity(k, g, p)) for g in gammas] for p in ps]


def choose(gamma_hat, p_hat, options=OPTIONS):
    return min(options, key=lambda k: infidelity(k, gamma_hat, p_hat))


POLICIES = ("oracle", "qg witness (none/phase/leung)", "§27 policy (none/phase)", "always leung",
            "always phase", "always none")


def policy_table(n_instances=300, shots=1000, seed=5):
    """gamma and p log-uniform in [1e-3, 5e-2] (as §27)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_instances):
        g, p = np.exp(rng.uniform(math.log(1e-3), math.log(5e-2), 2))
        cost = {k: infidelity(k, g, p) for k in OPTIONS}
        best = min(cost, key=cost.get)
        gh, ph = estimate_noise(*sample_witness(g, p, shots, rng))
        picks = {"oracle": best, "qg witness (none/phase/leung)": choose(gh, ph),
                 "§27 policy (none/phase)": choose(gh, ph, ("none", "phase")),
                 "always leung": "leung", "always phase": "phase", "always none": "none"}
        rows.append({k: (cost[v], v == best, v) for k, v in picks.items()})
    oracle = np.mean([r["oracle"][0] for r in rows])
    out = {}
    for k in POLICIES:
        m = float(np.mean([r[k][0] for r in rows]))
        out[k] = {"mean_infidelity": m, "regret": m / oracle - 1.0,
                  "right_choice": float(np.mean([r[k][1] for r in rows]))}
    out["oracle_share"] = {o: float(np.mean([r["oracle"][2] == o for r in rows])) for o in OPTIONS}
    return out


# --------------------------------------------------------------------- #
# Qiskit cross-check
# --------------------------------------------------------------------- #
def qiskit_leung_infidelity(gamma: float, p: float) -> float:
    """Encode the two halves of a Bell pair with a unitary, apply the noise
    with Aer amplitude/phase damping errors, recover with the Kraus map,
    decode and read the entanglement fidelity from the density matrix."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix, Kraus, Operator, partial_trace
    from qiskit_aer.noise import amplitude_damping_error, phase_damping_error

    v = leung_encoder()
    # unitary with columns |0_L>, |1_L> on inputs |0000>, |0001> (qubit 0 = logical input)
    basis = [v[:, 0], v[:, 1]]
    for k in range(16):
        e = np.zeros(16)
        e[k] = 1.0
        for b in basis:
            e = e - np.dot(b, e) * b
        if np.linalg.norm(e) > 1e-9:
            basis.append(e / np.linalg.norm(e))
        if len(basis) == 16:
            break
    enc = np.column_stack(basis)
    # reorder qiskit little-endian: logical qubit = qubit 0 of the 4 code qubits
    qc = QuantumCircuit(5)  # qubit 4 = reference
    qc.h(4)
    qc.cx(4, 0)
    qc.unitary(Operator(enc), [0, 1, 2, 3])
    noise = amplitude_damping_error(gamma)
    if p > 0:
        noise = noise.compose(phase_damping_error(1 - (1 - 2 * p) ** 2))
    channel = Kraus(noise.to_quantumchannel())
    for q in range(4):
        qc.append(channel, [q])
    qc.append(Kraus(polar_recovery(max(gamma, 1e-9))), [0, 1, 2, 3])
    qc.unitary(Operator(enc).adjoint(), [0, 1, 2, 3])
    rho = DensityMatrix(qc)
    red = partial_trace(rho, [1, 2, 3])
    bell = np.zeros(4)
    bell[0] = bell[3] = 1 / math.sqrt(2)
    fe = float(np.real(bell @ red.data @ bell))
    return float(1.0 - (2.0 * fe + 1.0) / 3.0)


def make_figure(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    gs = np.logspace(-3, -0.5, 25)
    style = {"none": ("#8c8c8c", "no code"), "bit": ("#e0a030", "bit-flip code"),
             "phase": ("#2e8b57", "phase-flip code"), "leung": ("#1f6fb2", "Leung [[4,1]] code")}
    for k, (c, lab) in style.items():
        f = (lambda g: repetition_infidelity("bit", g, 0)) if k == "bit" else (lambda g, k=k: infidelity(k, g, 0))
        ax1.loglog(gs, [f(g) for g in gs], color=c, lw=2, label=lab)
    ax1.set_xlabel("amplitude damping gamma (no dephasing)")
    ax1.set_ylabel("logical infidelity")
    ax1.set_title("Pure T1: only the Leung code corrects it (O(gamma^2))", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")
    g2 = np.logspace(-3, math.log10(5e-2), 28)
    p2 = np.logspace(-3.5, math.log10(5e-2), 28)
    idx = {"none": 0, "phase": 1, "leung": 2}
    grid = np.array([[idx[min(OPTIONS, key=lambda k: infidelity(k, g, p))] for g in g2] for p in p2])
    ax2.pcolormesh(g2, p2, grid, cmap=ListedColormap(["#d9d9d9", "#a8dcb9", "#9ecae1"]), shading="nearest")
    ax2.plot(g2, g2 / 4, "k--", lw=1, label="p = gamma / 4")
    ax2.plot(g2, g2, "k:", lw=1, label="p = gamma")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("gamma (T1), from qg_Z of |1>")
    ax2.set_ylabel("dephasing p, from qg_X of |+>")
    ax2.set_title("Best option: Leung (blue), no code (gray), phase code (green)", fontsize=9)
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    print("Pure T1: logical infidelity")
    print(f"{'gamma':>7} {'none':>9} {'bit':>9} {'phase':>9} {'leung':>9} {'leung Petz':>11}")
    for row in gamma_scaling():
        print(f"{row[0]:7.3f} " + " ".join(f"{x:9.2e}" for x in row[1:5]) + f" {row[5]:11.2e}")
    print(f"Leung beats no code for gamma < {crossover_gamma():.3f}")
    print("\nWith dephasing (gamma = 0.02):")
    for p in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05):
        print(f"  p = {p:.3f}: " + "  ".join(f"{k} {infidelity(k, 0.02, p):.2e}" for k in OPTIONS))
    print("Leung = no code at p/gamma = " + ", ".join(
        f"{leung_boundary_ratio(g):.3f} (gamma {g})" for g in (0.002, 0.01, 0.04)))
    print("Best option by T2/T1 (idle round t = 0.01 T1): "
          + ", ".join(f"{r}: {o}" for r, o in best_by_t2_ratio()))
    print("\nBest option map (rows p, columns gamma):")
    gs = [1e-3, 3e-3, 1e-2, 3e-2, 5e-2]
    ps = [1e-3, 3e-3, 1e-2, 3e-2, 5e-2]
    print("        " + " ".join(f"{g:>7.3f}" for g in gs))
    for p, row in zip(ps, best_option_map(gs, ps)):
        print(f"p={p:5.3f} " + " ".join(f"{o:>7s}" for o in row))
    print("\nPolicies (300 instances, gamma, p log-uniform in [1e-3, 5e-2], 1000 shots per witness):")
    t = policy_table()
    for k in POLICIES:
        r = t[k]
        print(f"  {k:32s} mean {r['mean_infidelity']:.5f}  regret {100 * r['regret']:+6.1f}%  "
              f"right {100 * r['right_choice']:.0f}%")
    print("  oracle picks:", {k: f"{100 * v:.0f}%" for k, v in t["oracle_share"].items()})
    print("\nqg witness policy vs shots per witness experiment:")
    for shots in (100, 1000, 10000, 100000):
        r = policy_table(shots=shots)["qg witness (none/phase/leung)"]
        print(f"  {shots:6d} shots: regret {100 * r['regret']:+5.1f}%, right {100 * r['right_choice']:.0f}%")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"))
