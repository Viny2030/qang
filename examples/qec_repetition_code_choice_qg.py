"""
Quantum error correction under T1-biased noise: can the qg witness pick
the right repetition code?

Setting (code capacity: one round of noise on the data qubits, perfect
syndrome extraction and correction). Each qubit suffers amplitude
damping gamma (T1) followed by pure dephasing p (T_phi). Three options
for storing one logical qubit:

  none    a bare qubit
  bit     3-qubit bit-flip code  a|000> + b|111>, corrects one X
  phase   3-qubit phase-flip code a|+++> + b|--->, corrects one Z

Score: logical average infidelity 1 - F_avg, computed exactly from the
Kraus operators (and cross-checked with a Qiskit density-matrix circuit
that extracts the syndrome on two ancillas, reads it as their qg_Z, and
corrects).

Witness. Two single-qubit idle experiments, read in qg units:

  prepare |1>, idle, measure Z:  qg_Z = -1 + 2 gamma             (T1 only)
  prepare |+>, idle, measure X:  qg_X = sqrt(1 - gamma) (1 - 2p)  (T1 and T_phi)

so gamma_hat = (1 + qg_Z)/2 and p_hat = (1 - qg_X / sqrt(1 - gamma_hat))/2.
The policy plugs (gamma_hat, p_hat) into the exact model and picks the
option with the lowest predicted infidelity. Compared with the oracle
(best option in hindsight), with fixed choices, and with a qg_Z-only
witness (the §24 witness, which sees T1 but not dephasing).

Findings (400 random noise instances, gamma and p log-uniform in
[1e-3, 5e-2]; 1,000 shots per witness experiment):

  exact logical infidelity at a few points (none / bit / phase):
    pure T1, gamma = 0.02         0.0067 / 0.0101 / 0.0197
    pure dephasing, p = 0.02      0.0133 / 0.0384 / 0.0008
    gamma = p = 0.02              0.0199 / 0.0474 / 0.0204

  * Neither 3-qubit repetition code helps against T1. Amplitude damping
    has an X+iY jump and a no-jump part that shrinks |1> on every qubit;
    the bit-flip code fixes neither at first order, and it is worse than
    a bare qubit at every point tested. Against dephasing the phase-flip
    code is excellent (17x better at p = 0.02). So the choice is really
    "phase code or no code", and it is set by the ratio p / gamma.
  * The boundary is simply p = gamma: the phase code wins when the
    dephasing per round exceeds the damping (the crossover ratio p/gamma
    is 1.00, 1.02, 1.10 at gamma = 0.002, 0.01, 0.04). In qg units: use
    the phase code when the |+> witness has lost more than the |1>
    witness explains, qg_X < sqrt(1 - gamma_hat) (1 - 2 gamma_hat).

      policy            mean infidelity   regret vs oracle   right choice
      oracle            0.00844           -                  100%
      qg witness        0.00867           +2.7%              85%
      always phase      0.01231           +46%               53%
      always none       0.01258           +49%               48%
      always bit        0.03023           +258%              0%
      qg_Z only         0.01258           +49%               48%

    (The qg_Z-only witness sees T1 and no dephasing, so the model always
    recommends no code: it cannot see the noise the phase code fixes.)
  * The witness needs both Bloch components, qg_Z and qg_X: this is
    T1/T2 characterization written in qg. Its mistakes come from shot
    noise near the boundary, where the options cost almost the same:
    regret 16% / 2.7% / 0.3% / 0.06% and right choice 70% / 85% / 95% /
    97% at 100 / 1,000 / 10,000 / 100,000 shots per witness experiment.
  * The Qiskit circuit (syndrome on two ancillas, coherent correction)
    reproduces the exact numbers to 1e-6.
  * The syndrome ancillas' qg_Z is itself a running witness: in the
    phase code the nontrivial-syndrome rate (1 - qg_Z)/2 is 0.044 at
    gamma = 0.01, p = 0.02, close to the dephasing-only 2p(1-p) = 0.039.

Honest summary: this is not a new code and not an advantage over
standard characterization. It is a correct, cheap decision rule, and it
confirms the known result that repetition codes do not correct
amplitude damping; a T1-tailored code (e.g. the 4-qubit Leung code) is
the next thing to try.
"""

import math
import sys

import numpy as np

I2 = np.eye(2)
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.diag([1.0, -1.0])
H = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)

OPTIONS = ("none", "bit", "phase")


# --------------------------------------------------------------------- #
# noise and codes
# --------------------------------------------------------------------- #
def kraus_t1_tphi(gamma: float, p: float):
    """Amplitude damping gamma followed by dephasing p (Kraus list)."""
    ad = [np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]]),
          np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]])]
    dp = [math.sqrt(1.0 - p) * I2, math.sqrt(p) * Z]
    return [d @ a for d in dp for a in ad]


def _kron(*ms):
    out = np.array([[1.0]])
    for m in ms:
        out = np.kron(out, m)
    return out


def _encoder(kind: str) -> np.ndarray:
    v = np.zeros((8, 2))
    v[0, 0] = v[7, 1] = 1.0
    return _kron(H, H, H) @ v if kind == "phase" else v


def _recovery(kind: str):
    """Syndrome projectors of Z0Z1 and Z1Z2 followed by the X correction
    (conjugated by H^3 for the phase code)."""
    proj = lambda s: 0.5 * (np.eye(8) + s)  # noqa: E731
    zz1, zz2 = _kron(Z, Z, I2), _kron(I2, Z, Z)
    fix = {(0, 0): np.eye(8), (1, 0): _kron(X, I2, I2), (1, 1): _kron(I2, X, I2), (0, 1): _kron(I2, I2, X)}
    h3 = _kron(H, H, H)
    ops = []
    for (a, b), c in fix.items():
        k = c @ proj((-1) ** a * zz1) @ proj((-1) ** b * zz2)
        ops.append(h3 @ k @ h3 if kind == "phase" else k)
    return ops


def logical_infidelity(kind: str, gamma: float, p: float) -> float:
    """1 - average fidelity of the logical channel (exact)."""
    k1 = kraus_t1_tphi(gamma, p)
    if kind == "none":
        fe = sum(abs(np.trace(k)) ** 2 for k in k1) / 4.0
    else:
        v = _encoder(kind)
        dec = v.T
        noise = [_kron(a, b, c) for a in k1 for b in k1 for c in k1]
        fe = sum(abs(np.trace(dec @ r @ e @ v)) ** 2 for r in _recovery(kind) for e in noise) / 4.0
    return float(1.0 - (2.0 * fe + 1.0) / 3.0)


def phase_syndrome_rate(gamma: float, p: float) -> float:
    """P(nontrivial Z0Z1-type syndrome) in the phase code = (1 - qg_Z(ancilla))/2,
    averaged over the logical basis states |+_L>, |-_L>."""
    k1 = kraus_t1_tphi(gamma, p)
    v = _encoder("phase")
    s = _kron(H, H, H) @ _kron(Z, Z, I2) @ _kron(H, H, H)  # X0 X1
    rate = 0.0
    for col in (0, 1):
        psi = v[:, col]
        rho = np.outer(psi, psi)
        noise = [_kron(a, b, c) for a in k1 for b in k1 for c in k1]
        rho = sum(e @ rho @ e.T for e in noise)
        rate += 0.5 * (1.0 - float(np.trace(s @ rho))) / 2.0
    return rate


# --------------------------------------------------------------------- #
# witness and policies
# --------------------------------------------------------------------- #
def witness_qg(gamma: float, p: float):
    """Exact qg_Z of |1> and qg_X of |+> after one idle round."""
    return -1.0 + 2.0 * gamma, math.sqrt(1.0 - gamma) * (1.0 - 2.0 * p)


def sample_witness(gamma: float, p: float, shots: int, rng: np.random.Generator):
    qz, qx = witness_qg(gamma, p)
    kz = rng.binomial(shots, (1.0 + qz) / 2.0)
    kx = rng.binomial(shots, (1.0 + qx) / 2.0)
    return 2.0 * kz / shots - 1.0, 2.0 * kx / shots - 1.0


def estimate_noise(qg_z: float, qg_x: float):
    g = min(max((1.0 + qg_z) / 2.0, 0.0), 0.999)
    p = min(max(0.5 * (1.0 - qg_x / math.sqrt(1.0 - g)), 0.0), 0.5)
    return g, p


def choose(gamma_hat: float, p_hat: float) -> str:
    return min(OPTIONS, key=lambda k: logical_infidelity(k, gamma_hat, p_hat))


POLICIES = ("oracle", "qg witness", "always phase", "always none", "always bit", "qg_Z only")


def policy_table(n_instances: int = 400, shots: int = 1000, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_instances):
        g, p = np.exp(rng.uniform(math.log(1e-3), math.log(5e-2), 2))
        cost = {k: logical_infidelity(k, g, p) for k in OPTIONS}
        best = min(cost, key=cost.get)
        qz, qx = sample_witness(g, p, shots, rng)
        gh, ph = estimate_noise(qz, qx)
        picks = {
            "oracle": best,
            "qg witness": choose(gh, ph),
            "always phase": "phase",
            "always none": "none",
            "always bit": "bit",
            "qg_Z only": choose(gh, 0.0),
        }
        rows.append({k: (cost[v], v == best) for k, v in picks.items()})
    oracle = np.mean([r["oracle"][0] for r in rows])
    out = {}
    for k in POLICIES:
        m = float(np.mean([r[k][0] for r in rows]))
        out[k] = {"mean_infidelity": m, "regret": m / oracle - 1.0,
                  "right_choice": float(np.mean([r[k][1] for r in rows]))}
    return out


def boundary_ratio(gamma: float, lo: float = 1e-5, hi: float = 0.2) -> float:
    """p/gamma at which the phase code and no code cost the same."""
    f = lambda p: logical_infidelity("phase", gamma, p) - logical_infidelity("none", gamma, p)  # noqa: E731
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi) / gamma


# --------------------------------------------------------------------- #
# Qiskit cross-check: ancilla syndrome read as qg_Z, then correction
# --------------------------------------------------------------------- #
def qiskit_logical_infidelity(kind: str, gamma: float, p: float) -> float:
    """Same number as logical_infidelity, from a 5-qubit density-matrix
    circuit: encode, noise, syndrome on ancillas 3-4, coherent correction,
    decode; averaged over the six Pauli eigenstates (a 2-design)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix, Kraus, partial_trace, state_fidelity

    noise = Kraus(kraus_t1_tphi(gamma, p))
    preps = [[], ["x"], ["h"], ["x", "h"], ["h", "s"], ["x", "h", "s"]]
    fids = []
    for prep in preps:
        ref = QuantumCircuit(1)
        for g in prep:
            getattr(ref, g)(0)
        qc = QuantumCircuit(5)
        for g in prep:
            getattr(qc, g)(0)
        if kind != "none":
            qc.cx(0, 1)
            qc.cx(0, 2)
            if kind == "phase":
                qc.h([0, 1, 2])
        for q in ([0] if kind == "none" else [0, 1, 2]):
            qc.append(noise, [q])
        if kind != "none":
            if kind == "phase":
                qc.h([0, 1, 2])
            qc.cx(0, 3)
            qc.cx(1, 3)
            qc.cx(1, 4)
            qc.cx(2, 4)
            # syndrome (a3, a4): (1,0)->X0, (1,1)->X1, (0,1)->X2
            qc.x(4)
            qc.ccx(3, 4, 0)
            qc.x(4)
            qc.ccx(3, 4, 1)
            qc.x(3)
            qc.ccx(3, 4, 2)
            qc.x(3)
            qc.cx(0, 2)
            qc.cx(0, 1)
        rho = partial_trace(DensityMatrix(qc), [1, 2, 3, 4])
        fids.append(state_fidelity(rho, DensityMatrix(ref)))
    return float(1.0 - np.mean(fids))


# --------------------------------------------------------------------- #
def make_figure(path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ps = np.geomspace(1e-3, 5e-2, 40)
    g = 0.01
    colors = {"none": "#8c8c8c", "bit": "#c0392b", "phase": "#1f6fb2"}
    for k in OPTIONS:
        ax[0].plot(ps, [logical_infidelity(k, g, p) for p in ps], color=colors[k], label=k)
    ax[0].axvline(boundary_ratio(g) * g, color="k", ls=":", lw=1)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("dephasing p   (gamma = 0.01 fixed)")
    ax[0].set_ylabel("logical infidelity")
    ax[0].set_title("Which option wins: set by p / gamma")
    ax[0].legend(fontsize=8)

    gg, pp = np.meshgrid(np.geomspace(1e-3, 5e-2, 30), np.geomspace(1e-3, 5e-2, 30))
    best = np.vectorize(lambda a, b: OPTIONS.index(min(OPTIONS, key=lambda k: logical_infidelity(k, a, b))))(gg, pp)
    from matplotlib.colors import ListedColormap

    ax[1].pcolormesh(gg, pp, best, cmap=ListedColormap([colors[k] for k in OPTIONS]), vmin=-0.5, vmax=2.5,
                     alpha=0.5, shading="nearest")
    ax[1].plot([1e-3, 5e-2], [1e-3, 5e-2], "k:", lw=1)
    ax[1].text(2e-3, 1.4e-3, "p = gamma", rotation=33, fontsize=8)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("gamma  (from qg_Z of |1>: (1 + qg_Z)/2)")
    ax[1].set_ylabel("p  (from qg_X of |+>)")
    ax[1].set_title("Best option (grey: no code, blue: phase code)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    for g, p in ((0.02, 0.0), (0.0, 0.02), (0.02, 0.02), (0.01, 0.05), (0.05, 0.005)):
        print(f"gamma={g} p={p}: " + "  ".join(f"{k}={logical_infidelity(k, g, p):.4f}" for k in OPTIONS))
    for g in (0.002, 0.01, 0.04):
        print(f"boundary p/gamma at gamma={g}: {boundary_ratio(g):.2f}")
    print(f"phase-code syndrome rate at gamma=0.01, p=0.02: {phase_syndrome_rate(0.01, 0.02):.4f} "
          f"(2p(1-p) = {2 * 0.02 * 0.98:.4f})")
    try:
        for k in OPTIONS:
            print(f"qiskit cross-check {k}: {qiskit_logical_infidelity(k, 0.02, 0.02):.6f} "
                  f"vs {logical_infidelity(k, 0.02, 0.02):.6f}")
    except ImportError:
        print("(qiskit not installed: skipping the cross-check)")
    t = policy_table()
    for k, v in t.items():
        print(f"{k:13s} mean infidelity {v['mean_infidelity']:.5f}  regret {v['regret']:+.1%}  "
              f"right choice {v['right_choice']:.1%}")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"))
