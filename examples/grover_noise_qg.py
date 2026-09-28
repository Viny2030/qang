"""
Grover search with noise, read in qg: when to stop, and a label-free
estimate of the success probability from the qubits' polar biases.

n search qubits, N = 2^n items, one marked item m. Ideal success after k
iterations: P_k = sin^2((2k+1) asin(1/sqrt N)). Circuits are compiled to
CNOTs (a multi-controlled Z costs 14, 36, 84 CNOTs for n = 4, 5, 6; two
per iteration) and simulated exactly as density matrices with a
two-qubit depolarizing error p2 on every CNOT, optional amplitude damping
gamma on both CNOT qubits, and readout error e.

The qg reading. If the unmarked outcomes are uniform (true for the ideal
algorithm and for depolarizing noise), each qubit's polar bias is

    qg_Z,i = s_i (N P - 1)/(N - 1),     s_i = +1 if m_i = 0, -1 if m_i = 1,

so (i) the SIGN of qg_Z,i reads bit i of the marked item, and (ii) the
MAGNITUDE gives P without knowing m:

    P_hat = (1 + (N - 1) q) / N,   q^2 = mean_i [qg_Z,i^2 - (1 - qg_Z,i^2)/S]

(the second term removes the shot-noise bias of qg^2 with S shots).

Questions:
  A. How far does noise move the best iteration, and what does it cost?
     A verified search (a candidate is checked classically, as in any
     search problem) spends about (k + 1)/P_k oracle calls per success.
  B. Label-free estimates of P_k from a few shots: qg magnitude vs the
     frequency of the most common outcome ("mode"), and the k each picks.
  C. Reading the marked item: per-qubit qg signs vs the most common
     bitstring, against shots.
  D. T1 breaks the uniformity: unmarked outcomes drift towards 0...0, the
     register-mean qg_Z moves up, and the qg reading biases.

Findings (python examples/grover_noise_qg.py):

  This is mostly a NEGATIVE result for qg: the per-qubit reading loses
  to the joint outcome.

  A. Noise moves the best iteration earlier and eats the speed-up (cost =
     oracle calls per verified success, classical exhaustive search
     ~N/2 on average):
       n = 4:  noiseless k = 2, cost 3.3;  p2 = 0.02 -> k = 1, cost 6.5
       n = 5:  noiseless k = 3, cost 4.5;  p2 = 0.002/0.005/0.01 -> k = 2/2/1,
               cost 6.3/8.9/13.3 (classical 16)
       n = 6:  noiseless k = 4, cost 6.1;  p2 = 0.001/0.002/0.005 -> k = 3/2/1,
               cost 10.2/15.0/27.6 (classical 32)
     With two multi-controlled Z per iteration (72 CNOTs at n = 5, 168 at
     n = 6), p2 = 0.005 leaves n = 6 only 14 % cheaper than classical search.
     This is known behaviour (noise shortens the optimal Grover run); the
     numbers are ours.
  B. The qg formula qg_Z,i = s_i (NP - 1)/(N - 1) holds only if the
     unmarked outcomes are uniform. Gate-level depolarizing noise is not
     uniform after the diffusion operator: the formula is off by up to
     0.015 (n = 4) and 0.067 (n = 5). The label-free estimate of P_k from
     qg magnitudes is WORSE than the frequency of the most common outcome
     at every setting tried: RMSE 0.037-0.074 vs 0.015-0.045. Choosing k by
     the smallest estimated cost: with 500 shots per k the qg choice
     costs 0.5-9 % extra, the mode choice 0.6-2.4 %; with 100 shots both
     are poor (qg 5-46 %, mode 1-59 %).
  C. Reading the marked item: per-qubit qg signs need more shots than the
     most common bitstring in every case (n = 5, P = 0.12: 100 shots give
     0.66 vs 0.91; n = 6, P = 0.085: 0.40 vs 0.87). The reason is
     structural: the qg margin is (NP - 1)/(N - 1) ~ P per qubit with
     unit per-shot variance, so the sign test needs ~1/P^2 shots, while
     the marked item stands out among N outcomes after ~a few/P shots.
     Marginals throw away the joint information Grover puts in one
     bitstring.
  D. T1 moves the register-mean qg_Z up (+0.039 at gamma = 0.02 against
     +0.001 for uniform noise at the same P), the §20 witness, but by then
     P is near 1/N; the qg reading breaks first (decode 0.07 vs mode 0.35
     at gamma = 0.01, 100 shots).

  Verdict: for Grover-type algorithms, whose answer is one bitstring,
  qg adds nothing measurable; use the outcome histogram. qg is the right
  readout when the answer is a set of single-qubit or few-body
  expectation values (VQE, dynamics, §60), not when it is a string.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MARKED = {4: 0b1011, 5: 0b10110, 6: 0b101101}


def _mcz(qc, n):
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)


def grover_circuit(n, k, marked=None):
    from qiskit import QuantumCircuit

    m = MARKED[n] if marked is None else marked
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(k):
        zeros = [q for q in range(n) if not (m >> q) & 1]
        if zeros:
            qc.x(zeros)
        _mcz(qc, n)
        if zeros:
            qc.x(zeros)
        qc.h(range(n))
        qc.x(range(n))
        _mcz(qc, n)
        qc.x(range(n))
        qc.h(range(n))
    return qc


def ideal_p(n, k):
    th = math.asin(1 / math.sqrt(2**n))
    return math.sin((2 * k + 1) * th) ** 2


_CACHE = {}


def probabilities(n, k, p2=0.0, gamma=0.0, e=0.0):
    key = (n, k, p2, gamma, e)
    if key in _CACHE:
        return _CACHE[key]
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, amplitude_damping_error, depolarizing_error

    t = transpile(grover_circuit(n, k), basis_gates=["cx", "rz", "sx", "x"], optimization_level=1,
                  seed_transpiler=1)
    nm = NoiseModel()
    err = depolarizing_error(p2, 2)
    if gamma > 0:
        ad = amplitude_damping_error(gamma)
        err = err.compose(ad.tensor(ad))
    if p2 > 0 or gamma > 0:
        nm.add_all_qubit_quantum_error(err, ["cx"])
    t.save_probabilities()
    p = np.asarray(AerSimulator(method="density_matrix", noise_model=nm).run(t).result().data()["probabilities"])
    if e > 0:
        p = p.reshape([2] * n)
        mtx = np.array([[1 - e, e], [e, 1 - e]])
        for ax in range(n):
            p = np.moveaxis(np.tensordot(mtx, p, axes=([1], [ax])), 0, ax)
        p = p.reshape(-1)
    _CACHE[key] = p
    return p


def cnots(n, k):
    from qiskit import transpile

    t = transpile(grover_circuit(n, k), basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=1)
    return t.count_ops().get("cx", 0)


def _z(n):
    idx = np.arange(2**n)
    return 1 - 2 * ((idx[:, None] >> np.arange(n)) & 1)


def qg_from_p(p, n):
    return _z(n).T @ p


def p_hat_qg(counts, n):
    """Label-free success estimate from the qubits' polar biases."""
    S = counts.sum()
    qg = _z(n).T @ counts / S
    q2 = np.mean(qg**2 - (1 - qg**2) / S)
    q = math.sqrt(max(q2, 0.0))
    return (1 + (2**n - 1) * q) / 2**n


def p_hat_mode(counts):
    return counts.max() / counts.sum()


def decode_qg(counts, n):
    qg = _z(n).T @ counts
    return int(sum(1 << i for i in range(n) if qg[i] < 0))


def decode_mode(counts):
    return int(np.argmax(counts))


def cost(n, k, P):
    """Oracle calls per verified success (k Grover calls + 1 check)."""
    return (k + 1) / max(P, 1e-12)


# --------------------------------------------------------------------- #
def study_a(n, p2, e=0.0, gamma=0.0, kmax=None):
    kmax = kmax or int(round(math.pi / 4 * math.sqrt(2**n))) + 2
    rows = []
    for k in range(0, kmax + 1):
        P = float(probabilities(n, k, p2, gamma, e)[MARKED[n]])
        rows.append((k, ideal_p(n, k), P, cost(n, k, P)))
    return rows


def study_b(n, p2, shots, reps, rng, e=0.0, gamma=0.0, kmax=None):
    """Label-free choice of k: pick the k with the smallest estimated cost.
    Returns RMSE of each estimator (over k) and the extra cost (regret) of
    the chosen k relative to the true best."""
    rows = study_a(n, p2, e, gamma, kmax)
    truth = np.array([r[2] for r in rows])
    true_cost = np.array([r[3] for r in rows])
    best = true_cost.min()
    err_q, err_m, reg_q, reg_m = [], [], [], []
    for _ in range(reps):
        eq, em = [], []
        for k, _, P, _ in rows:
            p = probabilities(n, k, p2, gamma, e)
            c = rng.multinomial(shots, p / p.sum())
            eq.append(p_hat_qg(c, n))
            em.append(p_hat_mode(c))
        eq, em = np.array(eq), np.array(em)
        err_q.append(eq - truth)
        err_m.append(em - truth)
        ks = np.arange(len(rows))
        reg_q.append(true_cost[np.argmin((ks + 1) / np.maximum(eq, 1e-9))] / best - 1)
        reg_m.append(true_cost[np.argmin((ks + 1) / np.maximum(em, 1e-9))] / best - 1)
    err_q, err_m = np.array(err_q), np.array(err_m)
    return {
        "rmse_qg": float(np.sqrt((err_q**2).mean())),
        "rmse_mode": float(np.sqrt((err_m**2).mean())),
        "bias_qg_small_P": float(err_q[:, truth < 2 / 2**n].mean()) if (truth < 2 / 2**n).any() else float("nan"),
        "bias_mode_small_P": float(err_m[:, truth < 2 / 2**n].mean()) if (truth < 2 / 2**n).any() else float("nan"),
        "regret_qg": float(np.mean(reg_q)),
        "regret_mode": float(np.mean(reg_m)),
    }


def study_c(n, k, p2, shots_list, reps, rng, e=0.0, gamma=0.0):
    p = probabilities(n, k, p2, gamma, e)
    out = []
    for S in shots_list:
        okq = okm = 0
        for _ in range(reps):
            c = rng.multinomial(S, p / p.sum())
            okq += decode_qg(c, n) == MARKED[n]
            okm += decode_mode(c) == MARKED[n]
        out.append((S, okq / reps, okm / reps))
    return float(p[MARKED[n]]), out


def make_figure(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    n = 5
    for p2, c in ((0.0, "k"), (0.002, "#1f6fb2"), (0.005, "#e0a030"), (0.01, "#8c2d04")):
        rows = study_a(n, p2, kmax=8)
        ax1.plot([r[0] for r in rows], [r[2] for r in rows], "o-", color=c, label=f"p2 = {p2}")
    ax1.axhline(1 / 2**n, color="grey", ls=":", lw=1, label="random guess 1/N")
    ax1.set_xlabel("Grover iterations k")
    ax1.set_ylabel("success probability P_k")
    ax1.set_title("n = 5: noise lowers and moves the peak", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    rng = np.random.default_rng(2)
    shots = [5, 10, 20, 50, 100, 200, 500]
    for (nn, k, p2), c in (((5, 4, 0.01), "#e0a030"), ((6, 3, 0.005), "#8c2d04")):
        P, rows = study_c(nn, k, p2, shots, 600, rng)
        ax2.plot(shots, [r[2] for r in rows], "o-", color=c, label=f"most common bitstring, n={nn} (P={P:.2f})")
        ax2.plot(shots, [r[1] for r in rows], "s--", color=c, label=f"qg signs per qubit, n={nn}")
    ax2.set_xscale("log")
    ax2.set_xlabel("shots")
    ax2.set_ylabel("probability of reading the marked item")
    ax2.set_title("Per-qubit qg signs lose to the joint outcome", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(figure=False):
    rng = np.random.default_rng(1)
    # exact check of the qg formula
    for n, k, p2 in ((4, 2, 0.005), (5, 3, 0.01)):
        p = probabilities(n, k, p2)
        P = p[MARKED[n]]
        s = np.array([1 - 2 * ((MARKED[n] >> i) & 1) for i in range(n)])
        pred = s * (2**n * P - 1) / (2**n - 1)
        print(f"check n={n} k={k} p2={p2}: max |qg - formula| = {np.abs(qg_from_p(p, n) - pred).max():.1e}")
    print("\nA. success and cost per verified success (oracle calls), best k marked *")
    for n, p2s in ((4, (0.0, 0.005, 0.01, 0.02)), (5, (0.0, 0.002, 0.005, 0.01)), (6, (0.0, 0.001, 0.002, 0.005))):
        print(f"  n = {n} (N = {2**n}), CNOTs per iteration {cnots(n, 1) - cnots(n, 0)}")
        for p2 in p2s:
            rows = study_a(n, p2)
            kb = min(rows, key=lambda r: r[3])
            ideal_best = min(study_a(n, 0.0), key=lambda r: r[3])
            txt = " ".join(f"{r[2]:.3f}" for r in rows)
            print(f"    p2 = {p2:<6} P_k(k=0..) {txt}   best k = {kb[0]}, cost {kb[3]:.1f}"
                  f" (classical ~{2**n / 2:.0f}, noiseless {ideal_best[3]:.1f})")
    print("\nB. label-free estimate of P_k and choice of k (shots per k, 300 repetitions)")
    for n, p2 in ((4, 0.01), (5, 0.005), (5, 0.01)):
        for S in (100, 500):
            r = study_b(n, p2, S, 300, rng)
            print(f"  n={n} p2={p2} S={S}: RMSE qg {r['rmse_qg']:.4f} mode {r['rmse_mode']:.4f} | bias where P<2/N:"
                  f" qg {r['bias_qg_small_P']:+.4f} mode {r['bias_mode_small_P']:+.4f} |"
                  f" extra cost of chosen k: qg {r['regret_qg']:.1%} mode {r['regret_mode']:.1%}")
    print("\nC. reading the marked item: success rate vs shots")
    for n, k, p2 in ((5, 4, 0.005), (5, 4, 0.01), (5, 2, 0.02), (6, 3, 0.005)):
        P, rows = study_c(n, k, p2, (10, 30, 100, 300), 1000, rng)
        print(f"  n={n} k={k} p2={p2} (P = {P:.3f}, 1/N = {1 / 2**n:.3f}): "
              + "  ".join(f"S={S}: qg {a:.2f} mode {b:.2f}" for S, a, b in rows))
    print("\nD. T1 (n = 5, p2 = 0.005): register-mean qg_Z and the qg reading")
    for gamma in (0.0, 0.005, 0.01, 0.02):
        rows = study_a(5, 0.005, gamma=gamma, kmax=6)
        est = []
        for k, _, P, _ in rows:
            p = probabilities(5, k, 0.005, gamma)
            est.append(p_hat_qg(np.round(p * 1e6), 5))
        k = 3
        p = probabilities(5, k, 0.005, gamma)
        mean_qg = float(qg_from_p(p, 5).mean())
        s = np.array([1 - 2 * ((MARKED[5] >> i) & 1) for i in range(5)])
        mean_qg_ideal_sign = float((s * (32 * p[MARKED[5]] - 1) / 31).mean())
        P_hat_err = max(abs(a - r[2]) for a, r in zip(est, rows))
        pk, rowsc = study_c(5, k, 0.005, (100,), 1000, rng, gamma=gamma)
        print(f"  gamma {gamma}: k=3 P {pk:.3f}; register-mean qg_Z {mean_qg:+.3f} (uniform-noise value {mean_qg_ideal_sign:+.3f});"
              f" max |P_hat - P| over k {P_hat_err:.3f}; decode at 100 shots qg {rowsc[0][1]:.2f} mode {rowsc[0][2]:.2f}")
    if figure:
        make_figure(__file__.replace(".py", ".png"))


if __name__ == "__main__":
    main("--figure" in sys.argv)
