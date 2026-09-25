"""
Constrained optimization with QAOA: "choose exactly K", with and without
qg.

Problem: maximum K-vertex cover. On a graph with n vertices choose
exactly K of them so that as many edges as possible touch a chosen
vertex. With x_i in {0, 1} (1 = chosen) and x_i = (1 - Z_i)/2,

    cover(x) = sum_{(i,j) in E} (x_i + x_j - x_i x_j),   sum_i x_i = K.

In qg units the constraint is a statement about the register mean qg_Z:

    sum_i x_i = K   <=>   mean qg_Z = 1 - 2K/n     (0.25 for n = 8, K = 3)

-- the same identity as the electron number in chemistry (§21). It gives
the three qg ingredients tested here:

  * qg filter: keep only shots with Hamming weight K (per-shot
    qg_Z = 1 - 2K/n).
  * qg budget start: prepare every qubit at qg = 1 - 2K/n (Ry(theta)
    with cos theta = 1 - 2K/n), so the initial state already satisfies
    the constraint on average; mixer rotated about that axis (Egger et
    al., Quantum 5, 479 (2021)). No classical pre-solve needed.
  * qg warm start: qg_i = 1 - 2 c_i from the LP relaxation c of the
    problem, clipped to |qg_i| <= 0.5 (Egger's epsilon = 0.25, i.e. pole
    damping in qg).

QAOA variants (depth p, angles optimized noiselessly, then run noisy):

  standard   |+>^n start, X mixer, cost - lambda (sum x - K)^2
  qg budget  as standard, qg-budget start and rotated mixer
  qg warm    as standard, LP warm start and rotated mixer
  XY         constraint-preserving: Dicke-state start (all weight-K
             strings), XY ring mixer,
             cost without penalty (Hadfield et al. 2019)

Classical baselines: brute force (exact), greedy, and uniform random
sampling of feasible sets.

Findings (n = 8, K = 3, 5 random graphs G(8, 1/2), 4000 shots; P(opt)
= probability that one shot is an optimal cover; random feasible guess:
P(opt) 0.096, ratio 0.772; classical greedy: ratio 0.985, optimal on 4
of 5 graphs; brute force: 56 feasible sets, instant):

  all-to-all device (0.6% depolarizing per CX), P(opt) noiseless / noisy
  / noisy + qg filter:
    p = 1  standard   0.053 / 0.046 / 0.101     (56 CX)
           qg budget  0.066 / 0.058 / 0.108
           qg warm    0.195 / 0.155 / 0.297
           XY         0.346 / 0.150 / 0.278     (179 CX)
    p = 2  standard   0.099 / 0.079 / 0.176     (112 CX)
           qg budget  0.087 / 0.062 / 0.136
           qg warm    0.169 / 0.110 / 0.256
           XY         0.511 / 0.177 / 0.366     (221 CX)
  fake_brisbane (heavy-hex routing, 134-497 ECR): after the filter every
  variant is at the random-feasible level (P(opt) 0.09-0.12, ratio
  0.77-0.79) except the warm start at p = 1 (0.173, 0.813).

  * The qg filter is free and never hurts. With the constraint-preserving
    XY mixer, whose ideal output is 100% feasible, the kept fraction is
    a pure error witness (0.54 at p = 1) and the filter doubles P(opt)
    under noise (0.150 -> 0.278; 0.177 -> 0.366 at p = 2).
  * Penalty QAOA at p <= 2 is no better than guessing: after filtering,
    standard QAOA gives P(opt) 0.10-0.18 and ratio 0.78-0.79, the level
    of a random feasible set. The qg budget start raises the raw
    feasibility (kept 0.46 -> 0.54, raw ratio 0.354 -> 0.422 at p = 1)
    but not the quality of the feasible shots.
  * The warm start is the best penalty variant (3x the random guess),
    but its LP relaxation is already integral, i.e. already the answer,
    on 3 of 5 graphs: the value comes from the classical pre-solve.
  * The witness says when nothing is left. On fake_brisbane the kept
    fraction falls to 0.24-0.33, close to 56/256 = 0.22, the value of a
    fully mixed register: the filtered samples are then random feasible
    sets, as measured.
  * Classical wins: greedy reaches 98.5% of the optimum and brute force
    is instant at n = 8. No variant reaches the greedy ratio. The qg
    ingredients improve QAOA relative to QAOA, not relative to classical
    optimization.
"""

import os
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XXPlusYYGate
from scipy.optimize import linprog, minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_NODES, K_CHOSEN = 8, 3
IDX = np.arange(2**N_NODES)
BITS = (IDX[:, None] >> np.arange(N_NODES)) & 1  # x_i = bit i
WEIGHT = BITS.sum(1)
FEASIBLE = WEIGHT == K_CHOSEN
QG_BUDGET = 1.0 - 2.0 * K_CHOSEN / N_NODES
VARIANTS = ("standard", "qg_budget", "qg_warm", "xy")


# --------------------------------------------------------------------- #
# Problem and classical baselines
# --------------------------------------------------------------------- #
def random_graph(seed, n=N_NODES, p_edge=0.5):
    rng = np.random.default_rng(seed)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p_edge]


def cover_values(edges):
    """cover(x) for every bit string (vector of length 2^n)."""
    c = np.zeros(2**N_NODES)
    for i, j in edges:
        c += BITS[:, i] | BITS[:, j]
    return c


def brute_force(edges):
    c = cover_values(edges)
    best = c[FEASIBLE].max()
    return float(best), np.nonzero(FEASIBLE & (c == best))[0]


def greedy(edges):
    chosen, left = set(), set(edges)
    for _ in range(K_CHOSEN):
        gain = {v: sum(1 for e in left if v in e) for v in range(N_NODES) if v not in chosen}
        v = max(gain, key=lambda u: (gain[u], -u))
        chosen.add(v)
        left = {e for e in left if v not in e}
    return float(len(edges) - len(left))


def lp_relaxation(edges):
    """max sum y_e, y_e <= x_i + x_j, y_e <= 1, sum x = K, 0 <= x <= 1."""
    n, m = N_NODES, len(edges)
    cost = np.concatenate([np.zeros(n), -np.ones(m)])
    a_ub = np.zeros((m, n + m))
    for k, (i, j) in enumerate(edges):
        a_ub[k, [i, j]] = -1.0
        a_ub[k, n + k] = 1.0
    a_eq = np.concatenate([np.ones(n), np.zeros(m)])[None]
    res = linprog(cost, A_ub=a_ub, b_ub=np.zeros(m), A_eq=a_eq, b_eq=[K_CHOSEN],
                  bounds=[(0, 1)] * (n + m), method="highs")
    return res.x[:n]


def start_qg(variant, edges):
    """Per-qubit qg of the initial product state (None for XY)."""
    if variant == "standard":
        return np.zeros(N_NODES)
    if variant == "qg_budget":
        return np.full(N_NODES, QG_BUDGET)
    if variant == "qg_warm":
        return np.clip(1.0 - 2.0 * lp_relaxation(edges), -0.5, 0.5)
    return None


def dicke_state(qc: QuantumCircuit, n=N_NODES, k=K_CHOSEN):
    """Uniform superposition of all weight-k strings (Baertschi &
    Eidenbenz 2019, split-and-cyclic-shift construction)."""
    from qiskit.circuit.library import RYGate

    for i in range(n - k, n):
        qc.x(i)

    def scs(m, kk):  # paper qubit i -> qubit i - 1
        qc.cx(m - 2, m - 1)
        qc.cry(2 * np.arccos(np.sqrt(1 / m)), m - 1, m - 2)
        qc.cx(m - 2, m - 1)
        for l in range(2, kk + 1):
            qc.cx(m - l - 1, m - 1)
            qc.append(RYGate(2 * np.arccos(np.sqrt(l / m))).control(2, annotated=False), [m - 1, m - l, m - l - 1])
            qc.cx(m - l - 1, m - 1)

    for m in range(n, k, -1):
        scs(m, k)
    for m in range(k, 1, -1):
        scs(m, m - 1)


def penalty_weight(edges):
    deg = np.bincount(np.array(edges).ravel(), minlength=N_NODES)
    return float(deg.max())


def objective_values(edges, variant):
    """Values F(x) to MINIMIZE (diagonal of the cost Hamiltonian)."""
    f = -cover_values(edges)
    if variant != "xy":
        f = f + penalty_weight(edges) * (WEIGHT - K_CHOSEN) ** 2
    return f


def ising_terms(f):
    """F = c0 + sum h_i Z_i + sum J_ij Z_i Z_j (exact for quadratic F)."""
    z = 1 - 2 * BITS
    h = {i: float(np.mean(f * z[:, i])) for i in range(N_NODES)}
    J = {(i, j): float(np.mean(f * z[:, i] * z[:, j])) for i in range(N_NODES) for j in range(i + 1, N_NODES)}
    return h, {k: v for k, v in J.items() if abs(v) > 1e-12}


# --------------------------------------------------------------------- #
# Circuits (the same circuit is used noiseless and noisy)
# --------------------------------------------------------------------- #
def qaoa_circuit(edges, variant, params, measure=True):
    p = len(params) // 2
    gammas, betas = params[:p], params[p:]
    h, J = ising_terms(objective_values(edges, variant))
    qc = QuantumCircuit(N_NODES)
    qg = start_qg(variant, edges)
    if variant == "xy":
        dicke_state(qc)
    else:
        thetas = np.arccos(qg)
        for q in range(N_NODES):
            qc.ry(float(thetas[q]), q)
    for layer in range(p):
        g, b = float(gammas[layer]), float(betas[layer])
        for (i, j), v in J.items():
            qc.rzz(2 * g * v, i, j)
        for i, v in h.items():
            if abs(v) > 1e-12:
                qc.rz(2 * g * v, i)
        if variant == "xy":
            ring = [(i, (i + 1) % N_NODES) for i in range(0, N_NODES, 2)] + \
                   [(i, (i + 1) % N_NODES) for i in range(1, N_NODES, 2)]
            for i, j in ring:
                qc.append(XXPlusYYGate(4 * b), [i, j])
        else:
            for q in range(N_NODES):
                qc.ry(-float(thetas[q]), q)
                qc.rz(-2 * b, q)
                qc.ry(float(thetas[q]), q)
    if measure:
        qc.measure_all()
    return qc


def ideal_distribution(edges, variant, params):
    from qiskit.quantum_info import Statevector

    return np.abs(Statevector(qaoa_circuit(edges, variant, params, measure=False)).data) ** 2


def optimize_angles(edges, variant, p=1, restarts=6, seed=0):
    """Minimize the noiseless expected F (penalized cost; plain cost for XY)."""
    f = objective_values(edges, variant)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(0, 0.6, p), rng.uniform(0, np.pi / 2, p)])
        res = minimize(lambda x: float(ideal_distribution(edges, variant, x) @ f), x0, method="COBYLA",
                       options={"maxiter": 300, "rhobeg": 0.2})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


# --------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------- #
def sample_metrics(probs, edges, filtered):
    """P(optimal) per shot and mean approximation ratio cover/opt
    (infeasible shots score 0), optionally after the qg filter."""
    c = cover_values(edges)
    opt, _ = brute_force(edges)
    p = np.where(FEASIBLE, probs, 0.0) if filtered else np.asarray(probs, dtype=float)
    kept = p.sum()
    if filtered:
        p = p / kept
    good = FEASIBLE & (c == opt)
    return {
        "p_opt": float(p[good].sum()),
        "ratio": float((p * np.where(FEASIBLE, c / opt, 0.0)).sum()),
        "kept": float(np.where(FEASIBLE, probs, 0).sum()),
        "mean_qg_z": float(probs @ (1 - 2 * WEIGHT / N_NODES)),
    }


def random_feasible_baseline(edges):
    c = cover_values(edges)
    opt, sols = brute_force(edges)
    return {"p_opt": len(sols) / FEASIBLE.sum(), "ratio": float(c[FEASIBLE].mean() / opt)}


# --------------------------------------------------------------------- #
# Noisy runs
# --------------------------------------------------------------------- #
def _probs(counts, shots):
    p = np.zeros(2**N_NODES)
    for key, v in counts.items():
        p[int(key.replace(" ", ""), 2)] += v
    return p / shots


ALL_TO_ALL = {"p2": 0.006, "p1": 0.0005, "dephase2": 0.001, "readout": 0.005}


def all_to_all_noise_model(params=ALL_TO_ALL):
    """Generic all-to-all device dominated by depolarizing two-qubit errors
    (same figures as examples/hubbard_trotter_qg_filters.py)."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, phase_damping_error

    nm = NoiseModel()
    deph = phase_damping_error(params["dephase2"])
    nm.add_all_qubit_quantum_error(depolarizing_error(params["p2"], 2).compose(deph.tensor(deph)), ["cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(params["p1"], 1), ["sx", "x"])
    r = params["readout"]
    nm.add_all_qubit_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]))
    return nm


def run_noisy(circuits, device="all_to_all", shots=4000, seed=11):
    from qiskit_aer import AerSimulator

    if device == "all_to_all":
        tc = transpile(circuits, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=1)
        sim = AerSimulator(noise_model=all_to_all_noise_model())
    elif device == "brisbane":
        from nisq_hardware_validation import choose_layout
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane

        backend = FakeBrisbane()
        tc = transpile(circuits, backend=backend, initial_layout=choose_layout(backend, N_NODES),
                       optimization_level=1, seed_transpiler=1, scheduling_method="alap")
        sim = AerSimulator.from_backend(backend)
    else:
        raise ValueError(device)
    res = sim.run(tc, shots=shots, seed_simulator=seed).result()
    twoq = [sum(c.count_ops().get(g, 0) for g in ("cx", "ecr", "cz")) for c in tc]
    return [_probs(res.get_counts(i), shots) for i in range(len(tc))], twoq


def experiment(graph_seeds=range(5), p=1, device="all_to_all", shots=4000, seed=11):
    """Per graph and variant: noiseless and noisy metrics, raw and filtered."""
    rows = []
    for gs in graph_seeds:
        edges = random_graph(gs)
        params = {v: optimize_angles(edges, v, p) for v in VARIANTS}
        circuits = [qaoa_circuit(edges, v, params[v]) for v in VARIANTS]
        noisy, twoq = run_noisy(circuits, device, shots, seed)
        for v, pn, n2 in zip(VARIANTS, noisy, twoq):
            pi = ideal_distribution(edges, v, params[v])
            rows.append({
                "graph": gs, "variant": v, "two_qubit_gates": n2,
                "ideal": sample_metrics(pi, edges, False), "ideal_f": sample_metrics(pi, edges, True),
                "noisy": sample_metrics(pn, edges, False), "noisy_f": sample_metrics(pn, edges, True),
            })
        rows.append({"graph": gs, "variant": "random_feasible", **random_feasible_baseline(edges),
                     "greedy_ratio": greedy(edges) / brute_force(edges)[0]})
    return rows


def summarize(rows):
    out = {}
    for v in VARIANTS:
        rs = [r for r in rows if r["variant"] == v]
        out[v] = {k: {m: float(np.mean([r[k][m] for r in rs])) for m in ("p_opt", "ratio", "kept", "mean_qg_z")}
                  for k in ("ideal", "ideal_f", "noisy", "noisy_f")}
        out[v]["two_qubit_gates"] = float(np.mean([r["two_qubit_gates"] for r in rs]))
    rb = [r for r in rows if r["variant"] == "random_feasible"]
    out["random_feasible"] = {"p_opt": float(np.mean([r["p_opt"] for r in rb])),
                              "ratio": float(np.mean([r["ratio"] for r in rb])),
                              "greedy_ratio": float(np.mean([r["greedy_ratio"] for r in rb]))}
    return out


def make_figure(summaries, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"standard": "standard\n(penalty)", "qg_budget": "qg budget\nstart", "qg_warm": "qg warm\nstart (LP)",
             "xy": "XY mixer\n(Dicke)"}
    fig, axes = plt.subplots(1, len(summaries), figsize=(4 * len(summaries), 3.8), sharey=True)
    for ax, (label, s) in zip(axes, summaries.items()):
        x = np.arange(len(VARIANTS))
        ax.bar(x - 0.27, [s[v]["ideal"]["p_opt"] for v in VARIANTS], 0.26, color="#c7c7c7", label="noiseless")
        ax.bar(x, [s[v]["noisy"]["p_opt"] for v in VARIANTS], 0.26, color="#8c8c8c", label="noisy")
        ax.bar(x + 0.27, [s[v]["noisy_f"]["p_opt"] for v in VARIANTS], 0.26, color="#1f6fb2", label="noisy + qg filter")
        ax.axhline(s["random_feasible"]["p_opt"], color="#c0392b", ls="--", lw=1)
        ax.text(3.45, s["random_feasible"]["p_opt"] + 0.01, "random feasible guess", ha="right", fontsize=7,
                color="#c0392b")
        for i, v in enumerate(VARIANTS):
            ax.text(i, -0.075, f"{s[v]['two_qubit_gates']:.0f} 2q", ha="center", fontsize=7, color="#555")
        ax.set_xticks(x)
        ax.set_xticklabels([names[v] for v in VARIANTS], fontsize=7.5)
        ax.set_title(label, fontsize=9)
        ax.set_ylim(-0.09, 0.56)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("P(optimal) per shot")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Maximum 3-vertex cover on 8-node graphs (mean of 5 graphs); classical greedy reaches 98.5% of optimum",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    summaries = {}
    for device in ("all_to_all", "brisbane"):
        for p in (1, 2):
            s = summaries[f"{device}, p = {p}"] = summarize(experiment(p=p, device=device))
            rb = s["random_feasible"]
            print(f"\n=== {device}, p = {p} (mean over 5 graphs, n = {N_NODES}, K = {K_CHOSEN}) ===")
            print(f"random feasible: P(opt) {rb['p_opt']:.3f}, ratio {rb['ratio']:.3f}; greedy ratio {rb['greedy_ratio']:.3f}")
            print(f"{'variant':10s} {'2q':>5} | {'ideal P(opt)':>12} {'kept':>5} | {'noisy P(opt)':>12} {'+qg filter':>10} "
                  f"| {'ratio':>6} {'+filter':>7} | {'kept':>5} {'qg_Z':>6}")
            for v in VARIANTS:
                r = s[v]
                print(f"{v:10s} {r['two_qubit_gates']:5.0f} | {r['ideal']['p_opt']:12.3f} {r['ideal']['kept']:5.2f} | "
                      f"{r['noisy']['p_opt']:12.3f} {r['noisy_f']['p_opt']:10.3f} | {r['noisy']['ratio']:6.3f} "
                      f"{r['noisy_f']['ratio']:7.3f} | {r['noisy']['kept']:5.2f} {r['noisy']['mean_qg_z']:+6.3f}")
    if "--figure" in sys.argv:
        del summaries["brisbane, p = 2"]
        make_figure(summaries, __file__.replace(".py", ".png"))
