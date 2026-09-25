"""
Circuit knitting in practice: choosing WHICH gates to cut with the qg
cost formula.

§15.2 gives the sampling cost of cutting one two-qubit Pauli rotation
in closed form, gamma^2 = (1 + 2 sqrt(1 - qg^2))^2 with qg = cos(theta).
Here that formula is used to decide where to split a circuit that is too
large for one device, and the decision is compared with the angle-blind
alternatives people actually use.

Problem. An 8-qubit circuit of RZZ couplings on a sparse graph (ring
plus 4 random chords, 12 couplings) must run on devices of at most 4
qubits, so some gates have to be cut. Two families of angles:

  trotter  2 Trotter steps of a disordered Ising model,
           theta = 2 J dt, J ~ U(0.2, 1.5), dt = 0.25  ->  theta in [0.1, 0.75]
  qaoa     one weighted-MaxCut QAOA layer, theta = 2 gamma w,
           w ~ {1..6}, gamma = 0.45                 ->  theta in [0.9, 5.4]

Cut-selection rules, each evaluated by the true shot multiplier
(product of gamma^2 over the cut gates):

  count      fewest cut gates (the usual angle-blind rule; ties averaged)
  weakest    smallest total |theta| cut ("cut the weakest bonds"; ties averaged)
  qg         smallest sum of log gamma^2(qg): the exact optimum, found by
             scoring every partition into blocks of <= 4 qubits (3,795 of
             them) with the closed form
  addon      qiskit-addon-cutting's find_cuts (gate cuts only), as the
             reference implementation

Then one 6-qubit instance is cut end to end with the addon, once where
the count rule says and once where the qg rule says, with the same total
shot budget, to see whether the predicted cost shows up as error.

Findings (200 random instances per family; overhead = shot multiplier,
geometric mean over instances):

                  count    weakest   qg      addon  | qg strictly cheaper than
                                                    | count      weakest
    trotter       7,254    3,838     3,642   3,642  |  60%        12%
    qaoa            976    2,174       753     753  |  52%        62%

  * Trotter regime (every theta < pi/2): gamma grows with |theta|, so
    "cut the weakest bonds" is nearly the qg rule (5% more shots). The
    angle-blind count rule needs 2.0x more shots.
  * Large angles (QAOA): the cost is not monotone in the coupling, since
    RZZ(theta) near pi is almost a local Z(x)Z and nearly free to cut.
    "Cut the weakest bonds" becomes the worst rule (2.9x the qg optimum,
    worse than counting gates); qg is 1.3x cheaper than the count rule.
  * qiskit-addon-cutting's find_cuts finds the same optimum in all 400
    instances: it already prices each gate by its own QPD 1-norm, which
    is what the qg formula gives in closed form. What qg adds is the
    formula itself: at this size, exhaustively scoring all 3,795
    partitions takes 1-3 ms, vs 17-170 ms for the addon's search (but
    enumeration grows exponentially and the search does not).
  * End to end (6 qubits, 3 per device, 60,000 shots, 10 seeds; RMSE of
    the six <X_i>): the count rule cuts 2 gates (overhead 81), the qg
    rule 3 gates (overhead 7.4).
        shots per QPD term proportional to |c_i|:  count 0.037, qg 0.013
            -> 2.9x smaller error, predicted sqrt(81/7.4) = 3.3
        same shots for every subexperiment:        count 0.037, qg 0.083
            -> the ranking flips
    The gamma^2 law assumes importance sampling of the QPD terms. With
    uniform shots the cost grows with the number of subexperiments
    (6^k for k cuts) instead, and cutting fewer gates wins. So the qg
    cost is the right criterion only together with proportional shot
    allocation (the addon does this when num_samples is finite).

Honest summary: a cost saving, not a physical advantage, and the leading
tool already makes the angle-aware choice. The measurable gain is
against angle-blind rules (count) and magnitude rules (weakest bond),
largest when gate angles pass pi/2, and it only materializes with
shots allocated in proportion to the QPD weights.
"""

import math
import sys
import time

import numpy as np

from qang.knitting import cut_sampling_overhead

N_QUBITS = 8
CAPACITY = 4


# --------------------------------------------------------------------- #
# instances
# --------------------------------------------------------------------- #
def random_graph(n: int, n_chords: int, rng: np.random.Generator):
    """Ring plus n_chords random non-adjacent chords (no repeats)."""
    edges = {(i, (i + 1) % n) if i < (i + 1) % n else ((i + 1) % n, i) for i in range(n)}
    while len(edges) < n + n_chords:
        a, b = sorted(rng.choice(n, 2, replace=False).tolist())
        edges.add((a, b))
    return sorted(edges)


def instance(kind: str, seed: int, n: int = N_QUBITS):
    """List of RZZ gates (a, b, theta), in circuit order."""
    rng = np.random.default_rng(seed)
    edges = random_graph(n, n // 2, rng)
    if kind == "trotter":
        dt, steps = 0.25, 2
        thetas = {e: 2.0 * rng.uniform(0.2, 1.5) * dt for e in edges}
        return [(a, b, thetas[(a, b)]) for _ in range(steps) for (a, b) in edges]
    if kind == "qaoa":
        gamma = 0.45
        return [(a, b, 2.0 * gamma * int(rng.integers(1, 7))) for (a, b) in edges]
    raise ValueError(kind)


def gate_log_cost(theta: float) -> float:
    """log of gamma^2 for cutting RZZ(theta), from qg = cos(theta)."""
    return math.log(cut_sampling_overhead(max(-1.0, min(1.0, math.cos(theta)))))


# --------------------------------------------------------------------- #
# partitions and rules
# --------------------------------------------------------------------- #
def capacity_partitions(n: int, cap: int):
    """All set partitions of range(n) into blocks of size <= cap, as label
    tuples (restricted growth strings)."""
    out = []

    def rec(labels, sizes):
        i = len(labels)
        if i == n:
            out.append(tuple(labels))
            return
        for b in range(len(sizes) + 1):
            if b == len(sizes):
                rec(labels + [b], sizes + [1])
            elif sizes[b] < cap:
                sizes[b] += 1
                rec(labels + [b], sizes)
                sizes[b] -= 1

    rec([], [])
    return out


def _cut_mask(gates, parts: np.ndarray) -> np.ndarray:
    a = np.array([g[0] for g in gates])
    b = np.array([g[1] for g in gates])
    return parts[:, a] != parts[:, b]  # (n_partitions, n_gates)


RULES = ("count", "weakest", "qg")


def rule_overheads(gates, parts: np.ndarray) -> dict:
    """True log-overhead of the partition each rule picks (ties averaged)."""
    cut = _cut_mask(gates, parts)
    logc = np.array([gate_log_cost(t) for _, _, t in gates])
    true_cost = cut @ logc
    scores = {
        "count": cut.sum(axis=1).astype(float),
        "weakest": cut @ np.array([abs(t) for _, _, t in gates]),
        "qg": true_cost,
    }
    out = {}
    for rule, s in scores.items():
        best = np.isclose(s, s.min(), rtol=0, atol=1e-9)
        out[rule] = float(true_cost[best].mean())
    return out


def addon_log_overhead(gates, n: int = N_QUBITS, cap: int = CAPACITY, seed: int = 1) -> float:
    from qiskit import QuantumCircuit
    from qiskit_addon_cutting import DeviceConstraints, OptimizationParameters, find_cuts

    qc = QuantumCircuit(n)
    for a, b, t in gates:
        qc.rzz(t, a, b)
    _, meta = find_cuts(qc, OptimizationParameters(seed=seed, gate_lo=True, wire_lo=False, max_gamma=1e12),
                        DeviceConstraints(qubits_per_subcircuit=cap))
    return math.log(float(meta["sampling_overhead"]))


def survey(kind: str, seeds=range(200), with_addon: bool = True) -> dict:
    parts = np.array(capacity_partitions(N_QUBITS, CAPACITY))
    rows = []
    t_qg = t_addon = 0.0
    for s in seeds:
        g = instance(kind, s)
        t0 = time.perf_counter()
        r = rule_overheads(g, parts)
        t_qg += time.perf_counter() - t0
        if with_addon:
            t0 = time.perf_counter()
            r["addon"] = addon_log_overhead(g)
            t_addon += time.perf_counter() - t0
        rows.append(r)
    keys = list(rows[0])
    gm = {k: float(np.exp(np.mean([r[k] for r in rows]))) for k in keys}
    better = {k: float(np.mean([r["qg"] < r[k] - 1e-9 for r in rows])) for k in ("count", "weakest")}
    out = {"geo_mean_overhead": gm, "qg_strictly_better": better, "n_partitions": len(parts),
           "seconds_per_instance_qg": t_qg / len(rows)}
    if with_addon:
        out["addon_matches_qg"] = float(np.mean([abs(r["addon"] - r["qg"]) < 1e-6 for r in rows]))
        out["addon_worse_than_qg"] = float(np.mean([r["addon"] > r["qg"] + 1e-6 for r in rows]))
        out["seconds_per_instance_addon"] = t_addon / len(rows)
    return out


# --------------------------------------------------------------------- #
# end to end: does the predicted overhead show up as error?
# --------------------------------------------------------------------- #
# 6 qubits, 3 per device. The count rule's unique choice {0,1,2}|{3,4,5}
# cuts the two strong couplings (theta = pi/2, gamma^2 = 9 each: overhead
# 81); the qg rule's choice {0,1,3}|{2,4,5} cuts three weak ones
# (theta = 0.2, gamma^2 = 1.95 each: overhead 7.4).
E2E_GATES = [
    (0, 1, 1.1), (4, 5, 1.2),                      # inside both choices
    (0, 2, 0.2), (1, 2, 0.2), (3, 4, 0.2),         # weak: cut by the qg choice
    (2, 5, math.pi / 2), (1, 3, math.pi / 2),      # strong: cut by the count choice
]
COUNT_LABELS = "AAABBB"  # qubits 0,1,2 | 3,4,5
QG_LABELS = "AABABB"     # qubits 0,1,3 | 2,4,5


def e2e_circuit():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(6)
    qc.h(range(6))
    for a, b, t in E2E_GATES:
        qc.rzz(t, a, b)
    return qc


def cut_estimate(labels: str, total_shots: int, seed: int, allocation: str = "proportional"):
    """Reconstruct <X_i> for all i with the addon, cutting where `labels` says.

    allocation = "proportional": term i of the QPD gets shots proportional
    to |c_i| (the importance sampling behind the gamma^2 law);
    "uniform": every subexperiment gets the same shots.
    Returns (estimates, number of distinct circuits)."""
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_addon_cutting import (generate_cutting_experiments, partition_problem,
                                      reconstruct_expectation_values)
    from qiskit_aer.primitives import SamplerV2

    obs = SparsePauliOp(["I" * (5 - i) + "X" + "I" * i for i in range(6)])
    part = partition_problem(circuit=e2e_circuit(), partition_labels=labels, observables=obs.paulis)
    subexp, coeffs = generate_cutting_experiments(circuits=part.subcircuits, observables=part.subobservables,
                                                  num_samples=np.inf)
    n_terms = len(coeffs)
    n_circ = sum(len(v) for v in subexp.values())
    per_term_budget = total_shots / len(subexp)  # split the budget across the subcircuits
    w = np.abs([c for c, _ in coeffs])
    if allocation == "proportional":
        term_shots = np.maximum(1, np.round(per_term_budget * w / w.sum())).astype(int)
    elif allocation == "uniform":
        term_shots = np.full(n_terms, max(1, int(per_term_budget // n_terms)))
    else:
        raise ValueError(allocation)
    sampler = SamplerV2(seed=seed)
    results = {}
    for k, circs in subexp.items():
        per_circ = len(circs) // n_terms  # subexperiments per term (one per observable group)
        pubs = [(c, None, int(term_shots[i // per_circ])) for i, c in enumerate(circs)]
        results[k] = sampler.run(pubs).result()
    return np.array(reconstruct_expectation_values(results, coeffs, part.subobservables)), n_circ


def e2e_errors(total_shots: int = 60000, seeds=range(10), allocation: str = "proportional") -> dict:
    from qiskit.quantum_info import SparsePauliOp, Statevector

    sv = Statevector(e2e_circuit())
    exact = np.array([sv.expectation_value(SparsePauliOp("I" * (5 - i) + "X" + "I" * i)).real for i in range(6)])
    out = {}
    for name, labels in (("count", COUNT_LABELS), ("qg", QG_LABELS)):
        errs = []
        for s in seeds:
            est, n_circ = cut_estimate(labels, total_shots, s, allocation)
            errs.append(np.sqrt(np.mean((est - exact) ** 2)))
        cut = [g for g in E2E_GATES if labels[g[0]] != labels[g[1]]]
        out[name] = {
            "rmse": float(np.sqrt(np.mean(np.square(errs)))),
            "n_cut": len(cut),
            "overhead": float(np.prod([cut_sampling_overhead(math.cos(t)) for *_, t in cut])),
            "n_circuits": n_circ,
        }
    return out


# --------------------------------------------------------------------- #
def make_figure(results: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    th = np.linspace(0, 2 * np.pi, 400)
    ax[0].plot(th, [cut_sampling_overhead(math.cos(t)) for t in th], color="#1f6fb2")
    ax[0].set_xlabel("RZZ angle theta")
    ax[0].set_ylabel("gamma^2 = (1 + 2 sqrt(1 - qg^2))^2")
    ax[0].set_title("Cost of one cut: not monotone in the coupling")
    ax[0].axvspan(0.1, 0.75, color="#2e8b57", alpha=0.15, label="trotter angles")
    ax[0].axvspan(0.9, 5.4, color="#e0a030", alpha=0.12, label="qaoa angles")
    ax[0].legend(fontsize=8)

    kinds = list(results)
    labels = ["count", "weakest", "qg", "addon"]
    colors = {"count": "#8c8c8c", "weakest": "#c0392b", "qg": "#1f6fb2", "addon": "#2e8b57"}
    x = np.arange(len(kinds))
    for j, lab in enumerate(labels):
        vals = [results[k]["geo_mean_overhead"].get(lab, np.nan) for k in kinds]
        ax[1].bar(x + (j - 1.5) * 0.2, vals, 0.18, color=colors[lab], label=lab)
    ax[1].set_yscale("log")
    ax[1].set_ylim(1, 2e4)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(kinds)
    ax[1].set_ylabel("shot multiplier (geometric mean, 200 instances)")
    ax[1].set_title("Which gates to cut: 8 qubits on 4-qubit devices")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    results = {}
    for kind in ("trotter", "qaoa"):
        r = survey(kind)
        results[kind] = r
        gm = r["geo_mean_overhead"]
        print(f"{kind:8s} partitions={r['n_partitions']}  overhead: "
              + "  ".join(f"{k}={v:,.1f}" for k, v in gm.items())
              + f"  | qg strictly better than count {r['qg_strictly_better']['count']:.0%}, "
                f"weakest {r['qg_strictly_better']['weakest']:.0%}"
              + f"  | addon = qg in {r['addon_matches_qg']:.0%}, worse in {r['addon_worse_than_qg']:.0%}"
              + f"  | time/instance qg {1e3 * r['seconds_per_instance_qg']:.1f} ms, "
                f"addon {1e3 * r['seconds_per_instance_addon']:.1f} ms")
    for alloc in ("proportional", "uniform"):
        e = e2e_errors(allocation=alloc)
        for k, v in e.items():
            print(f"e2e {alloc:12s} {k:5s}: cut {v['n_cut']} gates, overhead {v['overhead']:.1f}, "
                  f"{v['n_circuits']} circuits, RMSE <X_i> {v['rmse']:.4f}")
        print(f"e2e {alloc}: RMSE ratio count/qg = {e['count']['rmse'] / e['qg']['rmse']:.2f}, "
              f"predicted sqrt(overhead ratio) = {math.sqrt(e['count']['overhead'] / e['qg']['overhead']):.2f}")
    if "--figure" in sys.argv:
        make_figure(results, __file__.replace(".py", ".png"))
