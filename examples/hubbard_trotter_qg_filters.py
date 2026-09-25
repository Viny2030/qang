"""
Materials simulation: Trotterized dynamics of the 1D Fermi-Hubbard model
on a noisy device, with and without the qg symmetry witnesses/filters.

Model: L = 4 sites, open chain, hopping J = 1, on-site repulsion U = 2,

    H = -J sum_{<ij>,s} (c+_is c_js + h.c.) + U sum_i n_i,up n_i,down.

Jordan-Wigner with the spin sectors in blocks: qubits 0..L-1 hold spin
up, L..2L-1 spin down, so each hop is a nearest-neighbour XX+YY term
(no Z strings) and each interaction a ZZ + Z term between qubits i and
L+i. First-order Trotter steps (dt = 0.25), 8 qubits.

Initial state: charge-density wave, sites 0 and 2 doubly occupied
(N_up = N_down = 2). Observables, all in the Z basis:

  charge imbalance  I(t) = (n_0 - n_1 + n_2 - n_3) / N      (I(0) = 1)
  double occupancy  D(t) = (1/L) sum_i <n_i,up n_i,down>     (D(0) = 1/2)

H conserves N_up and N_down separately, so each spin register has a
known ideal mean qg_Z = 1 - 2 N_s / L = 0 at every time. Unlike H2
(§21, §28), here EVERY observable of interest is measured in the Z
basis, so the qg filters act on all of them.

Methods (all on top of tensored readout mitigation):
  raw          readout-mitigated
  N filter     keep shots with total Hamming weight N = 4 (§21)
  spin filter  keep shots with N_up = 2 and N_down = 2 (§28)
  ZNE          every two-qubit gate of the routed device circuit folded
               to 3 and 5 copies, Richardson to zero noise (§24)
  ZNE + spin   spin filter at each scale, then extrapolate

The reference is the noiseless output of the SAME Trotter circuit (we
measure device error, not Trotter error; the Trotter error vs exact
evolution is printed separately).

Two devices: a generic all-to-all device dominated by depolarizing
two-qubit errors (0.6% per CX, trapped-ion-like figures, no routing: 20
CX per Trotter step), and fake_brisbane (heavy-hex; routing the ladder
of hops and interactions costs 56 ECR per step, plus idle T1).

Findings (20,000 shots per circuit, 5 seeds, t = J dt * steps up to 2):

  all_to_all        steps 1    2    4    6    8   | time average
    double occ. raw      .008 .002 .017 .027 .042 | .019
       + N filter        .003 .001 .010 .020 .030 | .013
       + spin filters    .003 .001 .007 .014 .021 | .009
       + ZNE             .001 .002 .002 .009 .017 | .006
       + ZNE + spin      .001 .001 .002 .004 .007 | .003
    imbalance (time avg): raw .026, N .013, spin .011, ZNE .006,
                          ZNE + spin .008
    witnesses: mean qg_Z of both registers stays at 0 (the noise is
    unital) but the kept fraction drops 0.88 -> 0.50; spin leak 0 -> 0.08

  fake_brisbane     steps 1    2   (4, 6, 8: noise floor, see below)
    double occ. raw      .081 .056
       + N filter        .050 .029
       + spin filters    .039 .016
       + ZNE             .031 .031
       + ZNE + spin      .009 .006
    imbalance      raw   .206 .111 ; N .124 .070 ; spin .100 .047 ;
                   ZNE   .036 .010 ; ZNE + spin .035 .071
    witnesses: qg_up ~0, qg_down +0.035 .. +0.067; spin leak 0.08, 0.16,
    0.41, 0.45, 0.47

  * The qg filters roughly halve the error on the all-to-all device and
    the second (spin) filter adds a further 30% at depth, because the
    spin leak grows with depth. ZNE + spin filters is the best estimate
    of the double occupancy on both devices (7-9x below raw).
  * Routing creates spin leak. On fake_brisbane 8-47% of the
    right-N shots sit in the wrong spin sector (H2 in §28: < 1%), so the
    spin filters beat the N filter clearly at 1-2 steps (double occ.
    0.050 -> 0.039, 0.029 -> 0.016).
  * The two witnesses localize the damage: with the fake_brisbane
    snapshot of qiskit-ibm-runtime 0.49 mostly the spin-down register
    drifts towards +1 (T1), i.e. the physical qubits that hold it are
    worse; with the older 0.43 snapshot both registers drift (+0.066,
    +0.070 at 2 steps). The asymmetry is a property of the calibration,
    not of the method.
  * Limit: from 4 steps (>= 188 ECR, 18% of shots kept) every method
    sits on the noise floor -- D -> 1/4, the value of the maximally mixed
    state inside the spin sector, and I -> 0 -- as for LiH in §21.
  * On the all-to-all device the mitigated error (0.003-0.01) is already
    comparable to the Trotter error itself (0.01-0.02 vs exact evolution).
  * Classically, 8 qubits (and the 1D Hubbard model up to ~20 sites by
    exact diagonalization, far more with tensor networks) are easy: this
    ranks error mitigation for a quantum simulation, not quantum vs
    classical.
"""

import os
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp, Statevector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

L_SITES = 4
N_QUBITS = 2 * L_SITES
J_HOP, U_INT, DT = 1.0, 2.0, 0.25
N_UP = N_DOWN = 2
INITIAL_DOUBLY_OCCUPIED = (0, 2)
SCALES = (1, 3, 5)
RICHARDSON = np.array([15.0, -10.0, 3.0]) / 8.0

_IDX = np.arange(2**N_QUBITS)
_BITS = (_IDX[:, None] >> np.arange(N_QUBITS)) & 1  # (256, 8), qubit q = bit q
UP = list(range(L_SITES))
DOWN = list(range(L_SITES, N_QUBITS))
_N_UP = _BITS[:, UP].sum(1)
_N_DOWN = _BITS[:, DOWN].sum(1)


def up(i):
    return i


def down(i):
    return L_SITES + i


# --------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------- #
def hamiltonian(L=L_SITES, J=J_HOP, U=U_INT) -> SparsePauliOp:
    n = 2 * L
    terms = []

    def pauli(ops):
        s = ["I"] * n
        for q, p in ops.items():
            s[n - 1 - q] = p
        return "".join(s)

    for s in (0, L):
        for i in range(L - 1):
            a, b = s + i, s + i + 1
            terms += [(pauli({a: "X", b: "X"}), -J / 2), (pauli({a: "Y", b: "Y"}), -J / 2)]
    for i in range(L):
        a, b = i, L + i
        terms += [(pauli({}), U / 4), (pauli({a: "Z"}), -U / 4), (pauli({b: "Z"}), -U / 4),
                  (pauli({a: "Z", b: "Z"}), U / 4)]
    return SparsePauliOp.from_list(terms).simplify()


def initial_state(qc: QuantumCircuit):
    for i in INITIAL_DOUBLY_OCCUPIED:
        qc.x(up(i))
        qc.x(down(i))


def trotter_step(qc: QuantumCircuit, dt=DT, J=J_HOP, U=U_INT, L=L_SITES):
    """exp(-i H dt), first order: even bonds, odd bonds, then interaction."""
    for parity in (0, 1):
        for i in range(parity, L - 1, 2):
            for s in (0, L):
                # exp(+i J dt/2 (XX + YY)); XXPlusYYGate(theta) = exp(-i theta/4 (XX + YY))
                qc.append(XXPlusYYGate(-2 * J * dt), [s + i, s + i + 1])
    for i in range(L):
        # exp(-i dt U/4 (ZZ - Z_up - Z_down)), global phase dropped
        qc.rzz(U * dt / 2, i, L + i)
        qc.rz(-U * dt / 2, i)
        qc.rz(-U * dt / 2, L + i)


def trotter_circuit(steps: int, measure=True) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)
    initial_state(qc)
    for _ in range(steps):
        trotter_step(qc)
    if measure:
        qc.measure_all()
    return qc


def ideal_probabilities(steps: int) -> np.ndarray:
    return np.abs(Statevector(trotter_circuit(steps, measure=False)).data) ** 2


def exact_probabilities(t: float) -> np.ndarray:
    """exp(-i H t) on the initial state (no Trotter error)."""
    from scipy.linalg import expm

    qc = QuantumCircuit(N_QUBITS)
    initial_state(qc)
    psi = expm(-1j * t * hamiltonian().to_matrix()) @ Statevector(qc).data
    return np.abs(psi) ** 2


# --------------------------------------------------------------------- #
# Observables, witnesses, filters
# --------------------------------------------------------------------- #
def site_densities(p):
    occ = p @ _BITS  # <n_q> per qubit
    return occ[UP] + occ[DOWN]


def charge_imbalance(p):
    n = site_densities(p)
    sign = np.array([(-1) ** i for i in range(L_SITES)])
    return float(sign @ n / n.sum())


def double_occupancy(p):
    both = _BITS[:, UP] * _BITS[:, DOWN]
    return float((p @ both).mean())


def spin_witnesses(p):
    """(mean qg_Z of the up register, of the down register, N kept
    fraction, spin-filter kept fraction)."""
    qg_up = float(p @ (1 - 2 * _N_UP / L_SITES))
    qg_down = float(p @ (1 - 2 * _N_DOWN / L_SITES))
    kept_n = float(p[(_N_UP + _N_DOWN) == N_UP + N_DOWN].sum())
    kept_s = float(p[(_N_UP == N_UP) & (_N_DOWN == N_DOWN)].sum())
    return qg_up, qg_down, kept_n, kept_s


def n_filter(p):
    k = np.where((_N_UP + _N_DOWN) == N_UP + N_DOWN, p, 0.0)
    return k / k.sum()


def spin_filter(p):
    k = np.where((_N_UP == N_UP) & (_N_DOWN == N_DOWN), p, 0.0)
    return k / k.sum()


# --------------------------------------------------------------------- #
# Noisy runs
# --------------------------------------------------------------------- #
def _counts_to_probs(counts, shots):
    p = np.zeros(2**N_QUBITS)
    for key, c in counts.items():
        p[int(key.replace(" ", ""), 2)] += c
    return p / shots


def readout_inverse(p_all0, p_all1):
    """Tensored inverse of per-qubit confusion matrices (2^n x 2^n)."""
    inv = np.array([[1.0]])
    for q in range(N_QUBITS - 1, -1, -1):
        e01 = p_all0[_BITS[:, q] == 1].sum()
        e10 = p_all1[_BITS[:, q] == 0].sum()
        inv = np.kron(inv, np.linalg.inv(np.array([[1 - e01, e10], [e01, 1 - e10]])))
    return inv


def fold_two_qubit(qc: QuantumCircuit, scale: int, names=("ecr", "cx", "cz")) -> QuantumCircuit:
    """Replace every self-inverse two-qubit gate G by G^scale (scale odd)."""
    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale must be a positive odd integer.")
    out = qc.copy_empty_like()
    for inst in qc.data:
        reps = scale if inst.operation.name in names else 1
        for _ in range(reps):
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def _layout(backend):
    """Snake order on a line of 2L physical qubits: up_0..up_{L-1},
    down_{L-1}..down_0, so hops are nearest neighbours."""
    from nisq_hardware_validation import choose_layout

    line = choose_layout(backend, N_QUBITS)
    order = UP + DOWN[::-1]
    layout = [None] * N_QUBITS
    for phys, logical in zip(line, order):
        layout[logical] = phys
    return layout


def _calibration():
    out = []
    for bit in (0, 1):
        qc = QuantumCircuit(N_QUBITS)
        if bit:
            qc.x(range(N_QUBITS))
        qc.measure_all()
        out.append(qc)
    return out


ALL_TO_ALL = {"p2": 0.006, "p1": 0.0005, "dephase2": 0.001, "readout": 0.005}


def all_to_all_noise_model(params=ALL_TO_ALL):
    """Generic all-to-all device dominated by depolarizing two-qubit
    errors (trapped-ion-like figures of merit; not a vendor calibration)."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, phase_damping_error

    nm = NoiseModel()
    deph = phase_damping_error(params["dephase2"])
    nm.add_all_qubit_quantum_error(depolarizing_error(params["p2"], 2).compose(deph.tensor(deph)), ["cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(params["p1"], 1), ["sx", "x"])
    r = params["readout"]
    nm.add_all_qubit_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]))
    return nm


def run_noisy(steps_list=(1, 2, 3, 4), shots=20000, seed=11, device="brisbane", zne=True):
    """For each Trotter depth: observables per method and witnesses.
    device: "brisbane" (fake_brisbane, heavy-hex routing, idle T1) or
    "all_to_all" (no routing, depolarizing-dominated noise)."""
    from qiskit_aer import AerSimulator

    circuits = [trotter_circuit(s) for s in steps_list] + _calibration()
    scales = SCALES if zne else (1,)
    if device == "brisbane":
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane

        backend = FakeBrisbane()
        routed = transpile(circuits, backend=backend, initial_layout=_layout(backend),
                           optimization_level=1, seed_transpiler=1)
        batch = [fold_two_qubit(c, k) for k in scales for c in routed[:len(steps_list)]] + routed[-2:]
        batch = transpile(batch, backend=backend, initial_layout=list(range(backend.num_qubits)),
                          optimization_level=0, scheduling_method="alap", seed_transpiler=1)
        sim = AerSimulator.from_backend(backend)
    elif device == "all_to_all":
        routed = transpile(circuits, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=1)
        batch = [fold_two_qubit(c, k) for k in scales for c in routed[:len(steps_list)]] + routed[-2:]
        sim = AerSimulator(noise_model=all_to_all_noise_model())
    else:
        raise ValueError(device)
    res = sim.run(batch, shots=shots, seed_simulator=seed).result()
    P = [_counts_to_probs(res.get_counts(i), shots) for i in range(len(batch))]
    inv = readout_inverse(P[-2], P[-1])
    m = len(steps_list)
    out = []
    for j, steps in enumerate(steps_list):
        per_scale = [inv @ P[k * m + j] for k in range(len(scales))]
        row = {"steps": steps, "two_qubit_gates": sum(routed[j].count_ops().get(g, 0) for g in ("ecr", "cx", "cz"))}
        row["qg_up"], row["qg_down"], row["kept_n"], row["kept_spin"] = spin_witnesses(P[j])
        for obs_name, obs in (("imbalance", charge_imbalance), ("double_occ", double_occupancy)):
            row[(obs_name, "raw")] = obs(per_scale[0])
            row[(obs_name, "n_filter")] = obs(n_filter(per_scale[0]))
            row[(obs_name, "spin_filter")] = obs(spin_filter(per_scale[0]))
            if zne:
                row[(obs_name, "zne")] = float(RICHARDSON @ [obs(p) for p in per_scale])
                row[(obs_name, "zne_spin")] = float(RICHARDSON @ [obs(spin_filter(p)) for p in per_scale])
        out.append(row)
    return out


METHODS = ("raw", "n_filter", "spin_filter", "zne", "zne_spin")
OBSERVABLES = {"imbalance": charge_imbalance, "double_occ": double_occupancy}


def error_table(steps_list=(1, 2, 3, 4), seeds=(11, 12, 13), shots=20000, device="brisbane"):
    """Mean absolute error vs the noiseless Trotter circuit, averaged over
    seeds: {steps: {(obs, method): (mean err, std), witnesses...}}."""
    runs = [run_noisy(steps_list, shots, s, device) for s in seeds]
    table = {}
    for j, steps in enumerate(steps_list):
        ideal = ideal_probabilities(steps)
        rows = [r[j] for r in runs]
        entry = {k: float(np.mean([r[k] for r in rows])) for k in ("qg_up", "qg_down", "kept_n", "kept_spin")}
        entry["two_qubit_gates"] = rows[0]["two_qubit_gates"]
        for obs_name, obs in OBSERVABLES.items():
            entry[(obs_name, "ideal")] = obs(ideal)
            for m in METHODS:
                errs = [abs(r[(obs_name, m)] - obs(ideal)) for r in rows]
                entry[(obs_name, m)] = (float(np.mean(errs)), float(np.std([r[(obs_name, m)] for r in rows])))
                entry[(obs_name, m, "value")] = float(np.mean([r[(obs_name, m)] for r in rows]))
        entry["spin_leak"] = 1.0 - entry["kept_spin"] / entry["kept_n"]
        table[steps] = entry
    return table


STEPS = (1, 2, 4, 6, 8)
SEEDS = (11, 12, 13, 14, 15)
DEVICES = ("all_to_all", "brisbane")


def time_averaged_errors(table):
    """{(obs, method): mean over depths of the mean |error|}."""
    return {(o, m): float(np.mean([e[(o, m)][0] for e in table.values()])) for o in OBSERVABLES for m in METHODS}


def make_figure(tables, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"raw": ("#8c8c8c", "o", "readout-mitigated"), "n_filter": ("#9ecae1", "s", "+ N filter"),
             "spin_filter": ("#1f6fb2", "D", "+ spin filters (N_up, N_down)"),
             "zne_spin": ("#2e8b57", "^", "+ ZNE + spin filters")}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    fine = np.arange(0, max(STEPS) + 1)
    for r, device in enumerate(DEVICES):
        table = tables[device]
        for c, (obs_name, obs) in enumerate(OBSERVABLES.items()):
            ax = axes[r, c]
            ax.plot(fine * DT, [obs(ideal_probabilities(k)) for k in fine], "k-", lw=1.5, label="noiseless Trotter")
            for m, (col, mk, lab) in style.items():
                ax.plot([s * DT for s in table], [e[(obs_name, m, "value")] for e in table.values()],
                        mk, color=col, ms=6, label=lab, mec="white", mew=0.8)
            title = "all-to-all, depolarizing-dominated" if device == "all_to_all" else "fake_brisbane (heavy-hex, routed)"
            ax.set_title(f"{title}: {'charge imbalance' if obs_name == 'imbalance' else 'double occupancy'}",
                         fontsize=9)
            ax.grid(alpha=0.3)
            if r == 1:
                ax.set_xlabel("time J t")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("1D Hubbard, L = 4 (8 qubits), U = 2J: CDW melting on noisy devices", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    print("Trotter error vs exact evolution (noiseless):")
    for s in STEPS:
        pi, pe = ideal_probabilities(s), exact_probabilities(s * DT)
        print(f"  steps {s}: I trotter {charge_imbalance(pi):+.3f} exact {charge_imbalance(pe):+.3f}  "
              f"D trotter {double_occupancy(pi):.3f} exact {double_occupancy(pe):.3f}")
    tables = {}
    for device in DEVICES:
        table = tables[device] = error_table(STEPS, SEEDS, device=device)
        print(f"\n=== {device} ===")
        for obs_name in OBSERVABLES:
            print(f"{obs_name}: |error| vs noiseless Trotter circuit (mean over {len(SEEDS)} seeds)")
            print(f"{'steps':>5} {'2q':>4} {'qg_up':>6} {'qg_dn':>6} {'keptN':>6} {'keptS':>6} {'ideal':>6} | "
                  + " ".join(f"{m:>11s}" for m in METHODS))
            for s, e in table.items():
                print(f"{s:5d} {e['two_qubit_gates']:4d} {e['qg_up']:+6.3f} {e['qg_down']:+6.3f} {e['kept_n']:6.2f} "
                      f"{e['kept_spin']:6.2f} {e[(obs_name, 'ideal')]:+6.3f} | "
                      + " ".join(f"{e[(obs_name, m)][0]:11.3f}" for m in METHODS))
        avg = time_averaged_errors(table)
        for obs_name in OBSERVABLES:
            print(f"  time-averaged {obs_name}: " + "  ".join(f"{m} {avg[(obs_name, m)]:.3f}" for m in METHODS))
        print("  spin leak: " + "  ".join(f"{s}:{e['spin_leak']:.2f}" for s, e in table.items()))
    if "--figure" in sys.argv:
        make_figure(tables, __file__.replace(".py", ".png"))
