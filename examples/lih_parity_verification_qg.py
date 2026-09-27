"""
LiH: reaching the X/Y measurement groups with an electron-parity check.

§50 showed that the qg electron-number filter cannot help LiH because the
hardware error sits in the 16 measurement groups read in rotated (X/Y)
bases, which a Z-basis filter never sees. The electron-number PARITY
P = Z_0 Z_1 ... Z_5 commutes with every Hamiltonian term (each
number-conserving Jordan-Wigner term has an even number of X/Y factors),
so it can be measured on an ancilla before the basis rotation and used to
post-select every group: keep the shots with P = (-1)^N = +1. It catches
every error that changes N by an odd amount (a single bit flip), not the
ones that change it by two.

Three levels, all against the noiseless energy of the same circuit
(mHa), LiH at 3.0 A, ansatz with 1-3 layers:

  A. exact density matrix, ideal projections (no shots, no readout):
       raw          Tr(rho H)
       parity       Tr(P+ rho P+ H) / Tr(P+ rho)      what a perfect parity check gives
       mod 4        the same with the projector onto N = 2 mod 4
       number       the same with the projector onto N = 2
                    (the full symmetry-verification ceiling)
  B. the parity check built with an ancilla: CNOT from each of the 6
     qubits into the ancilla, then the group's basis rotation, 200,000
     shots, readout corrected on all 7 qubits, post-selection on the
     ancilla (and on N = 2 in the Z group). The 6 extra CNOTs are noisy.
     Variant "parity + mod 4": a second ancilla runs a Hadamard test of
     U = exp(i pi N / 2) = S_0 S_1 ... S_5 (six controlled phases); on the
     even sector U = +-1, and outcome 1 means N = 2 mod 4. This also rejects
     the errors that change N by 2, which parity cannot see.

Noise: the generic all-to-all model of §29 (depolarizing 2-qubit gates,
IonQ-like, no routing), and heavy-hex fake_brisbane for part A.

Findings (python examples/lih_parity_verification_qg.py):

  A. Ideal projections, exact density matrix (mHa vs noiseless circuit;
     kept fraction in brackets):

                     no check   parity         N mod 4       N = 2 (ceiling)
    all-to-all L1    23.7       15.3 (0.93)    0.3 (0.91)    0.2 (0.91)
    all-to-all L2    49.6       31.7 (0.88)    2.5 (0.83)    2.2 (0.83)
    all-to-all L3    76.1       49.4 (0.83)    6.9 (0.77)    6.2 (0.77)
    brisbane   L1    41.0       25.5 (0.86)    0.7 (0.82)    0.4 (0.82)
    brisbane   L2    87.3       58.4 (0.76)    5.8 (0.69)    4.1 (0.69)
    brisbane   L3   135.6       98.4 (0.69)   17.3 (0.59)   13.3 (0.59)

  * Symmetry verification works for LiH once it reaches every group: the
    full number projection cuts the error 10-100x. The failure of §21/§50
    was reach, as §50 concluded, not a limit of symmetry verification.
  * Parity alone recovers only about a third (76 -> 49 mHa): most of the
    residual errors change N by 2 (a two-qubit error flipping both qubits
    of a pair), which parity cannot see. N mod 4 catches them and lands
    within 10-30 % of the full ceiling.

  B. With ancillas, 200,000 shots, all-to-all noise:

                     raw*    checked   + N filter on Z group   kept   extra CX
    L1 parity        25.2    15.9      13.7                    0.92    6
    L1 parity+mod4   30.1     3.8       4.5                    0.86   18
    L2 parity        52.7    33.7      26.0                    0.86    6
    L2 parity+mod4   56.7     6.9       6.0                    0.78   18
    L3 parity        79.2    52.1      49.7                    0.81    6
    L3 parity+mod4   83.9    13.4      14.8                    0.72   18
    (* raw includes the noise the ancilla gates add to the system)

  * The ancilla implementation keeps almost all of the ideal gain even
    though it adds 6-18 noisy two-qubit gates: parity 52 vs 49 ideal,
    parity + mod 4 13.4 vs 6.9 ideal at L3.
  * Parity + mod 4 cuts the LiH error 6-8x (L3: 83.9 -> 13.4 mHa; §50's
    best was 76.6). At L3, whose noiseless energy is 0.66 mHa above FCI,
    that is about 14 mHa above FCI, just below classical Hartree-Fock
    (16.3 mHa) on this noise model: the first LiH configuration in this
    repository where the noisy quantum estimate beats Hartree-Fock.
  * The cost: 1-2 ancillas that must couple to every qubit (natural on
    trapped ions, SWAP-heavy on heavy-hex) and 14-28 % of the shots
    discarded.

Honest scope. Parity checks with an ancilla are standard symmetry
verification (Bonet-Monroig et al. 2018); the N mod 4 Hadamard test is
the obvious extension. The qg content is the diagnosis that led here
(§50: the error sits in the X/Y groups) and the witness reading of the
kept fraction. Simulated noise; the brisbane numbers are ideal
projections only (the ancilla circuits were not routed on heavy-hex).
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import chemistry_lih_deep_circuit as L  # noqa: E402
import filter_scaling_lih_qg as F  # noqa: E402

N, NE = L.N_QUBITS, L.N_ELECTRONS
SHOTS = 200_000
_W = np.array([bin(i).count("1") for i in range(2**N)])


def _noisy_density(circ, device):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import DensityMatrix

    if device == "brisbane":
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane

        from nisq_hardware_validation import choose_layout

        backend = FakeBrisbane()
        tc = transpile(circ, backend=backend, initial_layout=choose_layout(backend, N), optimization_level=1,
                       seed_transpiler=1)
        # keep only the logical qubits: the final layout maps logical i -> physical
        sim = AerSimulator.from_backend(backend, method="density_matrix")
        phys = tc.layout.final_index_layout()[:N]
        tc.save_density_matrix(qubits=phys)
    else:
        import hubbard_trotter_qg_filters as HB

        tc = transpile(circ, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=1)
        sim = AerSimulator(method="density_matrix", noise_model=HB.all_to_all_noise_model())
        tc.save_density_matrix()
    res = sim.run(tc).result()
    return np.asarray(DensityMatrix(res.data(0)["density_matrix"]).data)


def _project(rho, diag):
    P = np.diag(diag.astype(float))
    r = P @ rho @ P
    return r / np.real(np.trace(r)), float(np.real(np.trace(r)))


def exact_levels(layers=3, device="all_to_all"):
    from qiskit.quantum_info import Statevector

    circ = L.ansatz(L.PARAMS, layers)
    H = L.HAMILTONIAN.to_matrix()
    e_ideal = float(np.real(Statevector(circ).expectation_value(L.HAMILTONIAN)))
    rho = _noisy_density(circ, device)
    parity_even = (_W % 2) == (NE % 2)
    rho_p, keep_p = _project(rho, parity_even)
    rho_n, keep_n = _project(rho, _W == NE)
    rho_4, keep_4 = _project(rho, (_W % 4) == (NE % 4))
    e = lambda r: float(np.real(np.trace(r @ H)))  # noqa: E731
    return {"raw": 1e3 * (e(rho) - e_ideal), "parity": 1e3 * (e(rho_p) - e_ideal),
            "number": 1e3 * (e(rho_n) - e_ideal), "mod4": 1e3 * (e(rho_4) - e_ideal),
            "kept_parity": keep_p, "kept_number": keep_n, "kept_mod4": keep_4}


# --------------------------------------------------------------------- #
# B. ancilla parity check with shots
# --------------------------------------------------------------------- #
def _parity_circuits(layers, mod4=False):
    """Ancilla N measures the parity (CNOTs); with mod4, ancilla N+1 runs a
    Hadamard test of U = exp(i pi N_el / 2) = prod_j S_j (controlled phases),
    which on the even sector reads N_el mod 4 (outcome 1 <-> N_el = 2 mod 4)."""
    from qiskit import QuantumCircuit

    h = L.HAMILTONIAN
    groups, bases = F._groups(h, N)
    na = 2 if mod4 else 1
    circs = []
    for b in bases:
        qc = QuantumCircuit(N + na)
        qc.compose(L.ansatz(L.PARAMS, layers), qubits=range(N), inplace=True)
        for q in range(N):
            qc.cx(q, N)
        if mod4:
            qc.h(N + 1)
            for q in range(N):
                qc.cp(math.pi / 2, N + 1, q)
            qc.h(N + 1)
        for q, ch in enumerate(b):
            if ch == "X":
                qc.h(q)
            elif ch == "Y":
                qc.sdg(q)
                qc.h(q)
        qc.measure_all()
        circs.append(qc)
    for bit in (0, 1):
        qc = QuantumCircuit(N + na)
        if bit:
            qc.x(range(N + na))
        qc.measure_all()
        circs.append(qc)
    return groups, bases, circs


def ancilla_check(layers=3, seed=11, mod4=False):
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator

    import hubbard_trotter_qg_filters as HB

    groups, bases, circs = _parity_circuits(layers, mod4)
    tc = transpile(circs, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=1)
    res = AerSimulator(noise_model=HB.all_to_all_noise_model()).run(tc, shots=SHOTS, seed_simulator=seed).result()
    n1 = N + (2 if mod4 else 1)
    P = []
    for i in range(len(tc)):
        p = np.zeros(2**n1)
        for key, c in res.get_counts(i).items():
            p[int(key.replace(" ", ""), 2)] += c
        P.append(p / SHOTS)
    inv = F._readout_inverse(P[-2], P[-1], n1)
    anc = (np.arange(2**n1) >> N) & 1
    accept = anc == 0
    if mod4:
        accept = accept & (((np.arange(2**n1) >> (N + 1)) & 1) == 1)
    sysidx = np.arange(2**n1) & (2**N - 1)
    z = [k for k, b in enumerate(bases) if all(ch in "IZ" for ch in b)][0]
    ideal_groups = []
    for k, b in enumerate(bases):
        qc = L.ansatz(L.PARAMS, layers).copy()
        for q, ch in enumerate(b):
            if ch == "X":
                qc.h(q)
            elif ch == "Y":
                qc.sdg(q)
                qc.h(q)
        ideal_groups.append(np.real(Statevector(qc).probabilities()))
    e_ideal = sum(F._group_energy(g, p, N) for g, p in zip(groups, ideal_groups))
    tot = {"raw": 0.0, "parity": 0.0, "parity_plus_n": 0.0}
    kept = []
    for k, (g, p7) in enumerate(zip(groups, P[:-2])):
        pm = np.clip(inv @ p7, 0, None)
        sys_all = np.bincount(sysidx, weights=pm, minlength=2**N)
        sys_even = np.bincount(sysidx, weights=pm * accept, minlength=2**N)
        kept.append(sys_even.sum() / sys_all.sum())
        e_raw = F._group_energy(g, sys_all / sys_all.sum(), N)
        e_par = F._group_energy(g, sys_even / sys_even.sum(), N)
        tot["raw"] += e_raw
        tot["parity"] += e_par
        if k == z:
            nf = sys_even * (_W == NE)
            tot["parity_plus_n"] += F._group_energy(g, nf / nf.sum(), N)
        else:
            tot["parity_plus_n"] += e_par
    out = {k: 1e3 * (v - e_ideal) for k, v in tot.items()}
    out["kept_mean"] = float(np.mean(kept))
    out["extra_cx"] = tc[0].count_ops().get("cx", 0) - (
        transpile(L.ansatz(L.PARAMS, layers), basis_gates=["cx", "rz", "sx", "x"], optimization_level=1,
                  seed_transpiler=1).count_ops().get("cx", 0))
    return out


def make_figure(path, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{d}\nL{l}" for d, l in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    keys = [("raw", "#8c8c8c", "no check"), ("parity", "#1f6fb2", "ideal parity check (all groups)"),
            ("mod4", "#e0a030", "ideal N mod 4 check (all groups)"),
            ("number", "#8c2d04", "ideal N projection (all groups, ceiling)")]
    for i, (k, c, lab) in enumerate(keys):
        ax.bar(x + (i - 1.5) * 0.2, [abs(r[k]) for r in rows.values()], 0.2, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("|energy error| vs noiseless circuit (mHa)")
    ax.set_title("LiH: symmetry checks that reach every measurement group", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    rows = {}
    print("A. exact density matrix, ideal projections (mHa vs noiseless circuit)")
    for dev in ("all_to_all", "brisbane"):
        for layers in (1, 2, 3):
            r = exact_levels(layers, dev)
            rows[(dev, layers)] = r
            print(f"  {dev:10s} L{layers}: raw {r['raw']:7.1f}  parity {r['parity']:7.1f} (kept {r['kept_parity']:.3f})"
                  f"  mod4 {r['mod4']:7.1f} (kept {r['kept_mod4']:.3f})  number {r['number']:7.1f} (kept {r['kept_number']:.3f})")
    print("\nB. ancilla parity check with shots (all-to-all), mHa vs noiseless circuit")
    for layers in (1, 2, 3):
        for m4 in (False, True):
            b = ancilla_check(layers, mod4=m4)
            tag = "parity + mod-4" if m4 else "parity        "
            print(f"  L{layers} {tag}: raw {b['raw']:7.1f}  checked {b['parity']:7.1f}  + N filter on Z group "
                  f"{b['parity_plus_n']:7.1f}  (kept {b['kept_mean']:.3f}, +{b['extra_cx']} CX)")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"), rows)
