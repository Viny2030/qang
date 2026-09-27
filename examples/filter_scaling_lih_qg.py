"""
Why the qg filter helps H2 and not LiH (§21), and what would reach the
part it cannot.

§21 found the electron-number filter cutting the H2 error 3x but doing
nothing for LiH on 6 qubits, and read the witness as "unital scrambling
from 60 routed two-qubit gates". This study splits that explanation into
parts that can each be measured:

  1. Reach. The filter acts on the Z-basis shots only. Every Hamiltonian
     group measured in a rotated (X/Y) basis is untouched. H2 has 5
     measurement groups; LiH has 17, 16 of them off-diagonal. The share of
     the energy error that sits in the off-diagonal groups bounds what any
     Z-basis filter can do.
  2. Routing. The same LiH circuit on an all-to-all device (IonQ-like, no
     SWAPs) against heavy-hex fake_brisbane.
  3. Depth. LiH with 1, 2 and 3 ansatz layers.

and tests one qg-based remedy for the off-diagonal part: under global
depolarizing noise every traceless expectation value shrinks by (1 -
delta), and the filter's kept fraction K measures delta without extra
circuits:

    K = (1 - delta) + delta C(n, N)/2^n   =>   1 - delta = (K - c)/(1 - c),
    c = C(n, N)/2^n (the noise floor of §24),

so the off-diagonal groups are divided by (1 - delta). This is the
standard global-depolarizing rescaling with the depolarizing strength read
from the witness instead of from a calibration circuit.

All errors are measured against the NOISELESS energy of the same circuit
(not FCI), so the ansatz error is not mixed with the hardware error.
Readout errors are corrected with the two calibration circuits in every
method. 200,000 shots per circuit (shot noise below 1 mHa).

Findings (python examples/filter_scaling_lih_qg.py):

  Energy error vs the noiseless circuit (mHa); "Z" is the single all-Z
  group the filter acts on, "X/Y" the other groups:

    case             2q  groups kept  | Z raw  Z filt | X/Y raw  X/Y resc | total raw  +filter  ceiling
    H2  all-to-all    3    5   0.991  |  13.0   4.9   |   0.9      0.3    |   13.9      5.9      0.9
    H2  brisbane      6    5   0.973  |  20.6   2.8   |   2.4      0.7    |   23.0      5.2      2.4
    LiH all-to-all L1 20  17   0.910  |  -2.2  -1.9   |  26.2    -41.6    |   24.0     24.3     26.2
    LiH all-to-all L2 40  17   0.833  |   7.7  -2.2   |  41.5    -68.3    |   49.1     39.3     41.5
    LiH all-to-all L3 60  17   0.766  | -20.8 -12.4   |  97.4   -124.0    |   76.6     85.0     97.4
    LiH brisbane   L1 20  17   0.821  |  -0.8  -1.5   |  42.2   -107.4    |   41.4     40.7     42.2
    LiH brisbane   L2 40  17   0.686  |  14.4  -2.0   |  71.9   -179.2    |   86.3     69.8     71.9
    LiH brisbane   L3 60  17   0.585  | -30.5 -18.0   | 165.8   -346.3    |  135.3    147.9    165.8

    (ceiling = Z group exact, X/Y groups as measured: the best any Z-basis
    filter could do; "2q" = CX on the all-to-all model, ECR on brisbane.)

  * The main reason is reach, not routing. In H2 the hardware error sits
    in the Z group (13.0 of 13.9 mHa on the all-to-all model), which the
    filter can reach. In LiH it sits in the 16 X/Y groups (26-166 mHa),
    which it cannot; even a perfect Z-basis filter would leave the
    "ceiling" column. The LiH ansatz is nearest-neighbour, so it needs the
    same 20/40/60 two-qubit gates with or without all-to-all connectivity:
    §21's "60 routed gates" was not routing, and removing the heavy-hex
    device cuts the error by about 40 % without changing the picture.
  * Scrambling is real but secondary: the kept fraction falls from 0.91 to
    0.59 with depth, and the filter does reduce the Z-group error
    (14.4 -> -2.0 mHa at L2 on brisbane), but that group carries a small
    part of the LiH energy error.
  * The qg remedy for the X/Y groups fails. Rescaling them by the
    depolarizing strength read from the kept fraction overcorrects by a
    factor 2-3 (e.g. +97 -> -124 mHa): the kept fraction falls faster than
    the off-diagonal signals shrink. The actual shrink of the X/Y groups
    is 0.72-0.95 against 0.46-0.88 inferred from K, and their ratio (0.34-
    0.53 for LiH, above 1 for H2) is neither constant nor predictable from
    K alone. A global-depolarizing reading of the witness is wrong for
    local gate noise.

Scaling implication. For larger molecules the X/Y groups grow in number
(~n^4 terms) and carry most of the correlation energy, so a Z-basis
number filter reaches a shrinking share of the error. qg-style symmetry
checks would have to act inside the rotated bases (for example the
electron-number parity, which commutes with every number-conserving
term, measured with an ancilla or in an entangled basis): not tested here.

Honest scope. Two noise models, one molecule size beyond H2, no shot
budget study; the negative result for rescaling is about this simple
witness-based estimate, not about calibrated rescaling methods.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SHOTS = 200_000


def _molecule(name, layers=3):
    """(hamiltonian, circuit, n_qubits, n_electrons)."""
    if name == "H2":
        import chemistry_qg_symmetry_witness as CH

        return CH.H2_JW, CH.ansatz(CH.optimal_angle()), CH.N_QUBITS, CH.N_ELECTRONS
    import chemistry_lih_deep_circuit as L

    return L.HAMILTONIAN, L.ansatz(L.PARAMS, layers), L.N_QUBITS, L.N_ELECTRONS


def _groups(h, n):
    groups = h.group_commuting(qubit_wise=True)
    bases = []
    for g in groups:
        b = ["I"] * n
        for lab in g.paulis.to_labels():
            for q, ch in enumerate(reversed(lab)):
                if ch != "I":
                    b[q] = ch
        bases.append(b)
    return groups, bases


def _circuits(circ, bases, n):
    from qiskit import QuantumCircuit

    out = []
    for b in bases:
        qc = circ.copy()
        for q, ch in enumerate(b):
            if ch == "X":
                qc.h(q)
            elif ch == "Y":
                qc.sdg(q)
                qc.h(q)
        qc.measure_all()
        out.append(qc)
    for bit in (0, 1):
        qc = QuantumCircuit(n)
        if bit:
            qc.x(range(n))
        qc.measure_all()
        out.append(qc)
    return out


def _simulate(circuits, n, device, seed=11):
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    if device == "brisbane":
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane

        from nisq_hardware_validation import choose_layout

        backend = FakeBrisbane()
        tc = transpile(circuits, backend=backend, initial_layout=choose_layout(backend, n),
                       optimization_level=1, seed_transpiler=1)
        sim = AerSimulator.from_backend(backend)
        two_q = tc[0].count_ops().get("ecr", 0)
    else:
        import hubbard_trotter_qg_filters as HB

        tc = transpile(circuits, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=1)
        sim = AerSimulator(noise_model=HB.all_to_all_noise_model())
        two_q = tc[0].count_ops().get("cx", 0)
    res = sim.run(tc, shots=SHOTS, seed_simulator=seed).result()
    P = []
    for i in range(len(tc)):
        p = np.zeros(2**n)
        for key, c in res.get_counts(i).items():
            p[int(key.replace(" ", ""), 2)] += c
        P.append(p / SHOTS)
    return P, two_q


def _ideal(circuits, n):
    from qiskit.quantum_info import Statevector

    out = []
    for qc in circuits[:-2]:
        c = qc.remove_final_measurements(inplace=False)
        out.append(np.real(Statevector(c).probabilities()))
    return out


def _readout_inverse(p0, p1, n):
    inv = np.array([[1.0]])
    for q in range(n - 1, -1, -1):
        e01 = sum(p0[i] for i in range(2**n) if (i >> q) & 1)
        e10 = sum(p1[i] for i in range(2**n) if not (i >> q) & 1)
        inv = np.kron(inv, np.linalg.inv(np.array([[1 - e01, e10], [e01, 1 - e10]])))
    return inv


def _group_energy(g, p, n):
    e = 0.0
    for lab, c in zip(g.paulis.to_labels(), g.coeffs):
        sup = [q for q, ch in enumerate(reversed(lab)) if ch != "I"]
        signs = np.array([(-1) ** (sum((i >> q) & 1 for q in sup) % 2) for i in range(2**n)])
        e += c.real * float(np.sum(p * signs))
    return e


def _identity_part(g):
    return sum(c.real for lab, c in zip(g.paulis.to_labels(), g.coeffs) if set(lab) == {"I"})


def analyse(name="LiH", device="all_to_all", layers=3, seed=11):
    h, circ, n, n_el = _molecule(name, layers)
    groups, bases = _groups(h, n)
    circuits = _circuits(circ, bases, n)
    P, two_q = _simulate(circuits, n, device, seed)
    Pi = _ideal(circuits, n)
    inv = _readout_inverse(P[-2], P[-1], n)
    weight = np.array([bin(i).count("1") for i in range(2**n)])
    c = math.comb(n, n_el) / 2**n
    z_idx = [k for k, b in enumerate(bases) if all(ch in "IZ" for ch in b)]
    assert len(z_idx) == 1
    z = z_idx[0]
    pz = np.clip(inv @ P[z], 0, None)
    pz /= pz.sum()
    K = float(np.sum(pz * (weight == n_el)))
    shrink = max((K - c) / (1 - c), 1e-3)
    ez_ideal = _group_energy(groups[z], Pi[z], n)
    ez_raw = _group_energy(groups[z], pz, n)
    kept = pz * (weight == n_el)
    ez_filt = _group_energy(groups[z], kept / kept.sum(), n)
    off_ideal = off_raw = off_resc = off_const = 0.0
    for k, (g, p) in enumerate(zip(groups, P[:-2])):
        if k == z:
            continue
        pm = inv @ p
        e_id, e_n = _group_energy(g, Pi[k], n), _group_energy(g, pm, n)
        const = _identity_part(g)
        off_ideal += e_id
        off_raw += e_n
        off_resc += const + (e_n - const) / shrink
        off_const += const
    m = 1e3
    return {
        "two_qubit_gates": two_q, "groups": len(groups), "kept": K, "floor": c,
        "mean_qg_z": float(np.sum(pz * (1 - 2 * weight / n))), "ideal_mean_qg_z": 1 - 2 * n_el / n,
        "z_raw": m * (ez_raw - ez_ideal), "z_filter": m * (ez_filt - ez_ideal),
        "off_raw": m * (off_raw - off_ideal), "off_rescaled": m * (off_resc - off_ideal),
        "total_raw": m * (ez_raw + off_raw - ez_ideal - off_ideal),
        "total_filter": m * (ez_filt + off_raw - ez_ideal - off_ideal),
        "total_filter_rescaled": m * (ez_filt + off_resc - ez_ideal - off_ideal),
        "ceiling_exact_z": m * (off_raw - off_ideal),
        "shrink_from_kept": shrink,
        "shrink_actual": (off_raw - off_const) / (off_ideal - off_const),
    }


CASES = [("H2", "all_to_all", 3), ("H2", "brisbane", 3), ("LiH", "all_to_all", 1), ("LiH", "all_to_all", 2),
         ("LiH", "all_to_all", 3), ("LiH", "brisbane", 1), ("LiH", "brisbane", 2), ("LiH", "brisbane", 3)]


def study(cases=CASES):
    return {c: analyse(*c) for c in cases}


def make_figure(path, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{m}\n{'a2a' if d == 'all_to_all' else 'brisbane'}" + (f"\nL{l}" if m == "LiH" else "") for m, d, l in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.2))
    keys = [("total_raw", "#8c8c8c", "readout-corrected"), ("total_filter", "#1f6fb2", "+ qg filter (Z group)"),
            ("total_filter_rescaled", "#8c2d04", "+ kept-fraction rescaling of X/Y groups"),
            ("ceiling_exact_z", "#e0a030", "ceiling: Z group exact, X/Y raw")]
    w = 0.2
    for i, (k, col, lab) in enumerate(keys):
        ax.bar(x + (i - 1.5) * w, [abs(r[k]) for r in rows.values()], w, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("|energy error| vs noiseless circuit (mHa)")
    ax.set_title("A Z-basis filter cannot reach the X/Y groups; rescaling them by the kept fraction overcorrects", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    rows = study()
    print(f"{'case':22s} {'2q':>4} {'grp':>3} {'kept':>5} {'floor':>5} | {'Z raw':>7} {'Z filt':>7} | "
          f"{'off raw':>8} {'off resc':>8} | {'tot raw':>8} {'+filter':>8} {'+resc':>8} {'ceiling':>8}")
    for (m, d, l), r in rows.items():
        name = f"{m} {d} L{l}" if m == "LiH" else f"{m} {d}"
        print(f"{name:22s} {r['two_qubit_gates']:4d} {r['groups']:3d} {r['kept']:5.3f} {r['floor']:5.3f} | "
              f"{r['z_raw']:7.1f} {r['z_filter']:7.1f} | {r['off_raw']:8.1f} {r['off_rescaled']:8.1f} | "
              f"{r['total_raw']:8.1f} {r['total_filter']:8.1f} {r['total_filter_rescaled']:8.1f} {r['ceiling_exact_z']:8.1f}")
    if "--figure" in sys.argv:
        make_figure(__file__.replace(".py", ".png"), rows)
