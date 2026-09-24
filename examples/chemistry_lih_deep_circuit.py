"""
Where the qg filter stops helping: LiH at 3.0 Angstrom on 6 qubits.

Companion to examples/chemistry_qg_symmetry_witness.py (H2, 4 qubits,
3 CNOTs), here with a deeper circuit. LiH, STO-3G, frozen Li 1s, 2
electrons in 3 spatial orbitals, Jordan-Wigner: 6 qubits, 62 Pauli
terms in 17 qubit-wise commuting groups (Hamiltonian and optimized
parameters in examples/data/lih_3p0_jw.json, derived with PySCF +
OpenFermion). Ideal mean qg_Z = 1 - 2N/n = 1/3.

The ansatz alternates excitation-preserving XX+YY rotations with
controlled phases on neighbouring qubits (3 layers, 45 parameters); pure
Givens rotations alone cannot leave the Hartree-Fock (mean-field)
manifold. It reaches 0.66 mHa above FCI without noise, while classical
Hartree-Fock is 16.3 mHa above. Transpiled for fake_brisbane it needs 60
two-qubit (ECR) gates.

Finding (20,000 shots per group, 3 seeds): the noisy quantum energy is
~150 mHa above FCI raw, ~137 mHa with readout mitigation and ~150 mHa with
readout mitigation plus the qg filter -- the filter does NOT help, and
every quantum estimate is ~9x worse than classical Hartree-Fock. The
witness explains why: mean qg_Z falls from its ideal +0.333 to +0.18,
towards 0, not towards +1. With 60 noisy two-qubit gates the dominant
error is unital scrambling inside and across electron-number sectors, not
T1 loss of electrons; filtering keeps only 53% of the shots and the kept
ones are still scrambled. The qg witness correctly diagnoses the regime;
the qg filter is only useful when T1-type number violation dominates
(shallow circuits, long idle times), as in the H2 case.
"""

import json
import os
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp, Statevector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

with open(os.path.join(HERE, "data", "lih_3p0_jw.json"), encoding="utf-8") as fh:
    _DATA = json.load(fh)

HAMILTONIAN = SparsePauliOp.from_list(list(_DATA["terms"].items()))
FCI_ENERGY = _DATA["fci_active"]
PARAMS = np.array(_DATA["ansatz_params_L3"])
N_QUBITS, N_ELECTRONS = 6, 2
IDEAL_MEAN_QG_Z = 1.0 - 2.0 * N_ELECTRONS / N_QUBITS
_WEIGHT = np.array([bin(i).count("1") for i in range(2**N_QUBITS)])


def ansatz(params=PARAMS, layers=3) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)
    qc.x(0)
    qc.x(1)
    k = 0
    for _ in range(layers):
        for start in (0, 1):
            for a in range(start, N_QUBITS - 1, 2):
                qc.append(XXPlusYYGate(params[k], params[k + 1]), [a, a + 1])
                qc.cp(params[k + 2], a, a + 1)
                k += 3
    return qc


def hartree_fock_energy() -> float:
    hf = np.zeros(2**N_QUBITS)
    hf[0b000011] = 1.0
    return float(Statevector(hf).expectation_value(HAMILTONIAN).real)


def ideal_energy() -> float:
    return float(Statevector(ansatz()).expectation_value(HAMILTONIAN).real)


def _basis(group):
    b = ["I"] * N_QUBITS
    for lab in group.paulis.to_labels():
        for q, ch in enumerate(reversed(lab)):
            if ch != "I":
                b[q] = ch
    return b


def run_noisy(shots=20000, seed=11):
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane

    from nisq_hardware_validation import choose_layout

    backend = FakeBrisbane()
    groups = HAMILTONIAN.group_commuting(qubit_wise=True)
    bases = [_basis(g) for g in groups]
    circuits = []
    for b in bases:
        qc = ansatz()
        for q, ch in enumerate(b):
            if ch == "X":
                qc.h(q)
            elif ch == "Y":
                qc.sdg(q)
                qc.h(q)
        qc.measure_all()
        circuits.append(qc)
    for bit in (0, 1):
        qc = QuantumCircuit(N_QUBITS)
        if bit:
            qc.x(range(N_QUBITS))
        qc.measure_all()
        circuits.append(qc)
    tc = transpile(circuits, backend=backend, initial_layout=choose_layout(backend, N_QUBITS),
                   optimization_level=1, seed_transpiler=1)
    res = AerSimulator.from_backend(backend).run(tc, shots=shots, seed_simulator=seed).result()
    P = []
    for i in range(len(tc)):
        p = np.zeros(2**N_QUBITS)
        for key, c in res.get_counts(i).items():
            p[int(key.replace(" ", ""), 2)] += c
        P.append(p / shots)
    inv = np.array([[1.0]])
    for q in range(N_QUBITS - 1, -1, -1):
        e01 = sum(P[-2][i] for i in range(64) if (i >> q) & 1)
        e10 = sum(P[-1][i] for i in range(64) if not (i >> q) & 1)
        inv = np.kron(inv, np.linalg.inv(np.array([[1 - e01, e10], [e01, 1 - e10]])))

    def energy(filtered, readout):
        total = 0.0
        for g, b, p in zip(groups, bases, P[:-2]):
            pp = inv @ p if readout else p
            if filtered and all(ch in "IZ" for ch in b):
                pp = pp * (_WEIGHT == N_ELECTRONS)
                pp = pp / pp.sum()
            for lab, c in zip(g.paulis.to_labels(), g.coeffs):
                sup = [q for q, ch in enumerate(reversed(lab)) if ch != "I"]
                signs = np.array([(-1) ** (sum((i >> q) & 1 for q in sup) % 2) for i in range(64)])
                total += c.real * float(np.sum(pp * signs))
        return total

    z_index = [k for k, b in enumerate(bases) if all(ch in "IZ" for ch in b)][0]
    pz = P[z_index]
    return {
        "two_qubit_gates": tc[0].count_ops().get("ecr", 0),
        "mean_qg_z": float(np.sum(pz * (1 - 2 * _WEIGHT / N_QUBITS))),
        "kept_fraction": float(np.sum(pz * (_WEIGHT == N_ELECTRONS))),
        "raw": energy(False, False),
        "readout": energy(False, True),
        "readout_qg_filter": energy(True, True),
    }


if __name__ == "__main__":
    print(f"classical Hartree-Fock error: {1e3 * (hartree_fock_energy() - FCI_ENERGY):.1f} mHa")
    print(f"noiseless ansatz error:       {1e3 * (ideal_energy() - FCI_ENERGY):.2f} mHa")
    for seed in (11, 12, 13):
        r = run_noisy(seed=seed)
        print(f"seed {seed}: ECR {r['two_qubit_gates']}  mean qg_Z {r['mean_qg_z']:+.3f} (ideal {IDEAL_MEAN_QG_Z:+.3f})"
              f"  kept {r['kept_fraction']:.2f} | raw {1e3 * (r['raw'] - FCI_ENERGY):.1f}"
              f"  +readout {1e3 * (r['readout'] - FCI_ENERGY):.1f}"
              f"  +readout+qg {1e3 * (r['readout_qg_filter'] - FCI_ENERGY):.1f} mHa")
