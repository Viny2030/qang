"""
Quantum chemistry: classical methods vs a noisy quantum computation, with
and without a qg-based correction.

H2 (STO-3G, 0.735 Angstrom) in the Jordan-Wigner encoding: 4 qubits, one
per spin-orbital, 2 electrons. The Hamiltonian below was derived once with
PySCF + OpenFermion and is hard-coded, like the other molecules in this
repository.

Why qg is natural here. In the Jordan-Wigner encoding the occupation of
spin-orbital i is n_i = (1 - Z_i)/2, so the number of electrons is

    N = sum_i (1 - Z_i)/2   and   mean qg_Z = 1 - 2N/n.

For any number-conserving circuit the ideal register mean qg_Z is
therefore KNOWN in advance (0 here: N = 2, n = 4), whatever the
parameters. That fixes the limitation found in RESEARCH_NOTES §18 (on an
arbitrary circuit the ideal mean qg_Z is unknown, so a T1 shift cannot be
seen): here any deviation of mean qg_Z from 1 - 2N/n is a
reference-free witness of electrons lost or gained by the hardware
(mostly T1 decay, which pushes qg_Z towards +1). The same quantity, taken
shot by shot, gives the correction: keep only the Z-basis shots whose
register qg_Z equals 1 - 2N/n (Hamming weight N). This is the standard
"symmetry verification" of the error-mitigation literature (Bonet-Monroig
et al. 2018; McArdle et al. 2019) read in qg units -- the novelty is the
reading and the witness, not the filter.

Methods compared (energy error vs the exact FCI energy, mHa):

  classical  Hartree-Fock (mean field, polynomial cost)
  classical  FCI (exact diagonalization; exponential cost in general,
             trivial at this size -- it is the reference)
  quantum    raw: the exact VQE state (it reaches FCI without noise), run
             on the calibration-based fake_brisbane noise model
  quantum    + standard readout-error mitigation (tensored inversion of
             per-qubit confusion matrices, the usual first step)
  quantum    + readout mitigation + qg filter (per-shot qg_Z = 1 - 2N/n
             on the Z-basis group; the 4 XXYY-type terms cannot be
             filtered this way and keep readout mitigation only)

Findings (20,000 shots per circuit; mean over 3 simulator seeds; an idle
delay before measurement amplifies T1):

    delay | mean qg_Z | kept | raw   | +readout | +readout +qg filter | HF
    0 us  |  +0.011   | 0.91 |  74   |   18     |        5.5          | 20.3
    20 us |  +0.057   | 0.83 | 140   |   90     |       20            | 20.3
    50 us |  +0.122   | 0.72 | 231   |  188     |       30            | 20.3

  * The witness works: mean qg_Z rises from its known ideal 0 in step
    with the fraction of shots that lost an electron.
  * With the qg filter the noisy quantum energy (5.5 mHa at zero delay)
    is 3.2x better than readout mitigation alone and 3.7x better than
    classical Hartree-Fock (20.3 mHa); the filter's gain grows with T1
    (4.5x at 20 us, 6x at 50 us).
  * It is still ~3.4x above chemical accuracy (1.6 mHa); the residual
    comes mostly from the four XXYY-type terms, which this filter
    cannot reach, and from gate errors that preserve N.
  * Classically, FCI is exact and cheap at 4 qubits. The comparison
    shows what the qg correction buys a quantum computation, not a
    quantum advantage over classical chemistry.

Dissociation curve (no delay, mean of 3 seeds; Hamiltonians in
examples/data/h2_dissociation_jw.json). Stretching the bond makes
Hartree-Fock fail while the noisy quantum energy with the qg filter stays
within ~6-20 mHa, so the quantum + qg estimate beats classical
Hartree-Fock at every geometry and by 11x at 2.5 Angstrom:

    R (A) | HF    | raw  | +readout | +readout +qg filter
    0.50  |  12.2 |  92  |   19     |   5.9
    0.735 |  20.3 |  74  |   18     |   5.5
    1.00  |  35.0 |  63  |   17     |   6.4
    1.50  |  87.3 |  60  |   20     |  11.4
    2.00  | 164.8 |  67  |   22     |  16.3
    2.50  | 233.1 |  72  |   24     |  20.1

  The filter's gain over readout mitigation alone shrinks as the bond
  stretches (3.2x at 0.735 A, 1.2x at 2.5 A). For a deeper 6-qubit
  circuit where the filter stops helping see
  examples/chemistry_lih_deep_circuit.py.
"""

import os
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

H2_JW = SparsePauliOp.from_list([
    ("IIII", -0.090578986088), ("IIIZ", 0.172183932619), ("IIZI", 0.172183932619),
    ("IIZZ", 0.168927538701), ("IZII", -0.225753492224), ("IZIZ", 0.120912632618),
    ("IZZI", 0.166145432564), ("XXYY", -0.045232799946), ("XYYX", 0.045232799946),
    ("YXXY", 0.045232799946), ("YYXX", -0.045232799946), ("ZIII", -0.225753492224),
    ("ZIIZ", 0.166145432564), ("ZIZI", 0.120912632618), ("ZZII", 0.174643430683),
])
FCI_ENERGY = -1.1373060357534004
HF_ENERGY = -1.116998996754004
CHEMICAL_ACCURACY = 1.6e-3
N_QUBITS, N_ELECTRONS = 4, 2
IDEAL_MEAN_QG_Z = 1.0 - 2.0 * N_ELECTRONS / N_QUBITS  # = 0

def _terms(h):
    return [(p.to_label(), float(c.real)) for p, c in zip(h.paulis, h.coeffs)]


_TERMS = _terms(H2_JW)
XY_TERMS = [l for l, _ in _TERMS if any(ch in "XY" for ch in l)]
_WEIGHT = np.array([bin(i).count("1") for i in range(2**N_QUBITS)])


def ansatz(t: float) -> QuantumCircuit:
    """Number-conserving: cos(t/2)|0011> + sin(t/2)|1100> (qubits 0,1 hold
    the Hartree-Fock electrons). Hamming weight is 2 for every t."""
    qc = QuantumCircuit(N_QUBITS)
    qc.x(0)
    qc.x(1)
    qc.ry(t, 2)
    qc.cx(2, 3)
    qc.cx(2, 0)
    qc.cx(2, 1)
    return qc


def exact_energy(t: float, hamiltonian=H2_JW) -> float:
    return float(Statevector(ansatz(t)).expectation_value(hamiltonian).real)


def optimal_angle(hamiltonian=H2_JW) -> float:
    res = minimize_scalar(lambda t: exact_energy(t, hamiltonian), bounds=(-np.pi, np.pi),
                          method="bounded", options={"xatol": 1e-12})
    return float(res.x)


def _measure_circuit(t, label, delay_us):
    qc = ansatz(t)
    if delay_us > 0:
        qc.barrier()
        for q in range(N_QUBITS):
            qc.delay(delay_us, q, unit="us")
    for q, ch in enumerate(reversed(label)):
        if ch == "X":
            qc.h(q)
        elif ch == "Y":
            qc.sdg(q)
            qc.h(q)
    qc.measure_all()
    return qc


def _calibration_circuits():
    out = []
    for bit in (0, 1):
        qc = QuantumCircuit(N_QUBITS)
        if bit:
            qc.x(range(N_QUBITS))
        qc.measure_all()
        out.append(qc)
    return out


def _probs(counts, shots):
    p = np.zeros(2**N_QUBITS)
    for key, c in counts.items():
        p[int(key.replace(" ", ""), 2)] += c
    return p / shots


def _expval(p, label):
    support = [q for q, ch in enumerate(reversed(label)) if ch != "I"]
    signs = np.array([(-1) ** (sum((i >> q) & 1 for q in support) % 2) for i in range(len(p))])
    return float(np.sum(p * signs))


def _energy(p_z, p_xy, terms=None):
    e = 0.0
    for label, c in (terms or _TERMS):
        if label == "IIII":
            e += c
        elif label in p_xy:
            e += c * _expval(p_xy[label], label)
        else:
            e += c * _expval(p_z, label)
    return e


def _readout_inverse(p_all0, p_all1):
    """Tensored inverse of per-qubit confusion matrices."""
    inv = np.array([[1.0]])
    for q in range(N_QUBITS - 1, -1, -1):
        e01 = sum(p_all0[i] for i in range(len(p_all0)) if (i >> q) & 1)
        e10 = sum(p_all1[i] for i in range(len(p_all1)) if not (i >> q) & 1)
        inv = np.kron(inv, np.linalg.inv(np.array([[1 - e01, e10], [e01, 1 - e10]])))
    return inv


def qg_filter(p_z):
    """Keep only outcomes whose register qg_Z equals 1 - 2N/n (Hamming
    weight N) and renormalize."""
    kept = p_z * (_WEIGHT == N_ELECTRONS)
    return kept / kept.sum()


def run_noisy(delay_us=0.0, shots=20000, seed=11, backend=None, layout=None, hamiltonian=H2_JW):
    """Energies (Hartree) and diagnostics on a noisy backend."""
    from qiskit_aer import AerSimulator

    if backend is None:
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane

        backend = FakeBrisbane()
    if layout is None:
        from nisq_hardware_validation import choose_layout

        layout = choose_layout(backend, N_QUBITS)
    t = optimal_angle(hamiltonian)
    terms = _terms(hamiltonian)
    circuits = [_measure_circuit(t, "ZZZZ", delay_us)] + [_measure_circuit(t, l, delay_us) for l in XY_TERMS]
    tc = transpile(circuits + _calibration_circuits(), backend=backend, initial_layout=layout,
                   optimization_level=1, seed_transpiler=1, scheduling_method="alap")
    result = AerSimulator.from_backend(backend).run(tc, shots=shots, seed_simulator=seed).result()
    P = [_probs(result.get_counts(i), shots) for i in range(len(tc))]
    inv = _readout_inverse(P[-2], P[-1])
    p_z, p_xy = P[0], {l: P[1 + k] for k, l in enumerate(XY_TERMS)}
    p_z_ro, p_xy_ro = inv @ p_z, {l: inv @ p for l, p in p_xy.items()}
    return {
        "delay_us": delay_us,
        "mean_qg_z": float(np.sum(p_z * (1 - 2 * _WEIGHT / N_QUBITS))),
        "kept_fraction": float(np.sum(p_z * (_WEIGHT == N_ELECTRONS))),
        "raw": _energy(p_z, p_xy, terms),
        "readout": _energy(p_z_ro, p_xy_ro, terms),
        "readout_qg_filter": _energy(qg_filter(p_z_ro), p_xy_ro, terms),
    }


def dissociation_curve(seeds=(11, 12, 13)):
    """Rows (R, HF error, raw, readout, readout+qg) in Hartree, no delay."""
    import json

    from qiskit.quantum_info import SparsePauliOp as _SPO

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "h2_dissociation_jw.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = []
    for r_str, v in data.items():
        h = _SPO.from_list(list(v["terms"].items()))
        runs = [run_noisy(0, seed=s, hamiltonian=h) for s in seeds]
        m = lambda k: float(np.mean([x[k] for x in runs]))
        rows.append((float(r_str), v["hf"] - v["fci"], m("raw") - v["fci"],
                     m("readout") - v["fci"], m("readout_qg_filter") - v["fci"]))
    return rows


if __name__ == "__main__":
    t = optimal_angle()
    print(f"Classical  Hartree-Fock error: {1e3 * (HF_ENERGY - FCI_ENERGY):6.1f} mHa")
    print(f"Classical  FCI (reference):    {0.0:6.1f} mHa")
    print(f"Noiseless VQE with this ansatz: {1e3 * (exact_energy(t) - FCI_ENERGY):.1e} mHa\n")
    print(f"{'delay':>6} | {'mean qg_Z':>9} {'kept':>5} | {'raw':>6} {'+readout':>9} {'+readout+qg':>12}   (mHa, mean of 3 seeds)")
    for delay in (0, 20, 50):
        rows = [run_noisy(delay, seed=s) for s in (11, 12, 13)]
        m = lambda k: np.mean([r[k] for r in rows])
        print(f"{delay:4d}us | {m('mean_qg_z'):+9.3f} {m('kept_fraction'):5.2f} | "
              f"{1e3 * (m('raw') - FCI_ENERGY):6.1f} {1e3 * (m('readout') - FCI_ENERGY):9.1f} "
              f"{1e3 * (m('readout_qg_filter') - FCI_ENERGY):12.1f}")
    print("\nDissociation curve (mHa above FCI, no delay):")
    print(f"{'R (A)':>6} | {'HF':>6} | {'raw':>6} {'+readout':>9} {'+readout+qg':>12}")
    for r, hf, raw, ro, qg in dissociation_curve():
        print(f"{r:6.3f} | {1e3 * hf:6.1f} | {1e3 * raw:6.1f} {1e3 * ro:9.1f} {1e3 * qg:12.1f}")
