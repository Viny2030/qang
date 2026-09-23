"""
NISQ validation of qg_S / mean qg_Z with device-level noise, and a path to
running the same experiments on real IBM Quantum hardware.

Two backends are supported through the same code path:

  * ``--mode fake`` (default): a calibration-based fake backend from
    qiskit-ibm-runtime (``fake_brisbane`` by default). It is a local Aer
    simulation that uses a real IBM device's published calibration data
    (T1, T2, gate and readout errors, coupling map). No account or
    network access is needed, and the results are reproducible.
  * ``--mode ibm``: a real IBM Quantum device, through
    qiskit-ibm-runtime's SamplerV2. Needs a saved IBM Quantum account
    (``QiskitRuntimeService.save_account(...)``). NOT yet run for this
    repository -- the numbers quoted below are all from fake_brisbane.

Experiment 1 (T1 relaxation, Findings C/D with device noise)
------------------------------------------------------------
The 4-qubit, depth-4 Quantum Volume circuit (seed=0) used throughout this
codebase is run on a connected line of 4 physical qubits, followed by an
idle delay of increasing length before measurement. During the delay each
qubit relaxes towards |0> (T1), exactly the amplitude-damping scenario of
examples/quantum_volume_qg_s_realistic_noise.py, but now with the device's
own per-qubit T1/T2 and its gate/readout errors on top.

  * Heavy output probability (HOP) and linear XEB fall at every step.
  * mean qg_Z rises at every step (towards +1, the T1 fixed point).
  * qg_S is NOT monotonic: device noise first raises it above the
    noiseless value, it keeps rising for short delays, then falls as
    relaxation takes over, ending BELOW the noiseless value at the
    longest delay. Read alone, that last qg_S would suggest a cleaner
    device than the noiseless circuit -- which is why qg_S must be
    reported together with mean qg_Z whenever T1 may matter.

Experiment 2 (LiH energy, the NISQ reality check)
-------------------------------------------------
The 4-qubit LiH ansatz from examples/lih_vqe_ry_rx_ansatz.py is
evaluated, without any error mitigation, at the Hartree-Fock point and at
the ansatz's own optimum. The raw device-noise energy error (tens of
milli-Hartree) is an order of magnitude above chemical accuracy
(1.6 mHa) and far larger than the HF-to-optimum energy difference
(~0.23 mHa): on current hardware, the optimizer-space questions studied
elsewhere in this codebase (theta vs qg vs pole-damped) are dominated by
hardware noise unless error mitigation is added.

Usage:
    python examples/nisq_hardware_validation.py
    python examples/nisq_hardware_validation.py --backend fake_sherbrooke
    python examples/nisq_hardware_validation.py --mode ibm --backend ibm_brisbane

Requires: pip install ".[hardware]"   (qiskit, qiskit-aer, qiskit-ibm-runtime)
"""

import argparse
import math
import os
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import quantum_volume
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qang.multiqubit import joint_qg_s, joint_qg_s_from_counts, mean_qg_z, mean_qg_z_from_counts  # noqa: E402
from lih_vqe_ry_rx_ansatz import LIH_ELECTRONIC, NUCLEAR_REPULSION, lih_ansatz, lih_energy  # noqa: E402

N_QUBITS = 4
QV_SEED = 0
DELAYS_US = [0.0, 25.0, 50.0, 100.0, 200.0]
CHEMICAL_ACCURACY = 1.6e-3

# The two LiH operating points compared in Experiment 2. The optimum was
# found by direct classical minimization of lih_energy (it reproduces
# lih_vqe_ry_rx_ansatz.ANSATZ_OPTIMUM to < 1e-9 Ha).
LIH_POINTS = {
    "hartree_fock": [math.pi, math.pi, 0.0, 0.0],
    "ansatz_optimum": [3.12879513, math.pi, -0.03727566, 0.0],
}


# --------------------------------------------------------------------- #
# backend selection
# --------------------------------------------------------------------- #
def get_backend(mode: str = "fake", name: str = "fake_brisbane"):
    """Return a backend. mode="fake": a calibration-based fake backend
    from qiskit_ibm_runtime.fake_provider (e.g. "fake_brisbane").
    mode="ibm": a real device via a saved QiskitRuntimeService account
    (name=None picks the least busy operational device)."""
    if mode == "fake":
        from qiskit_ibm_runtime import fake_provider

        class_name = "".join(part.capitalize() for part in name.split("_"))
        cls = getattr(fake_provider, class_name, None)
        if cls is None:
            raise ValueError(f"Unknown fake backend {name!r} (looked for fake_provider.{class_name}).")
        return cls()
    if mode == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        if name is None or name.startswith("fake_"):
            return service.least_busy(operational=True, simulator=False, min_num_qubits=N_QUBITS)
        return service.backend(name)
    raise ValueError(f"mode must be 'fake' or 'ibm', got {mode!r}.")


def _is_fake(backend) -> bool:
    return type(backend).__module__.startswith("qiskit_ibm_runtime.fake_provider")


def choose_layout(backend, n_qubits: int = N_QUBITS):
    """A connected line of n_qubits physical qubits with physically valid
    calibration data (T2 <= 2*T1) whose worst T1 is as large as possible
    (ties broken by the summed two-qubit-gate and readout errors), so the
    idle-delay sweep probes relaxation on the device's best qubits.
    Returns a list of physical qubit indices, or None if none exists."""
    edges = {tuple(sorted(e)) for e in backend.coupling_map.get_edges()}
    neighbours = {}
    for a, b in edges:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)

    target = backend.target

    def valid(q):
        props = backend.qubit_properties(q)
        return props is not None and props.t1 and props.t2 and props.t2 <= 2 * props.t1

    def readout_err(q):
        try:
            return target["measure"][(q,)].error or 0.0
        except (KeyError, TypeError):
            return 0.0

    twoq_names = [g for g in ("ecr", "cz", "cx") if g in target.operation_names]

    def edge_err(a, b):
        for g in twoq_names:
            for pair in ((a, b), (b, a)):
                props = target[g].get(pair)
                if props is not None and props.error is not None:
                    return props.error
        return 1.0

    best, best_cost = None, (float("inf"), float("inf"))

    def extend(path):
        nonlocal best, best_cost
        if len(path) == n_qubits:
            err = sum(edge_err(a, b) for a, b in zip(path, path[1:])) + sum(readout_err(q) for q in path)
            cost = (-min(backend.qubit_properties(q).t1 for q in path), err)
            if cost < best_cost:
                best, best_cost = list(path), cost
            return
        for nxt in sorted(neighbours.get(path[-1], ())):
            if nxt not in path and valid(nxt):
                extend(path + [nxt])

    for start in sorted(neighbours):
        if valid(start):
            extend([start])
    return best


def _run_counts(backend, circuits, shots: int, seed: int):
    """Counts for already-transpiled circuits. Fake backends run locally
    on AerSimulator.from_backend (the device's calibrated noise model,
    seeded, reproducible). Real backends go through qiskit-ibm-runtime's
    SamplerV2. (SamplerV2's own local mode is not used for fake backends
    because it aborts on qubits whose published T2 exceeds 2*T1, which
    several fake devices contain even outside the chosen layout.)"""
    if _is_fake(backend):
        from qiskit_aer import AerSimulator

        sim = AerSimulator.from_backend(backend)
        result = sim.run(circuits, shots=shots, seed_simulator=seed).result()
        return [result.get_counts(i) for i in range(len(circuits))]

    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    job = sampler.run(circuits, shots=shots)
    print(f"  submitted job {job.job_id()} to {backend.name}; waiting ...")
    result = job.result()
    return [pub.data.meas.get_counts() for pub in result]


def _counts_to_index(counts: dict) -> dict:
    out = {}
    for key, c in counts.items():
        idx = int(key.replace(" ", ""), 2)
        out[idx] = out.get(idx, 0) + int(c)
    return out


# --------------------------------------------------------------------- #
# Experiment 1: idle-delay (T1) sweep
# --------------------------------------------------------------------- #
def _qv_circuit():
    return quantum_volume(N_QUBITS, depth=N_QUBITS, seed=QV_SEED).decompose()


def run_relaxation_experiment(backend, shots: int = 4000, seed: int = 42, delays_us=None, layout=None):
    """Rows of {delay_us, qg_s, mean_qg_z, hop, xeb}. The first row is the
    exact, noiseless reference (delay_us=None); the rest are measured on
    the backend after an idle delay of each length in delays_us."""
    delays_us = DELAYS_US if delays_us is None else list(delays_us)
    layout = choose_layout(backend, N_QUBITS) if layout is None else layout

    qv = _qv_circuit()
    sv = Statevector.from_instruction(qv)
    p_ideal = np.abs(sv.data) ** 2
    dim = len(p_ideal)
    heavy = p_ideal > np.median(p_ideal)

    rows = [{
        "delay_us": None,
        "qg_s": joint_qg_s(sv.data, N_QUBITS, normalize=True),
        "mean_qg_z": mean_qg_z(sv.data, N_QUBITS),
        "hop": float(p_ideal[heavy].sum()),
        "xeb": float(dim * np.sum(p_ideal ** 2) - 1.0),
    }]

    circuits = []
    for d in delays_us:
        qc = QuantumCircuit(N_QUBITS)
        qc.compose(qv, inplace=True)
        qc.barrier()
        if d > 0:
            for q in range(N_QUBITS):
                qc.delay(d, q, unit="us")
        qc.measure_all()
        circuits.append(qc)

    isa = transpile(
        circuits,
        backend=backend,
        initial_layout=layout,
        optimization_level=1,
        seed_transpiler=seed,
        scheduling_method="alap",
    )
    all_counts = _run_counts(backend, isa, shots, seed)

    for d, counts in zip(delays_us, all_counts):
        idx_counts = _counts_to_index(counts)
        total = sum(idx_counts.values())
        hop = sum(c for i, c in idx_counts.items() if heavy[i]) / total
        xeb = dim * sum(c * p_ideal[i] for i, c in idx_counts.items()) / total - 1.0
        rows.append({
            "delay_us": d,
            "qg_s": joint_qg_s_from_counts(idx_counts, N_QUBITS, normalize=True),
            "mean_qg_z": mean_qg_z_from_counts(idx_counts, N_QUBITS),
            "hop": float(hop),
            "xeb": float(xeb),
        })
    return rows


# --------------------------------------------------------------------- #
# Experiment 2: raw LiH energy on the device
# --------------------------------------------------------------------- #
def _measurement_groups():
    """Qubit-wise commuting groups of the LiH Hamiltonian: each group is
    measured with one circuit (single-qubit basis changes, then Z)."""
    return LIH_ELECTRONIC.group_commuting(qubit_wise=True)


def _basis_for_group(group):
    """Per-qubit measurement basis ('X', 'Y', 'Z' or 'I') of a qubit-wise
    commuting group. Pauli labels are little-endian (qubit 0 rightmost)."""
    basis = ["I"] * N_QUBITS
    for label in group.paulis.to_labels():
        for q, ch in enumerate(reversed(label)):
            if ch != "I":
                basis[q] = ch
    return basis


def _group_circuit(thetas, basis):
    qc = lih_ansatz(thetas)
    for q, b in enumerate(basis):
        if b == "X":
            qc.h(q)
        elif b == "Y":
            qc.sdg(q)
            qc.h(q)
    qc.measure_all()
    return qc


def _group_expectation(group, counts: dict) -> float:
    idx_counts = _counts_to_index(counts)
    total = sum(idx_counts.values())
    value = 0.0
    for label, coeff in zip(group.paulis.to_labels(), group.coeffs):
        support = [q for q, ch in enumerate(reversed(label)) if ch != "I"]
        s = 0
        for i, c in idx_counts.items():
            parity = sum((i >> q) & 1 for q in support) % 2
            s += c * (-1 if parity else 1)
        value += float(np.real(coeff)) * s / total
    return value


def run_lih_energy_experiment(backend, shots: int = 8000, seed: int = 42, layout=None):
    """Rows of {point, ideal, measured, error} for the Hartree-Fock point
    and the ansatz optimum. No error mitigation of any kind."""
    layout = choose_layout(backend, N_QUBITS) if layout is None else layout
    groups = _measurement_groups()

    circuits, index = [], []
    for point, thetas in LIH_POINTS.items():
        for g in groups:
            circuits.append(_group_circuit(thetas, _basis_for_group(g)))
            index.append((point, g))

    isa = transpile(circuits, backend=backend, initial_layout=layout, optimization_level=1, seed_transpiler=seed)
    all_counts = _run_counts(backend, isa, shots, seed)

    measured = {p: NUCLEAR_REPULSION for p in LIH_POINTS}
    for (point, g), counts in zip(index, all_counts):
        measured[point] += _group_expectation(g, counts)

    rows = []
    for point, thetas in LIH_POINTS.items():
        ideal = lih_energy(thetas)
        rows.append({
            "point": point,
            "ideal": ideal,
            "measured": measured[point],
            "error": abs(measured[point] - ideal),
        })
    return rows


# --------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=["fake", "ibm"], default="fake")
    parser.add_argument("--backend", default=None, help="fake_brisbane (default for --mode fake) or an IBM device name")
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    name = args.backend or ("fake_brisbane" if args.mode == "fake" else None)
    backend = get_backend(args.mode, name)
    layout = choose_layout(backend, N_QUBITS)
    print(f"Backend: {backend.name}   layout (physical qubits): {layout}")
    print()

    print("Experiment 1: idle delay before measurement (T1 relaxation).")
    print(f"  {'delay':>9}  {'qg_S':>7}  {'mean_qg_Z':>9}  {'HOP':>6}  {'XEB':>7}")
    for r in run_relaxation_experiment(backend, shots=args.shots, seed=args.seed, layout=layout):
        label = "noiseless" if r["delay_us"] is None else f"{r['delay_us']:.0f} us"
        print(f"  {label:>9}  {r['qg_s']:7.4f}  {r['mean_qg_z']:+9.4f}  {r['hop']:6.4f}  {r['xeb']:+7.4f}")
    print()

    print("Experiment 2: raw LiH energy (no error mitigation).")
    for r in run_lih_energy_experiment(backend, shots=2 * args.shots, seed=args.seed, layout=layout):
        print(
            f"  {r['point']:>15}: ideal={r['ideal']:+.6f}  measured={r['measured']:+.6f}  "
            f"error={1e3 * r['error']:.1f} mHa  (= {r['error'] / CHEMICAL_ACCURACY:.0f}x chemical accuracy)"
        )


if __name__ == "__main__":
    main()
