"""
The same qg read-out from Qiskit, Cirq and PennyLane (§130)

One 3-qubit W-like state prepared with each SDK's own qang gates, measured,
and read with qang.interop: the per-qubit qg_Z, the qg filter (weight 1) and
the register entropy come out the same whatever SDK produced the counts,
because every result is first put in one qubit order (qubit 0 first). SDKs
that are not installed are skipped.

python examples/interop_sdks_qg.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qang import interop as IO  # noqa: E402

SHOTS = 20000
A = 1 / math.sqrt(3)


def qiskit_counts():
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    # W state |100> + |010> + |001> (qubit 0 first)
    qc = QuantumCircuit(3)
    qc.ry(2 * math.asin(A), 0)
    qc.ch(0, 1, ctrl_state=0)
    qc.ccx(0, 1, 2, ctrl_state=0)
    qc.measure_all()
    sim = AerSimulator(seed_simulator=130)
    return IO.canonical_counts(sim.run(transpile(qc, sim), shots=SHOTS).result().get_counts(), sdk="qiskit")


def cirq_counts():
    import cirq

    q = cirq.LineQubit.range(3)
    c = cirq.Circuit(
        cirq.ry(2 * math.asin(A)).on(q[0]),
        cirq.H.on(q[1]).controlled_by(q[0], control_values=[0]),
        cirq.X.on(q[2]).controlled_by(q[0], q[1], control_values=[0, 0]),
        cirq.measure(*q, key="m"),
    )
    return IO.canonical_counts(cirq.Simulator(seed=130).run(c, repetitions=SHOTS))


def pennylane_counts():
    import pennylane as qml

    dev = qml.device("default.qubit", wires=3, seed=130)

    @qml.qnode(dev)
    def w():
        qml.RY(2 * math.asin(A), wires=0)
        qml.ctrl(qml.Hadamard, control=0, control_values=[0])(wires=1)
        qml.ctrl(qml.PauliX, control=[0, 1], control_values=[0, 0])(wires=2)
        return qml.counts()

    return IO.canonical_counts(qml.set_shots(w, shots=SHOTS)(), sdk="pennylane")


def main():
    rows = {}
    for name, f in (("Qiskit", qiskit_counts), ("Cirq", cirq_counts), ("PennyLane", pennylane_counts)):
        try:
            rows[name] = f()
        except ImportError as e:
            print(f"{name}: skipped ({e.name} not installed)")
    print("W state: ideal qg_Z = 1/3 on every qubit, all shots in weight 1, register entropy log2(3)/3 = 0.528")
    for name, c in rows.items():
        prof = " ".join(f"{q:+.3f}±{se:.3f}" for q, se in IO.qg_z_profile(c))
        _, kept = IO.weight_filter(c, 1)
        print(f"  {name:9s} qg_Z {prof}  kept {kept:.3f}  qg_S joint {IO.joint_qg_s(c):.3f}")
    return rows


if __name__ == "__main__":
    main()
