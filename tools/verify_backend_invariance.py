"""Backend invariance of qg_Z: one variational circuit on Qiskit, PennyLane and Cirq.

Circuit: RY(a)q0 RY(b)q1 CX(0,1) RY(c)q0.  Statuses: exact (state vectors agree to 1e-12);
sampled counts agree within 4 standard errors (qang.interop.qg_z_profile). qg_Z, not qga/m-qg.
"""
from __future__ import annotations
import sys
import numpy as np
from qang.interop import canonical_counts, qg_z_profile

A, B, C, SHOTS = 0.7, 1.9, 0.4, 20000


def qiskit_run():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit.primitives import StatevectorSampler
    qc = QuantumCircuit(2)
    qc.ry(A, 0); qc.ry(B, 1); qc.cx(0, 1); qc.ry(C, 0)
    sv = Statevector(qc)
    mc = qc.copy(); mc.measure_all()
    res = StatevectorSampler(seed=1).run([mc], shots=SHOTS).result()[0].data.meas.get_counts()
    return np.asarray(sv.data), canonical_counts(res, sdk="qiskit")


def pennylane_run():
    import pennylane as qml
    dev = qml.device("default.qubit", wires=2, shots=SHOTS, seed=2)
    def body():
        qml.RY(A, 0); qml.RY(B, 1); qml.CNOT([0, 1]); qml.RY(C, 0)
    sv = qml.QNode(lambda: (body(), qml.state())[1], qml.device("default.qubit", wires=2))()
    counts = qml.QNode(lambda: (body(), qml.counts(wires=[0, 1]))[1], dev)()
    return np.asarray(sv), canonical_counts(dict(counts), sdk="pennylane")


def cirq_run():
    import cirq
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.ry(A)(q[0]), cirq.ry(B)(q[1]), cirq.CNOT(q[0], q[1]), cirq.ry(C)(q[0]))
    sv = cirq.Simulator(dtype=np.complex128).simulate(c).final_state_vector
    m = c + cirq.measure(*q, key="m")
    r = cirq.Simulator(seed=3).run(m, repetitions=SHOTS)
    return np.asarray(sv), canonical_counts(r)


def qgz_exact(sv):
    p = np.abs(sv) ** 2
    return np.array([1 - 2 * (p[2] + p[3]), 1 - 2 * (p[1] + p[3])])  # qubit 0 = MSB


def main() -> int:
    ref = None
    ok = True
    for name, fn in (("qiskit*", qiskit_run), ("pennylane", pennylane_run), ("cirq", cirq_run)):
        sv, cnt = fn()
        if name == "qiskit*":   # qiskit statevector is little-endian: reorder to qubit0=MSB
            sv = sv.reshape(2, 2).T.reshape(-1)
            name = "qiskit"
        ex = qgz_exact(sv)
        ref = ex if ref is None else ref
        prof = qg_z_profile(cnt)
        d_exact = float(np.max(np.abs(ex - ref)))
        z = max(abs(p - e) / max(s, 1e-9) for (p, s), e in zip(prof, ex))
        print(f"{name:10s} qg_Z exact={np.round(ex,6)}  max|Δ vs ref|={d_exact:.2e}  sampled max z={z:.2f}")
        ok &= d_exact < 1e-12 and z < 4.0
    print("INVARIANCE", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
