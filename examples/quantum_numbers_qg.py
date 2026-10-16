"""
The four quantum numbers of an electron, each read with qang (§155).

python examples/quantum_numbers_qg.py

Every number is prepared as a circuit, sampled, and read with qang's own
read-outs (Qang objects, qg_Z, the qang angle, qg_S, the weight filter):

  m_s  one qubit: qg_Z = 2 m_s, and the qang angle arccos(qg_Z) is the tilt
       of the spin.
  m_l  a 2l-qubit symmetric register: m_l = l - w per shot (w = Hamming
       weight), the register mean qg_Z = m_l / l, and the weight filter is the
       projection onto one m_l. qg_S over m (base 2l + 1) separates p_z (sharp)
       from p_x, p_y (m_l = +-1 mixed), which all have qg_Z = 0.
  l    from a qg value of the SWAP operator: for two spins 1/2,
       <J^2> = 1 + qg_SWAP, so l (l + 1) is read from one qang number
       (qg_SWAP = +1 for l = 1, -1 for l = 0), measured in the Bell basis.
  n    shell n = 2 is exactly the 2-qubit register (2s = singlet, 2p = triplet,
       hydrogen's SO(4) symmetry): its 4 states are told apart by
       (qg_SWAP, register qg_Z). Shell n = 3 (4 qubits, 9 states, exact
       states) shows the register qg_Z = m_l / (n - 1) in every shell.

Part 1 is noiseless sampling (20,000 shots). Part 2 repeats the m_l = 0
(p_z) read-out on the FakeTorino noise model, raw and with the weight filter.
A demonstration, not a pre-registered test.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qang import angular as A  # noqa: E402
from qang import interop as IO  # noqa: E402
from qang.core import Qang  # noqa: E402

SHOTS = 20000


# ---------------------------------------------------------------- circuits

def orbital_circuit(name):
    """The l = 1 orbitals and the l = 0 singlet on 2 qubits (m_l = 1 - weight)."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    if name == "p-1":
        qc.x([0, 1])
    elif name in ("p0", "pz"):  # (|01> + |10>)/sqrt2
        qc.h(0), qc.cx(0, 1), qc.x(1)
    elif name == "px":  # proportional to |00> - |11>
        qc.h(0), qc.z(0), qc.cx(0, 1)
    elif name == "py":  # proportional to |00> + |11>
        qc.h(0), qc.cx(0, 1)
    elif name == "s":  # singlet (|01> - |10>)/sqrt2, l = 0
        qc.h(0), qc.z(0), qc.cx(0, 1), qc.x(1)
    return qc


def with_measure(qc, bell=False):
    """Z-basis measurement, or the Bell-basis rotation (singlet -> |11>) for the SWAP read-out."""
    c = qc.copy()
    if bell:
        c.cx(0, 1)
        c.h(0)
    c.measure_all()
    return c


def sample(circs, backend=None, seed=7):
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator.from_backend(backend, seed_simulator=seed) if backend else AerSimulator(seed_simulator=seed)
    t = transpile(circs, sim, optimization_level=1, seed_transpiler=137)
    res = sim.run(t, shots=SHOTS).result()
    return [IO.canonical_counts(res.get_counts(i), sdk="qiskit") for i in range(len(circs))]


# ---------------------------------------------------------------- qang read-outs

def m_distribution(counts, n):
    pm = np.zeros(n + 1)
    shots = sum(counts.values())
    for k, c in counts.items():
        pm[sum(k)] += c / shots
    return pm


def qg_swap(bell_counts):
    """qg_SWAP = <SWAP> = 1 - 2 P(singlet); in the Bell basis the singlet is the outcome 11."""
    shots = sum(bell_counts.values())
    return 1 - 2 * bell_counts.get((1, 1), 0) / shots


def show_qang(q):
    return f"{q.value:+.3f} qg = {q.milliqang:+7.1f} mqg, angle {q.to_theta('deg'):5.1f} deg"


def part_ms():
    print("m_s: one qubit; qg_Z = 2 m_s and the qang angle is the tilt of the spin")
    from qiskit import QuantumCircuit

    for theta in (0.0, math.pi / 3, math.pi / 2, math.pi):
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        (counts,) = sample([with_measure(qc)])
        z, se = IO.qg_z_profile(counts)[0]
        q = Qang(z)
        print(f"  RY({math.degrees(theta):5.1f} deg): qg_Z {show_qang(q)}  <m_s> = {z / 2:+.3f}  "
              f"qg_S = {q.to_entropic().value:.3f}")


def part_ml_l(names=("p+1", "p0", "p-1", "px", "py", "pz", "s")):
    print("m_l and l: 2-qubit register; m_l = 1 - weight, register qg_Z = <m_l>/l, <J^2> = 1 + qg_SWAP")
    zc = sample([with_measure(orbital_circuit(n)) for n in names])
    bc = sample([with_measure(orbital_circuit(n), bell=True) for n in names])
    for name, cz, cb in zip(names, zc, bc):
        pm = m_distribution(cz, 2)  # index = weight: m_l = +1, 0, -1
        zs = [Qang(z) for z, _ in IO.qg_z_profile(cz)]
        reg = Qang(float(np.mean([q.value for q in zs])))
        sw = qg_swap(cb)
        l_hat = A.l_from_j_squared(A.j_squared_from_swaps(2, [sw]))
        print(f"  {name:4s} P(m_l=+1,0,-1) = {np.round(pm, 3)}  register qg_Z {reg.value:+.3f} -> <m_l> "
              f"{reg.value:+.3f};  qg_S(m) = {A.qg_s_spin(pm):.3f};  qg_SWAP = {sw:+.3f} -> l = {l_hat:.2f}")


def part_n():
    print("n = 2: the four states of the shell are the 2-qubit register; (qg_SWAP, register qg_Z) -> (l, m_l)")
    names = {"2s": "s", "2p+1": "p+1", "2p0": "p0", "2p-1": "p-1"}
    zc = sample([with_measure(orbital_circuit(n)) for n in names.values()])
    bc = sample([with_measure(orbital_circuit(n), bell=True) for n in names.values()])
    for label, cz, cb in zip(names, zc, bc):
        sw = qg_swap(cb)
        reg = float(np.mean([z for z, _ in IO.qg_z_profile(cz)]))
        l_hat = round(A.l_from_j_squared(A.j_squared_from_swaps(2, [sw])))
        print(f"  {label:5s} qg_SWAP {sw:+.3f}, register qg_Z {reg:+.3f}  ->  l = {l_hat}, m_l = {round(reg * 1):+d}")
    print("n = 3: two symmetric 2-qubit registers (4 qubits, 9 states); register qg_Z = m_l / (n - 1) = m_l / 2")
    k = 2
    sym = np.array([A.dicke_state(k, w) for w in range(k + 1)]).T
    P = np.kron(sym, sym)
    Jx, Jy, Jz = A.spin_operators(2 * k)
    J2 = P.conj().T @ (Jx @ Jx + Jy @ Jy + Jz @ Jz) @ P
    Jz_s = P.conj().T @ Jz @ P
    ev, vec = np.linalg.eigh(J2 + 1e-3 * Jz_s)  # split the degenerate m inside each l
    rows = []
    for v in vec.T:
        psi = P @ v
        l = (-1 + math.sqrt(1 + 4 * np.real(np.vdot(psi, (Jx @ Jx + Jy @ Jy + Jz @ Jz) @ psi)))) / 2
        z = np.real(np.vdot(psi, Jz @ psi)) / k  # register-mean qg_Z
        rows.append((round(l), round(z * k), z))
    for l, m, z in sorted(rows, key=lambda r: (r[0], -r[1])):
        print(f"  l = {l}, m_l = {m:+d}: register qg_Z {z:+.3f} ({Qang(z).to_theta('deg'):5.1f} deg)")


def part_noise():
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    from qang import ibm

    b = fp.FakeTorino()
    sim = AerSimulator.from_backend(b, seed_simulator=7)
    qc = with_measure(orbital_circuit("pz"))
    default = transpile(qc, b, optimization_level=1, seed_transpiler=137)
    layout, pred, chosen = ibm.best_chain(qc, b, weight=1)
    print("noise (FakeTorino model): p_z has m_l = 0 exactly; the weight filter IS the m_l = 0 projection")
    for label, t in (("default layout", default), ("ibm.best_chain", chosen)):
        phys = ibm.measured_physical_qubits(t)
        cz = IO.canonical_counts(sim.run(t, shots=SHOTS).result().get_counts(), sdk="qiskit")
        pm = m_distribution(cz, 2)
        kept, frac = IO.weight_filter(cz, 1)
        pmf = m_distribution(kept, 2)
        print(f"  {label} {phys}, readout errors {[round(max(e), 3) for e in ibm.readout_errors(b, phys)]}")
        print(f"    raw      P(m_l=+1,0,-1) = {np.round(pm, 4)}  qg_S(m) = {A.qg_s_spin(pm):.3f}")
        print(f"    filtered P(m_l=+1,0,-1) = {np.round(pmf, 4)}  qg_S(m) = {A.qg_s_spin(pmf):.3f}  kept {frac:.3f}"
              + (f" (predicted {pred['kept']:.3f})" if t is chosen else ""))


def main():
    part_ms()
    part_ml_l()
    part_n()
    part_noise()


if __name__ == "__main__":
    main()
