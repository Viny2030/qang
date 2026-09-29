"""
Gauss-law witnesses and filters for a lattice gauge theory in qg.

Model: the 1D Z2 lattice gauge theory coupled to staggered fermions, a
standard test bed for quantum simulation of gauge theories. Matter qubits on
sites j = 0..L-1, gauge qubits on the links between them (open chain, L = 4:
7 qubits, order m0 l01 m1 l12 m2 l23 m3). The electric field is read in Z:

  H = -J sum_j (X_j Xl_j X_{j+1} + Y_j Xl_j Y_{j+1})/2      (hopping, flips the link)
      + m sum_j (-1)^j Z_j / 2                              (staggered mass)
      + h sum_links Zl / 2                                   (electric energy)

The Gauss law is local and diagonal in Z: G_j = Zl_{left} Zl_{right} Z_j
(missing links at the ends) commutes with H; physical states have every
G_j = g_j fixed by the initial state. H also conserves the fermion number
N. Start: the bare vacuum (odd sites filled), all links in the same state;
observable: the particle density nu(t) = (1/L) sum_j (1 + (-1)^j Z_j)/2 ...
measured, like everything here, in Z.

Every G_j is the qg of a weight-2 or weight-3 parity, so the register gives,
from the same Z shots:
  * L local Gauss witnesses <G_j> (ideal value g_j) and the global number
    witness (register-mean qg of the matter qubits);
  * three filters: keep shots with N conserved; with every G_j = g_j; with both.

Trotter steps with the hopping terms compiled by Qiskit to CNOTs; noise on
every CNOT: two-qubit depolarizing p2 and amplitude damping gamma; readout
error e. Exact density matrices (qiskit-aer).

Prediction, written before the run:
  P1 the Gauss filter removes more of the error than the number filter,
     because errors on the gauge qubits change G_j but not N;
  P2 Gauss + N removes the most, and the local witnesses <G_j> locate
     where the errors hit (they differ by site);
  P3 as in §60, decays inside a CNOT-compiled hopping gate can leave the
     sector and re-enter it, so under T1 the filters remove clearly less
     than 100 % of the error.

Findings (python examples/lattice_gauge_gauss_qg.py):

  Checks: [H, G_j] = 0 and [H, N] = 0 exactly. Each hopping term is two
  commuting 3-qubit Pauli rotations (4 CNOTs each, fidelity 1 - 1e-15 with
  the exact exponential): 24 CNOTs per Trotter step. (An earlier version let
  Qiskit synthesize the 3-qubit gate: 54 CNOTs per step, and the ranking of
  the filters under T1 changed between two machines with different numerical
  libraries -- the synthesized basis changes decide how much of the decay
  leaks through. The explicit decomposition is deterministic.)

  A. Summed density error over 1-6 Trotter steps, share removed:
       noise                            N      Gauss   Gauss + N
       depolarizing p2 = 0.01           36 %   50 %    55 %
       amplitude damping gamma = 0.01   19 %   25 %    27 %
       mixed (0.005, 0.002, e = 0.01)   39 %   49 %    53 %
       readout e = 0.02 (2 or 6 steps)  96 %   100 %   100 %
  A'. Single steps are noisier: at 2 steps the depolarizing error is removed
     88 % by N and 81 % by Gauss (92 % both); under T1 at 2 steps the raw
     error is small by accident (0.002) and every filter makes it worse
     (0.008-0.010): post-selection keeps the no-jump part of the decays
     inside the gates, which biases the surviving shots. At 6 steps the
     ranking is Gauss > N in every noise (46 vs 31 %, 27 vs 21 %).
  B. Mixed noise per step (kept fraction with both filters 0.85 -> 0.53
     from 1 to 6 steps): error 0.027 -> 0.006 at 1 step, 0.114 -> 0.056 at
     6 steps.

  * P1 holds over the summed error (Gauss 50 vs 36 %, 25 vs 19 %, 49 vs
    39 %) and at 6 steps, but not at every single step: at 2 steps under
    depolarizing noise the number filter is better.
  * P2 holds: Gauss + N is the best over the summed error in every noise.
  * P3 holds, and more strongly than predicted: under T1 the filters remove
    only 19-27 % over the run and can make an accidentally small error
    worse.
  * The local witnesses do not locate errors in this homogeneous model:
    |<G_j>| is lower in the bulk (weight-3 checks) than at the ends
    (weight 2), a weight effect.
  * Limits: the gain shrinks with depth (both filters: 78 % of the error
    at 1 step, 51 % at 6 steps under mixed noise) and costs shots (kept
    0.53 at 6 steps).

  What is new and what is not. Gauss-law post-selection and gauge-violation
  penalties are known in lattice-gauge quantum simulation (e.g. Stryker 2019;
  Nguyen et al. 2022; Halimeh and Hauke 2020). Ours: the Gauss checks read as
  qg of parities from the same Z shots as the observable, the comparison
  with the global number filter per noise type, and the compilation leak.
  Limitations: L = 4 sites, Z2 gauge group, open chain, one observable and
  one initial state, gate-attached noise models, no hardware.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

L = 4
N_Q = 2 * L - 1
MATTER = [2 * j for j in range(L)]
LINKS = [2 * j + 1 for j in range(L - 1)]
J_HOP, MASS, H_E = 1.0, 0.5, 0.5


def _pauli_op(pauli_on):
    """Dense operator from {qubit: 'X'|'Y'|'Z'} (bit i of the index = qubit i)."""
    mats = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.diag([1.0, -1.0])}
    op = np.array([[1.0]])
    for q in range(N_Q - 1, -1, -1):
        op = np.kron(op, mats[pauli_on.get(q, "I")])
    return op


def hopping_unitary(dt):
    """exp(-i dt (-J)(X Xl X + Y Xl Y)/2) on (m_j, l, m_{j+1}) as a 3-qubit matrix
    (qubit order within the gate: m_j, l, m_{j+1})."""
    import scipy.linalg as sl

    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    k = lambda a, b, c: np.kron(c, np.kron(b, a))  # noqa: E731  (little endian)
    Hh = -J_HOP * (k(X, X, X) + k(Y, X, Y)) / 2
    return sl.expm(-1j * dt * Hh)


def _pauli_rotation(qc, qubits, paulis, phi):
    """exp(-i phi/2 P) for P = tensor of paulis on qubits: basis change to Z,
    CNOT ladder, Rz, undo. Deterministic (no unitary synthesis)."""
    for q, pa in zip(qubits, paulis):
        if pa == "X":
            qc.h(q)
        elif pa == "Y":
            qc.sdg(q)
            qc.h(q)
    for a, b in zip(qubits[:-1], qubits[1:]):
        qc.cx(a, b)
    qc.rz(phi, qubits[-1])
    for a, b in reversed(list(zip(qubits[:-1], qubits[1:]))):
        qc.cx(a, b)
    for q, pa in zip(qubits, paulis):
        if pa == "X":
            qc.h(q)
        elif pa == "Y":
            qc.h(q)
            qc.s(q)


def hop(qc, a, l, b, dt):
    """exp(-i dt (-J)(X Xl X + Y Xl Y)/2) = R_XXX(-J dt) R_YXY(-J dt): the two
    Pauli strings commute, so the product is exact. 4 CNOTs each."""
    _pauli_rotation(qc, [a, l, b], "XXX", -J_HOP * dt)
    _pauli_rotation(qc, [a, l, b], "YXY", -J_HOP * dt)


def trotter_circuit(steps, dt=0.3, measure=False):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(N_Q)
    for j in range(L):
        if j % 2 == 1:
            qc.x(MATTER[j])  # bare vacuum: odd sites filled
    for _ in range(steps):
        for start in (0, 1):
            for j in range(start, L - 1, 2):
                hop(qc, MATTER[j], LINKS[j], MATTER[j + 1], dt)
        for j in range(L):
            qc.rz(2 * dt * MASS * (-1) ** j / 2, MATTER[j])
        for l in LINKS:
            qc.rz(2 * dt * H_E / 2, l)
    if measure:
        qc.measure_all()
    return qc


def noise_model(p2, gamma):
    from qiskit_aer.noise import NoiseModel, amplitude_damping_error, depolarizing_error

    nm = NoiseModel()
    err = depolarizing_error(p2, 2)
    if gamma > 0:
        ad = amplitude_damping_error(gamma)
        err = err.compose(ad.tensor(ad))
    if p2 > 0 or gamma > 0:
        nm.add_all_qubit_quantum_error(err, ["cx"])
    return nm


_T = {}


def _transpiled(steps, dt):
    from qiskit import transpile

    key = (steps, dt)
    if key not in _T:
        _T[key] = transpile(trotter_circuit(steps, dt), basis_gates=["cx", "rz", "sx", "x"], optimization_level=1,
                            seed_transpiler=1)
    return _T[key]


def probabilities(steps, dt=0.3, p2=0.0, gamma=0.0, e=0.0):
    from qiskit_aer import AerSimulator

    t = _transpiled(steps, dt).copy()
    t.save_probabilities()
    method = "statevector" if (p2 == 0 and gamma == 0) else "density_matrix"
    p = np.asarray(AerSimulator(method=method, noise_model=noise_model(p2, gamma)).run(t).result().data()["probabilities"])
    if e > 0:
        p = p.reshape([2] * N_Q)
        m = np.array([[1 - e, e], [e, 1 - e]])
        for ax in range(N_Q):
            p = np.moveaxis(np.tensordot(m, p, axes=([1], [ax])), 0, ax)
        p = p.reshape(-1)
    return p


def cnot_count(steps, dt=0.3):
    return _transpiled(steps, dt).count_ops().get("cx", 0)


_IDX = np.arange(2**N_Q)
_Z = 1 - 2 * ((_IDX[:, None] >> np.arange(N_Q)) & 1)


def gauss_values():
    """G_j per outcome (array [outcomes, L])."""
    G = np.ones((2**N_Q, L))
    for j in range(L):
        G[:, j] = _Z[:, MATTER[j]]
        if j > 0:
            G[:, j] *= _Z[:, LINKS[j - 1]]
        if j < L - 1:
            G[:, j] *= _Z[:, LINKS[j]]
    return G


def density():
    """Particle density per outcome: 1/L sum_j (1 + (-1)^j Z_j)/2 ... with odd sites
    filled in the vacuum, a particle on an even site is Z = -1 there and a hole
    (antiparticle) on an odd site is Z = +1."""
    nu = np.zeros(2**N_Q)
    for j in range(L):
        z = _Z[:, MATTER[j]]
        nu += (1 - z) / 2 if j % 2 == 0 else (1 + z) / 2
    return nu / L


def number():
    return ((1 - _Z[:, MATTER]) / 2).sum(axis=1)


G_OUT = gauss_values()
NU = density()
NUM = number()
G0 = G_OUT[int(sum(1 << MATTER[j] for j in range(L) if j % 2 == 1))]  # Gauss values of the initial state
N0 = L // 2


def estimates(p):
    ok_g = np.all(G_OUT == G0, axis=1)
    ok_n = NUM == N0
    out = {"raw": float(p @ NU)}
    for name, mask in (("N", ok_n), ("Gauss", ok_g), ("Gauss+N", ok_g & ok_n)):
        k = p[mask].sum()
        out[name] = float((p[mask] @ NU[mask]) / k)
        out["kept_" + name] = float(k)
    out["witness_G"] = [float(p @ G_OUT[:, j]) for j in range(L)]
    out["witness_N"] = float(p @ (1 - 2 * NUM / L))  # register-mean qg of the matter qubits
    return out


def study(p2, gamma, e, steps_list=(1, 2, 4, 6), dt=0.3):
    rows = []
    for s in steps_list:
        ideal = float(probabilities(s, dt) @ NU)
        est = estimates(probabilities(s, dt, p2, gamma, e))
        rows.append({"steps": s, "cx": cnot_count(s, dt), "ideal": ideal, **est})
    return rows


def one_noise(steps=2, dt=0.3):
    ideal = float(probabilities(steps, dt) @ NU)
    out = []
    for name, kw in (("depolarizing p2 = 0.01", dict(p2=0.01)), ("amplitude damping gamma = 0.01", dict(gamma=0.01)),
                     ("readout e = 0.02", dict(e=0.02))):
        est = estimates(probabilities(steps, dt, **kw))
        err = {k: abs(est[k] - ideal) for k in ("raw", "N", "Gauss", "Gauss+N")}
        out.append((name, err, est))
    return ideal, out


def aggregate(steps=range(1, 7), dt=0.3, **noise):
    """Sum over Trotter steps of |estimate - ideal| for each estimator: a
    ranking that does not depend on one step where the raw error happens to
    be small by accident."""
    tot = {k: 0.0 for k in ("raw", "N", "Gauss", "Gauss+N")}
    for s in steps:
        ideal = float(probabilities(s, dt) @ NU)
        est = estimates(probabilities(s, dt, **noise))
        for k in tot:
            tot[k] += abs(est[k] - ideal)
    return tot


def check_symmetries():
    """[H, G_j] = 0 and [H, N] = 0 for the dense Hamiltonian."""
    H = np.zeros((2**N_Q, 2**N_Q), dtype=complex)
    for j in range(L - 1):
        H += -J_HOP / 2 * (_pauli_op({MATTER[j]: "X", LINKS[j]: "X", MATTER[j + 1]: "X"})
                           + _pauli_op({MATTER[j]: "Y", LINKS[j]: "X", MATTER[j + 1]: "Y"}))
    for j in range(L):
        H += MASS * (-1) ** j / 2 * _pauli_op({MATTER[j]: "Z"})
    for l in LINKS:
        H += H_E / 2 * _pauli_op({l: "Z"})
    worst = 0.0
    for j in range(L):
        Gop = np.diag(G_OUT[:, j])
        worst = max(worst, np.abs(H @ Gop - Gop @ H).max())
    Nop = np.diag(NUM.astype(float))
    return worst, float(np.abs(H @ Nop - Nop @ H).max())


def main():
    cg, cn = check_symmetries()
    print(f"checks: max |[H, G_j]| = {cg:.1e}, |[H, N]| = {cn:.1e}; CNOTs per Trotter step {cnot_count(1)}")
    print("\nA. summed density error over 1-6 Trotter steps, and share removed")
    for label, kw in (("depolarizing p2 = 0.01", dict(p2=0.01)), ("amplitude damping gamma = 0.01", dict(gamma=0.01)),
                      ("mixed p2 = 0.005, gamma = 0.002, e = 0.01", dict(p2=0.005, gamma=0.002, e=0.01))):
        t = aggregate(**kw)
        print(f"  {label:42s} raw {t['raw']:.3f} | N {t['N']:.3f} ({1 - t['N'] / t['raw']:.0%})"
              f" | Gauss {t['Gauss']:.3f} ({1 - t['Gauss'] / t['raw']:.0%})"
              f" | Gauss+N {t['Gauss+N']:.3f} ({1 - t['Gauss+N'] / t['raw']:.0%})")
    for st in (2, 6):
      ideal, rows = one_noise(steps=st)
      print(f"\nA'. one noise at a time ({st} steps, ideal density {ideal:.4f}): error and share removed")
      for name, err, est in rows:
        print(f"  {name:32s} raw {err['raw']:.4f} | N {err['N']:.4f} ({1 - err['N'] / err['raw']:.0%})"
              f" | Gauss {err['Gauss']:.4f} ({1 - err['Gauss'] / err['raw']:.0%})"
              f" | Gauss+N {err['Gauss+N']:.4f} ({1 - err['Gauss+N'] / err['raw']:.0%})"
              f" | kept N {est['kept_N']:.2f} Gauss {est['kept_Gauss']:.2f} both {est['kept_Gauss+N']:.2f}")
    for label, kw in (("mixed: p2 = 0.005, gamma = 0.002, e = 0.01", dict(p2=0.005, gamma=0.002, e=0.01)),):
        print(f"\nB. {label}")
        for r in study(**kw):
            print(f"  steps {r['steps']} ({r['cx']} CX): ideal {r['ideal']:.4f} | error raw {abs(r['raw'] - r['ideal']):.4f}"
                  f" N {abs(r['N'] - r['ideal']):.4f} Gauss {abs(r['Gauss'] - r['ideal']):.4f}"
                  f" both {abs(r['Gauss+N'] - r['ideal']):.4f} | kept both {r['kept_Gauss+N']:.2f}"
                  f" | <G_j> {' '.join(f'{w:+.3f}' for w in r['witness_G'])}")


if __name__ == "__main__":
    main()
