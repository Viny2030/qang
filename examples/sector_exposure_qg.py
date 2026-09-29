"""
A noise-free predictor of how much T1 error the qg filter lets through.

§60, §61 and §66 found that the share of the amplitude-damping (T1) error the
Hamming-weight filter removes depends on how the circuit is compiled: 99.5 %
with number-conserving gates, about 50 % with 3-CNOT bonds. The mechanism: a
decay acting while part of the state is outside the conserved sector can be
rotated back into it by later gates and pass the filter.

qang.sectors.sector_exposure measures, on a NOISELESS statevector, the
population outside the sector right after each two-qubit gate (where the
noise model applies the decay). If the mechanism is right, this number,
computed before any noisy run, should predict the filter's T1 leak
(1 - share of the T1 error removed).

Test set: the §60 XXZ chain (n = 6, 4 Trotter steps) in four compilations
(3 CNOT, number-conserving XY + ZZ, MS rotations, MS with ZZ by basis change)
at dt = 0.1, 0.25, 0.5, 0.75, and the §66 Z2 gauge theory (Gauss + number
filter is not used here: the number filter only, N conserved) at 2, 4, 6
steps. T1 leak from exact density matrices with gamma = 0.01 per two-qubit
gate.

Prediction, written before the run: across the 16 XXZ points the Spearman
rank correlation between mean exposure and T1 leak is at least 0.8, and
every compilation with zero exposure removes at least 95 % of the T1 error.

Findings (python examples/sector_exposure_qg.py):

  XXZ chain (4 steps, gamma = 0.01): mean exposure and share of the T1 error
  the filter lets through
    compilation                 exposure (dt 0.1 / 0.25 / 0.5 / 0.75)   T1 leak
    3 CNOT                      0.42 at every dt                       16 / 52 / 39 / 46 %
    MS, ZZ by basis change      0.16 / 0.17 / 0.22 / 0.24              78 / 30 / 31 / 20 %
    MS rotations                0.0006 / 0.005 / 0.026 / 0.047         0.4 / 3.3 / 9.9 / 17 %
    number-conserving           0                                      2.0 / 0.5 / 0.1 / 0.0 %
  Spearman rank correlation over the 16 points: 0.80.
  Z2 gauge theory (Pauli-rotation hopping, number filter): exposure 0.64-0.65,
  leak 79-483 % (the filter makes the T1 error worse at 2 and 4 steps).

  * The prediction holds, at its threshold: rank correlation 0.80 (>= 0.8)
    and every zero-exposure circuit removes at least 98 % of the T1 error.
  * The exposure ranks COMPILATIONS, not time steps. Within the MS-rotation
    compilation it tracks the leak well (both grow with dt, as the rotation
    angle grows); within 3 CNOT it is constant (0.42) while the leak moves
    between 16 and 52 %, because at small dt the raw error itself is small
    and the leak ratio is noisy. It is a coarse, noise-free screen:
    exposure ~0 -> the filter will catch T1; exposure >~ 0.2 -> expect half
    or more of the T1 error to pass.
  * The gauge circuit sits at the top of the scale (0.65) and is the case
    where post-selection made T1 worse (§66): consistent with the screen.

  What is new. That basis changes around two-qubit gates spoil symmetry
  verification against amplitude damping follows from §60; the contribution
  here is a noise-free number, computable at compile time from a statevector,
  that ranks compilations by how much T1 error the qg filter will let
  through. Limitations: 6-7 qubits, one noise type, a rank (not a
  quantitative) prediction, and the threshold 0.8 is met with no margin.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from qang.sectors import sector_exposure  # noqa: E402

COMPS = (False, True, "ms", "ionq")
NAMES = {False: "3 CNOT", True: "number-conserving", "ms": "MS rotations", "ionq": "MS, ZZ by basis change"}


def xxz_points(dts=(0.1, 0.25, 0.5, 0.75), steps=4, n=6, gamma=0.01):
    from qiskit import transpile

    import xxz_trotter_filter_qg as X

    rows = []
    for dt in dts:
        ideal, _ = X.imbalance(X.probabilities(n, steps, dt), n)
        for c in COMPS:
            qc = X.trotter_circuit(n, steps, dt, measure=False, native=c)
            if not c:
                qc = transpile(qc, basis_gates=["cx", "rz", "ry", "x"], optimization_level=0)
            exp = sector_exposure(qc, n // 2)["mean"]
            p = X.probabilities(n, steps, dt, gamma=gamma, native=c)
            raw, _ = X.imbalance(p, n)
            f, _ = X.imbalance(p, n, True)
            leak = abs(f - ideal) / abs(raw - ideal)
            rows.append({"model": "XXZ", "comp": NAMES[c], "dt": dt, "exposure": exp, "leak": leak})
    return rows


def gauge_points(steps_list=(2, 4, 6), gamma=0.01):
    from qiskit import transpile

    import lattice_gauge_gauss_qg as G

    rows = []
    for s in steps_list:
        qc = transpile(G.trotter_circuit(s), basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=1)
        # exposure w.r.t. the matter number: population with matter N != N0 after each CX
        from qiskit.quantum_info import Statevector
        from qiskit import QuantumCircuit

        sv = Statevector.from_label("0" * G.N_Q)
        bad = G.NUM != G.N0
        per = []
        for inst in qc.data:
            sub = QuantumCircuit(G.N_Q)
            sub.append(inst.operation, [qc.find_bit(q).index for q in inst.qubits])
            sv = sv.evolve(sub)
            if len(inst.qubits) >= 2:
                per.append(float(np.sum(np.abs(sv.data[bad]) ** 2)))
        ideal = float(G.probabilities(s) @ G.NU)
        est = G.estimates(G.probabilities(s, gamma=gamma))
        leak = abs(est["N"] - ideal) / abs(est["raw"] - ideal)
        rows.append({"model": "Z2 gauge", "comp": "Pauli rotations", "dt": s, "exposure": float(np.mean(per)), "leak": leak})
    return rows


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    xr = xxz_points()
    print("XXZ chain, 4 steps, gamma = 0.01:")
    print(f"  {'compilation':24s} {'dt':>5} | {'exposure':>8} | {'T1 leak':>7}")
    for r in xr:
        print(f"  {r['comp']:24s} {r['dt']:5.2f} | {r['exposure']:8.4f} | {r['leak']:7.1%}")
    rho = spearman([r["exposure"] for r in xr], [r["leak"] for r in xr])
    print(f"  Spearman rank correlation (16 points): {rho:.3f}")
    zero = [r for r in xr if r["exposure"] < 1e-9]
    print(f"  zero-exposure points: {len(zero)}, worst share removed {1 - max(r['leak'] for r in zero):.1%}")
    gr = gauge_points()
    print("\nZ2 gauge theory (number filter), gamma = 0.01:")
    for r in gr:
        print(f"  steps {r['dt']}: exposure {r['exposure']:.4f} | T1 leak {r['leak']:.1%}")
    return xr, gr, rho


if __name__ == "__main__":
    main()
