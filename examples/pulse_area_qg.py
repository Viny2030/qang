"""
The minimal-area rule for fractional gates, with its qg error budget (§160)

On devices with fractional native gates (continuously parameterized RX(theta), RZZ(theta)) the pulse area grows
with |theta|, and so do amplitude-miscalibration error and the relaxation during the pulse. qang.pulse_area
moves every angle to its shortest equivalent: RX has period 2 pi (global phase), RZZ period pi (the
difference is Z x Z, two virtual RZ gates). Qiskit's WrapAngles enforces a target's bounds; a plain 2 pi wrap is
what a compiler does when it only reduces angles modulo 2 pi.

Checked during development, before the predictions (not a prediction): the rewrite preserves the unitary,
global phase included, to 8e-16 on 50 random 3-qubit circuits and never increases the summed area.

Set-up (fixed before the run):
  Part A, one qubit: theta in {0.3, 0.9, ..., 6.3} (step 0.6, 11 values), eps ~ N(0, 0.05^2), 20000 samples
    (seed 160): the mean of cos(theta (1 + eps)) against cos(theta) exp(-theta^2 sigma^2 / 2).
  Part B, QAOA MaxCut on a 4-qubit ring, p = 2, perfect |+>^4 start, cost layer RZZ(2 gamma) on the 4 edges,
    mixer RX(2 beta); 30 angle sets with gamma, beta uniform in [0, 2 pi) (seed 160); 5 device instances per
    set (seeds 0-4). Noise model of a fractional gate with angle t: the gate acts with angle t (1 + eps_g), eps_g
    fixed per qubit (RX) or per edge (RZZ) for a device instance, ~ N(0, 0.02^2); then thermal relaxation on each
    of its qubits for tau = tau_pi |t| / pi, tau_pi / T1 = 0.01, T2 = T1. RZ gates are exact and instantaneous.
    Exact density matrices. Compilations: as given (angles in [0, 4 pi)); 2 pi wrap (every angle in
    (-pi, pi]); minimal area (qang.pulse_area.minimal_area_circuit: RZZ in (-pi/2, pi/2]). Error: total
    variation distance (TVD) between the noisy and ideal output distributions.

Pre-registered predictions (sha256 of this file recorded in examples/data/pulse_area_qg.prereg before the run):
  P1  Part A: |Monte Carlo - formula| < 0.002 at all 11 angles.
  P2  median TVD: minimal area < 2 pi wrap < as given.
  P3  minimal area has TVD <= 2 pi wrap in at least 90% of the 150 instances.
  P4  median of TVD(2 pi wrap) / TVD(minimal area) >= 1.5.

python examples/pulse_area_qg.py

Findings (pre-registration sha256 6fca3078...2b6a, recorded 2026-10-17 20:27 UTC; the run used that exact file,
then only this block was added; data/pulse_area_qg_run.log):

  Part A: |Monte Carlo - cos(theta) exp(-theta^2 sigma^2 / 2)| <= 0.0016 at all 11 angles.
  Part B (150 instances):   median TVD   median summed area (rad)
    as given (angles to 4 pi)  0.134        101.3
    2 pi wrap                  0.033         23.1
    minimal area               0.024         19.7

  P1 PASS.
  P2 PASS: minimal area < 2 pi wrap < as given.
  P3 PASS: minimal area is at least as good in 93% of the instances.
  P4 FAIL: the median gain over the 2 pi wrap is 1.14x, not 1.5x. The period-pi RZZ wrap lowers the summed
     area by only 15% more, because the RX half of the area is already minimal after the 2 pi wrap.
  * The error follows the area: the 2 pi wrap cuts the area 4.4x and the TVD 4.1x; the RZZ wrap cuts the area
    1.17x and the TVD 1.14x.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import pulse_area as PA  # noqa: E402

SIGMA_A, N_MC = 0.05, 20000
N, P, SETS, DEVICES = 4, 2, 30, 5
SIGMA_CAL, TAU_PI_T1 = 0.02, 0.01
EDGES = [(k, (k + 1) % N) for k in range(N)]
OUT = os.path.join(HERE, "data", "pulse_area_qg.json")


def part_a():
    rng = np.random.default_rng(160)
    rows = []
    for th in np.arange(0.3, 6.31, 0.6):
        eps = rng.normal(0, SIGMA_A, N_MC)
        mc = float(np.mean(np.cos(th * (1 + eps))))
        rows.append((float(th), mc, PA.rx_qg_z_random_amplitude(th, SIGMA_A)))
    return rows


def qaoa(gammas, betas):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(N)
    for g, b in zip(gammas, betas):
        for a, c in EDGES:
            qc.rzz(2 * g, a, c)
        for q in range(N):
            qc.rx(2 * b, q)
    return qc


def wrap_2pi(qc):
    from qiskit import QuantumCircuit

    out = QuantumCircuit(N, global_phase=qc.global_phase)
    for inst in qc.data:
        op = inst.operation
        if op.name in ("rx", "rzz"):
            t, k = PA.minimal_area_angle(float(op.params[0]), 2 * math.pi)
            out.append(op.__class__(t), inst.qubits)
            out.global_phase += math.pi * (k % 2)
        else:
            out.append(op, inst.qubits)
    return out


def noisy_probs(qc, seed):
    from qiskit.circuit.library import RXGate, RZZGate
    from qiskit.quantum_info import DensityMatrix, Statevector
    from qiskit_aer.noise import thermal_relaxation_error

    rng = np.random.default_rng(seed)
    eps_q = rng.normal(0, SIGMA_CAL, N)
    eps_e = {e: rng.normal(0, SIGMA_CAL) for e in EDGES}
    rho = DensityMatrix(Statevector.from_label("+" * N))
    for inst in qc.data:
        op, qs = inst.operation, [qc.find_bit(q).index for q in inst.qubits]
        if op.name == "rx":
            t = float(op.params[0])
            rho = rho.evolve(RXGate(t * (1 + eps_q[qs[0]])), qs)
        elif op.name == "rzz":
            t = float(op.params[0])
            e = tuple(sorted(qs)) if tuple(sorted(qs)) in eps_e else (max(qs), min(qs))
            rho = rho.evolve(RZZGate(t * (1 + eps_e.get(e, eps_e.get(tuple(qs), 0.0)))), qs)
        else:
            rho = rho.evolve(op, qs)
            continue
        tau = TAU_PI_T1 * abs(t) / math.pi
        ch = thermal_relaxation_error(1.0, 1.0, tau).to_quantumchannel()
        for q in qs:
            rho = rho.evolve(ch, [q])
    return rho.probabilities()


def ideal_probs(qc):
    from qiskit.quantum_info import Statevector

    return Statevector.from_label("+" * N).evolve(qc).probabilities()


def main():
    a = part_a()
    print(json.dumps({"part A": [[round(t, 1), round(m, 4), round(f, 4)] for t, m, f in a]}), flush=True)
    rng = np.random.default_rng(160)
    rows = []
    for s in range(SETS):
        g, b = rng.uniform(0, 2 * math.pi, P), rng.uniform(0, 2 * math.pi, P)
        base = qaoa(g, b)
        ideal = ideal_probs(base)
        comps = {"as_given": base, "wrap_2pi": wrap_2pi(base), "min_area": PA.minimal_area_circuit(base)}
        for d in range(DEVICES):
            r = {"set": s, "device": d}
            for k, qc in comps.items():
                r[k] = 0.5 * float(np.abs(noisy_probs(qc, d) - ideal).sum())
                r["area_" + k] = PA.area(qc)
            rows.append(r)
    med = {k: float(np.median([r[k] for r in rows])) for k in ("as_given", "wrap_2pi", "min_area")}
    area = {k: float(np.median([r["area_" + k] for r in rows])) for k in ("as_given", "wrap_2pi", "min_area")}
    frac = float(np.mean([r["min_area"] <= r["wrap_2pi"] for r in rows]))
    ratio = float(np.median([r["wrap_2pi"] / r["min_area"] for r in rows]))
    print(json.dumps({"median TVD": {k: round(v, 4) for k, v in med.items()},
                      "median area": {k: round(v, 2) for k, v in area.items()},
                      "min_area <= wrap_2pi": frac, "median ratio wrap / min": round(ratio, 3)}), flush=True)
    v = {"P1": all(abs(m - f) < 0.002 for _, m, f in a),
         "P2": med["min_area"] < med["wrap_2pi"] < med["as_given"],
         "P3": frac >= 0.9,
         "P4": ratio >= 1.5}
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"part_a": a, "rows": rows, "median": med, "area": area, "fraction": frac, "ratio": ratio,
                   "verdict": v}, fh, indent=1)
    return v


if __name__ == "__main__":
    main()
