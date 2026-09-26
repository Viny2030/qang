"""
IonQ noisy simulator (forte-1): H2 at stretched bonds (1.5 and 2.5 A) with
the qg electron-number filter -- the simulator prediction for track A2 of
the hardware plan (examples/ionq_hardware_plan.py), recorded before any
QPU run. Same circuits as the plan: ansatz at the classically optimised
angle for each bond, the ZZZZ basis plus every XY basis of the Hamiltonian,
two readout-calibration circuits. 2000 shots per circuit (the noisy
simulator's cap; the hardware plan uses 5000).

Results go to examples/data/ionq_sim_results.json under
"ionq_sim|forte-1|h2stretched|<tag>". Free; nothing is sent to a QPU.

Findings (forte-1, 3 runs; error vs FCI in mHa, mean +- std):

    R      HF      readout-corrected   + qg filter    kept
    1.5    87.3    17.4 +- 2.2          9.7 +- 2.4    0.978
    2.5    233.1   11.1 +- 2.7          8.8 +- 2.8    0.980

The filter cuts the error by 44 % at 1.5 A and 21 % at 2.5 A (the gain
shrinks with the bond length, as in §21), and both stay far below HF.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import chemistry_qg_symmetry_witness as CH  # noqa: E402
import ionq_sim_hubbard_qaoa as I  # noqa: E402

BONDS = ("1.5", "2.5")


def _hamiltonian(r):
    from qiskit.quantum_info import SparsePauliOp

    with open(os.path.join(HERE, "data", "h2_dissociation_jw.json"), encoding="utf-8") as fh:
        d = json.load(fh)[r]
    return SparsePauliOp.from_list(list(d["terms"].items())), d["hf"], d["fci"]


def circuits():
    out, index = [], []
    for r in BONDS:
        h, _, _ = _hamiltonian(r)
        t = CH.optimal_angle(h)
        labels = ["ZZZZ"] + [l for l, _ in CH._terms(h) if any(ch in "XY" for ch in l)]
        for l in labels:
            out.append(CH._measure_circuit(t, l, 0))
            index.append((r, l))
    return out + CH._calibration_circuits(), index


def analyse(counts, index):
    P = [I._probs(c, CH.N_QUBITS) for c in counts]
    inv = CH._readout_inverse(P[-2], P[-1])
    rows = {}
    for r in BONDS:
        h, hf, fci = _hamiltonian(r)
        terms = CH._terms(h)
        block = {l: inv @ P[i] for i, (rr, l) in enumerate(index) if rr == r}
        raw_z = P[[i for i, (rr, l) in enumerate(index) if rr == r and l == "ZZZZ"][0]]
        p_z = block.pop("ZZZZ")
        rows[r] = {
            "hf_mHa": 1e3 * (hf - fci),
            "readout_mHa": 1e3 * (CH._energy(p_z, block, terms) - fci),
            "qg_filter_mHa": 1e3 * (CH._energy(CH.qg_filter(p_z), block, terms) - fci),
            "kept": float(np.sum(raw_z * (CH._WEIGHT == CH.N_ELECTRONS))),
            "mean_qg_z": float(np.sum(raw_z * (1 - 2 * CH._WEIGHT / CH.N_QUBITS))),
        }
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("local", "ionq_sim"), default="local")
    ap.add_argument("--noise", choices=("aria-1", "forte-1"), default="forte-1")
    ap.add_argument("--tag", default="run0")
    args = ap.parse_args(argv)
    key = f"{args.mode}|{args.noise if args.mode == 'ionq_sim' else 'generic'}|h2stretched|{args.tag}"
    res = I.load_results()
    if key in res:
        out = res[key]
    else:
        circs, index = circuits()
        backend = I.get_backend(args.mode) if args.mode == "ionq_sim" else None
        try:
            counts = I._run(backend, circs, args.noise if args.mode == "ionq_sim" else None, args.mode, key)
        except I.Pending as exc:
            print("pending:", key, "-", exc)
            return None
        out = analyse(counts, index)
        if args.mode == "ionq_sim":
            res[key] = out
            I.save_results(res)
            print("saved", key)
    for r, v in out.items():
        print(f"R = {r} A: HF {v['hf_mHa']:6.1f}  readout {v['readout_mHa']:6.1f}  + qg filter {v['qg_filter_mHa']:6.1f} mHa"
              f"  kept {v['kept']:.3f}")
    return out


if __name__ == "__main__":
    main()
