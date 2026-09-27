"""
IonQ noisy simulator (forte-1): LiH with the parity + N mod 4 checks of §52.

The prediction for a candidate hardware track, recorded before any QPU run.
Same circuits as §52 part B: the 3-layer LiH ansatz on 6 qubits, one
ancilla for the electron parity (6 CNOTs), one for N mod 4 (Hadamard test
of prod_j S_j, 6 controlled phases), then the basis rotation of each of the
17 measurement groups; 2 readout-calibration circuits. 8 qubits, 19
circuits, 2000 shots each (the simulator's cap). From the same shots:

  raw            ancillas ignored
  parity         keep ancilla 1 = 0
  parity + mod4  keep ancilla 1 = 0 and ancilla 2 = 1  (N = 2 mod 4)
  + N filter     and N = 2 in the Z group

Errors are against the noiseless energy of the same circuit (mHa); the
noiseless L3 circuit is 0.66 mHa above FCI and classical Hartree-Fock
16.3 mHa above it. Results in examples/data/ionq_sim_results.json under
"ionq_sim|forte-1|lih_mod4|<tag>". Free; nothing is sent to a QPU.

Findings:

  forte-1, 5 runs (mean +- std, mHa vs the noiseless circuit):

    raw               145.5 +- 10.1
    parity            116.3 +- 11.6    (kept 0.70)
    parity + mod 4     38.6 +-  8.7    (kept 0.56)
    + N filter (Z)     42.0 +- 10.2

  * The mod-4 check survives the vendor noise model: it cuts the error
    3.8x (parity alone 1.25x), as in §52 (6x on the generic model).
  * forte-1 is noisier than the generic all-to-all model (raw 146 vs 84
    mHa), so the checked estimate, about 39 mHa above FCI, stays above
    Hartree-Fock (16.3): the "beats HF" result of §52 does not carry over
    to this noise level.
  * With 2000 shots and 56 % kept, run-to-run spread is 9 mHa: the hardware
    track would need at least this many shots per circuit.

Pre-registered criterion for a hardware run: holds if parity + mod 4 cuts
the error by at least 2.5x in each run while parity alone gives less than
1.5x; fails if the cut is below 1.5x.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import filter_scaling_lih_qg as F  # noqa: E402
import lih_parity_verification_qg as V  # noqa: E402

N, NE = V.N, V.NE
LAYERS = 3
HF_MHA, ANSATZ_MHA = 16.3, 0.66


def circuits(layers=LAYERS):
    return V._parity_circuits(layers, mod4=True)


def ideal_energy(groups, bases, layers=LAYERS):
    from qiskit.quantum_info import Statevector

    e = 0.0
    for g, b in zip(groups, bases):
        qc = V.L.ansatz(V.L.PARAMS, layers).copy()
        for q, ch in enumerate(b):
            if ch == "X":
                qc.h(q)
            elif ch == "Y":
                qc.sdg(q)
                qc.h(q)
        e += F._group_energy(g, np.real(Statevector(qc).probabilities()), N)
    return e


def analyse(P, groups, bases, layers=LAYERS):
    """P: outcome distributions (8-qubit little-endian index) in circuit order."""
    n1 = N + 2
    inv = F._readout_inverse(P[-2], P[-1], n1)
    idx = np.arange(2**n1)
    a1, a2 = (idx >> N) & 1, (idx >> (N + 1)) & 1
    sysidx = idx & (2**N - 1)
    masks = {"raw": np.ones(2**n1, bool), "parity": a1 == 0, "parity_mod4": (a1 == 0) & (a2 == 1)}
    z = [k for k, b in enumerate(bases) if all(ch in "IZ" for ch in b)][0]
    tot = {k: 0.0 for k in masks}
    tot["parity_mod4_nfilter"] = 0.0
    kept = {k: [] for k in masks}
    for k, (g, p) in enumerate(zip(groups, P[:-2])):
        pm = np.clip(inv @ p, 0, None)
        full = pm.sum()
        for name, m in masks.items():
            s = np.bincount(sysidx, weights=pm * m, minlength=2**N)
            kept[name].append(s.sum() / full)
            e = F._group_energy(g, s / s.sum(), N)
            tot[name] += e
            if name == "parity_mod4":
                if k == z:
                    nf = s * (V._W == NE)
                    tot["parity_mod4_nfilter"] += F._group_energy(g, nf / nf.sum(), N)
                else:
                    tot["parity_mod4_nfilter"] += e
    e0 = ideal_energy(groups, bases, layers)
    out = {k: 1e3 * (v - e0) for k, v in tot.items()}
    out.update({f"kept_{k}": float(np.mean(v)) for k, v in kept.items()})
    return out


def run(mode, noise="forte-1", key=None, seed=11):
    import ionq_sim_hubbard_qaoa as I

    groups, bases, circs = circuits()
    if mode == "local":
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        import hubbard_trotter_qg_filters as HB

        tc = transpile(circs, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=1)
        res = AerSimulator(noise_model=HB.all_to_all_noise_model()).run(tc, shots=I.SHOTS, seed_simulator=seed).result()
        counts = [res.get_counts(i) for i in range(len(tc))]
    else:
        counts = I._run(I.get_backend("ionq_sim"), circs, noise, "ionq_sim", key)
    P = [I._probs(c, N + 2) for c in counts]
    return analyse(P, groups, bases)


def main(argv=None):
    import ionq_sim_hubbard_qaoa as I

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("local", "ionq_sim"), default="local")
    ap.add_argument("--noise", choices=("aria-1", "forte-1"), default="forte-1")
    ap.add_argument("--tag", default="run0")
    args = ap.parse_args(argv)
    key = f"{args.mode}|{args.noise if args.mode == 'ionq_sim' else 'generic'}|lih_mod4|{args.tag}"
    res = I.load_results()
    if key in res:
        out = res[key]
    else:
        try:
            out = run(args.mode, args.noise, key)
        except I.Pending as exc:
            print("pending:", key, "-", exc)
            return None
        if args.mode == "ionq_sim":
            res[key] = out
            I.save_results(res)
            print("saved", key)
    print(f"raw {out['raw']:7.1f}  parity {out['parity']:7.1f}  parity+mod4 {out['parity_mod4']:7.1f}  "
          f"+N filter {out['parity_mod4_nfilter']:7.1f} mHa  (kept {out['kept_parity']:.3f} / {out['kept_parity_mod4']:.3f})")
    return out


if __name__ == "__main__":
    main()
