"""
IonQ noisy cloud simulator, part 2: zero-noise extrapolation vs the qg
filter on H2 (§24), and noise-aware Grover amplitude estimation (§37).

A practical finding first. Folding CX gates (CX -> CX^3) does NOT
amplify the noise on IonQ: the service recompiles circuits written in
the standard gate set and the folded pairs disappear (a Bell pair with
1, 3, 9, 21 CX kept P(00) + P(11) = 0.99 on the aria-1 model). Folding
must be done in IonQ's native gate set: each Molmer-Sorensen gate
MS(phi0, phi1) becomes MS . MS(phi0 + 1/2, phi1) . MS (the middle one is
its inverse), which the service executes as written (the same Bell test
drops to 0.94 and 0.87 with 9 and 21 MS gates).

A. H2 (as §21, §24): energy error vs FCI, readout-mitigated, + qg
   filter, + ZNE (MS folding x1, x3, x5, Richardson), + ZNE + qg.
B. Grover amplitude estimation on 3 qubits: A = Ry(2t) on the flag,
   then CX to two more qubits (so the Grover iterate has real two-qubit
   gates), a = sin^2 t = 0.3, depths k = 0, 1, 2, 4. Estimators: MLAE
   that ignores noise; noise-aware MLAE that fits a visibility v per
   iteration together with a, qg_k = v^k T_{2k+1}(qg_0); and Monte Carlo
   with the same number of oracle queries (binomial error).

Modes: "local" (Aer, generic all-to-all noise; CX folding is valid
there) and "ionq_sim" (noise_model aria-1 / forte-1). Nothing goes to a
QPU. Results are appended to examples/data/ionq_sim_results.json.

Findings (recorded in examples/data/ionq_sim_results.json; 2000 shots
per circuit, the IonQ noisy-simulator limit):

  A. H2, energy error vs FCI (mHa), 6 runs per model (HF = 20.3):
                readout-mitigated  + qg filter    + ZNE          + ZNE + qg
     aria-1     31.5 ± 7.8         14.7 ± 5.8     4.2 ± 20.3     5.5 ± 16.9
     forte-1    34.8 ± 6.3         13.9 ± 5.4     6.6 ± 19.0     3.0 ± 17.8
     (mean ± std; RMS: filter 15.8 / 14.9, ZNE 20.7 / 20.1, ZNE+qg 17.8 / 18.1)
   * Native MS folding does amplify the noise (raw error 31 -> 84 -> 133
     mHa at scales 1, 3, 5 on aria-1); CX folding did not.
   * The qg filter halves the error in every run, for free, and its mean
     sits below Hartree-Fock. ZNE is nearly unbiased but, at 2000 shots,
     its extrapolation amplifies shot noise (±17-20 mHa), so its typical
     error is WORSE than the filter's. It would need ~10-15x more shots
     per scale to match the filter's spread.
   * The witness barely moves (mean qg_Z +0.004) and 2-3% of shots are
     dropped, as in §20: trapped-ion noise here is unital, but the few
     number-violating shots cost a lot of energy.

  B. Grover amplitude estimation (a = 0.3, 36,000 queries, 3 runs):
     error of a_hat: MLAE naive RMS 0.0012 / 0.0006, noise-aware
     0.0005 / 0.0015 (aria-1 / forte-1), fitted visibility 0.965-0.985
     per iteration. Monte Carlo with the same queries would have a
     binomial error of 0.0024, but on the device its circuit is biased by
     the noise: the depth-0 estimate is off by 0.011-0.019 RMS. MLAE is
     7-30x better than that realistic Monte Carlo. At this noise level
     (depth <= 4) modelling the visibility does not change the result.

  C. Leung code: not run. The IonQ noise models have no amplitude damping
     (trapped-ion T1 is effectively infinite), so by the §32 rule
     (Leung only if T2 > T1) the Leung code cannot help there.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import amplitude_estimation_chebyshev_qg as G  # noqa: E402
import chemistry_qg_symmetry_witness as CH  # noqa: E402
import ionq_sim_hubbard_qaoa as I  # noqa: E402
from error_mitigation_qg_vs_zne import RICHARDSON, SCALES, fold_cx  # noqa: E402

A_TRUE = 0.3
GROVER_DEPTHS = (0, 1, 2, 4)


# --------------------------------------------------------------------- #
# native-gate folding
# --------------------------------------------------------------------- #
def fold_ms(qc: QuantumCircuit, scale: int) -> QuantumCircuit:
    """MS -> MS (MS^-1 MS)^((scale-1)/2), with MS^-1 = MS(phi0 + 1/2, phi1, theta)."""
    from qiskit_ionq import MSGate

    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale must be a positive odd integer.")
    out = qc.copy_empty_like()
    for inst in qc.data:
        out.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name == "ms":
            phi0, phi1, theta = (float(x) for x in inst.operation.params)
            for _ in range((scale - 1) // 2):
                out.append(MSGate(phi0 + 0.5, phi1, theta), inst.qubits)
                out.append(MSGate(phi0, phi1, theta), inst.qubits)
    return out


def native_backend():
    from qiskit_ionq import IonQProvider

    from ionq_validation import read_api_key

    return IonQProvider(token=read_api_key()).get_backend("simulator", gateset="native")


def _submit(circuits, noise, mode, key, prebuilt=False):
    """prebuilt: circuits are already in the native gate set."""
    if mode == "local":
        from qiskit_aer import AerSimulator

        import hubbard_trotter_qg_filters as HB

        sim = AerSimulator(noise_model=HB.all_to_all_noise_model())
        tc = transpile(circuits, basis_gates=["cx", "rz", "sx", "x"], optimization_level=0)
        res = sim.run(tc, shots=I.SHOTS, seed_simulator=11).result()
        return [res.get_counts(i) for i in range(len(tc))]
    backend = native_backend() if prebuilt else I.get_backend("ionq_sim")
    return I._run(backend, circuits, noise, "ionq_sim", key, prebuilt=prebuilt)


# --------------------------------------------------------------------- #
# A. H2: ZNE vs qg filter
# --------------------------------------------------------------------- #
def h2_circuits():
    t = CH.optimal_angle()
    return [CH._measure_circuit(t, "ZZZZ", 0)] + [CH._measure_circuit(t, l, 0) for l in CH.XY_TERMS]


def h2_zne(noise, mode, key=None):
    base = h2_circuits()
    cal = CH._calibration_circuits()
    if mode == "local":
        batch = [fold_cx(c, s) for s in SCALES for c in base] + cal
        counts = _submit(batch, noise, mode, key)
    else:
        nb = native_backend()
        native = transpile(base + cal, backend=nb, optimization_level=1)
        batch = [fold_ms(c, s) for s in SCALES for c in native[:len(base)]] + native[len(base):]
        counts = _submit(batch, noise, mode, key, prebuilt=True)
    shots = I.SHOTS
    P = [CH._probs({k.replace(" ", ""): v for k, v in c.items()}, shots) if not any(k.startswith("0x") for k in c)
         else _hex_probs(c, shots) for c in counts]
    inv = CH._readout_inverse(P[-2], P[-1])
    k = len(base)
    e_raw, e_qg = [], []
    for i in range(len(SCALES)):
        block = [inv @ p for p in P[i * k:(i + 1) * k]]
        p_z, p_xy = block[0], dict(zip(CH.XY_TERMS, block[1:]))
        e_raw.append(CH._energy(p_z, p_xy, CH._TERMS))
        e_qg.append(CH._energy(CH.qg_filter(p_z), p_xy, CH._TERMS))
    p0 = P[0]
    return {
        "mean_qg_z": float(np.sum(p0 * (1 - 2 * CH._WEIGHT / CH.N_QUBITS))),
        "kept": [float(np.sum(P[i * k] * (CH._WEIGHT == CH.N_ELECTRONS))) for i in range(len(SCALES))],
        "raw_mHa": 1e3 * (e_raw[0] - CH.FCI_ENERGY),
        "qg_filter_mHa": 1e3 * (e_qg[0] - CH.FCI_ENERGY),
        "zne_mHa": 1e3 * (float(RICHARDSON @ np.array(e_raw)) - CH.FCI_ENERGY),
        "zne_qg_mHa": 1e3 * (float(RICHARDSON @ np.array(e_qg)) - CH.FCI_ENERGY),
        "raw_by_scale_mHa": [1e3 * (e - CH.FCI_ENERGY) for e in e_raw],
    }


def _hex_probs(counts, shots):
    p = np.zeros(2**CH.N_QUBITS)
    for key, c in counts.items():
        p[int(key, 16)] += c
    return p / shots


# --------------------------------------------------------------------- #
# B. Grover amplitude estimation on 3 qubits
# --------------------------------------------------------------------- #
def _a_op(t):
    qc = QuantumCircuit(3)
    qc.ry(2 * t, 0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    return qc


def grover_circuit(t, k):
    a = _a_op(t)
    qc = QuantumCircuit(3)
    qc.compose(a, inplace=True)
    for _ in range(k):
        qc.z(0)                                   # S_good: good = flag qubit 0 in |1>
        qc.compose(a.inverse(), inplace=True)
        qc.x([0, 1, 2])                           # S_0 = reflection about |000> (up to phase)
        qc.h(2)
        qc.ccx(0, 1, 2)
        qc.h(2)
        qc.x([0, 1, 2])
        qc.compose(a, inplace=True)
    qc.measure_all()
    return qc


def flag_counts(counts):
    """(# outcomes with flag 0, total) from 3-qubit counts."""
    n0 = tot = 0
    for key, c in counts.items():
        k = key.replace(" ", "")
        idx = int(k, 16) if k.startswith("0x") else int(k, 2)
        tot += c
        if not idx & 1:
            n0 += c
    return n0, tot


def mle_with_visibility(n0s, tots, ks, v_grid=np.linspace(0.5, 1.0, 101)):
    """Joint MLE of (qg_0, v) with qg_k = v^k T_{2k+1}(qg_0)."""
    best = (-np.inf, None, None)
    for v in v_grid:
        ll = np.zeros_like(G.Q_GRID)
        for n0, n, k in zip(n0s, tots, ks):
            pp = np.clip((1 + v**k * G.chebyshev_t(2 * k + 1, G.Q_GRID)) / 2, 1e-15, 1 - 1e-15)
            ll += n0 * np.log(pp) + (n - n0) * np.log(1 - pp)
        i = int(np.argmax(ll))
        if ll[i] > best[0]:
            best = (ll[i], float(G.Q_GRID[i]), float(v))
    return best[1], best[2]


def grover(noise, mode, key=None, a=A_TRUE):
    t = math.asin(math.sqrt(a))
    circuits = [grover_circuit(t, k) for k in GROVER_DEPTHS]
    counts = _submit(circuits, noise, mode, key)
    n0s, tots = zip(*[flag_counts(c) for c in counts])
    q_naive = G.mle(list(n0s), list(GROVER_DEPTHS), tots[0], 0.0)
    q_aware, v = mle_with_visibility(n0s, tots, GROVER_DEPTHS)
    queries = sum(n * (2 * k + 1) for n, k in zip(tots, GROVER_DEPTHS))
    return {
        "a_true": a, "a_naive": (1 - q_naive) / 2, "a_aware": (1 - q_aware) / 2, "visibility": v,
        "a_k0_only": 1 - n0s[0] / tots[0], "queries": queries,
        "mc_rmse_same_queries": math.sqrt(a * (1 - a) / queries),
        "flag_qg": [2 * n / t_ - 1 for n, t_ in zip(n0s, tots)],
        "flag_qg_ideal": [float(G.flag_qg(1 - 2 * a, k)) for k in GROVER_DEPTHS],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("local", "ionq_sim"), default="local")
    ap.add_argument("--noise", choices=("aria-1", "forte-1"), default="aria-1")
    ap.add_argument("--only", choices=("h2zne", "grover"), required=True)
    ap.add_argument("--tag", default="run0")
    args = ap.parse_args(argv)
    key = f"{args.mode}|{args.noise if args.mode == 'ionq_sim' else 'generic'}|{args.only}|{args.tag}"
    res = I.load_results()
    if key in res:
        print("already done:", key)
        return res[key]
    noise = args.noise if args.mode == "ionq_sim" else None
    try:
        out = h2_zne(noise, args.mode, key) if args.only == "h2zne" else grover(noise, args.mode, key)
    except I.Pending as exc:
        print("pending:", key, "-", exc)
        return None
    res = I.load_results()
    res[key] = out
    I.save_results(res)
    print("saved", key, json.dumps(out)[:300])
    return out


if __name__ == "__main__":
    main()
