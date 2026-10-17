"""
Quantum kernels (QSVC) with the qg filter against the standard diagonal normalization (§161)

A fidelity kernel estimated on a noisy device is shrunk and shifted: K_ii < 1, and every entry is pulled toward
the value of the noise. The usual repair divides by the diagonal, K_ij / sqrt(K_ii K_jj), which costs the
diagonal circuits. When the kernel circuit conserves the Hamming weight, the qg filter is another repair. The
kernel here is the weight-1 (unary) fidelity kernel K = (x . y)^2 (qang.kernels), estimated with the
inversion test of §149, a degree-2 polynomial kernel that is classically easy. It is a controlled test of
Gram-matrix repair, not a claim of advantage.

Checked during development, before the predictions (not a prediction): the circuit reproduces the exact
kernel to 7e-10; 12.6 two-qubit gates on FakeTorino; with 15 + 15 training and 30 test points the exact
kernel classifies 28 / 30; with the final 40 + 40 split it classifies 35 / 40.

Set-up (fixed before the run): breast-cancer data (scikit-learn), 40 training and 40 test points, 20 per class
each (seed 161); standardized on the training set, PCA to 4 features, a constant 1 appended, unit norm (5
qubits). Circuits: the training Gram matrix (780 pairs i < j and 40 diagonals), the test-train matrix (1600),
the test diagonals (40). FakeBrisbane, FakeTorino, FakeFez: transpile (optimization_level 1, seed 137) and
AerSimulator.from_backend (device noise model), 2000 shots, seed 161. Kernels: exact; raw; raw normalized
(by the raw diagonals); filtered (weight-1 shots); filtered normalized. QSVC: sklearn SVC(kernel="precomputed",
C=1). Gram error: mean |K - K_exact| over the 780 off-diagonal training pairs.

Pre-registered predictions (sha256 of this file recorded in examples/data/quantum_kernel_qg.prereg before the
run):
  K1  Gram error: filtered < raw normalized < raw on all 3 backends.
  K2  every noisy kernel classifies within 2 test points of the exact kernel on all 3 backends.
  K3  |alignment - exact alignment|: filtered < raw on all 3.
  K4  mean raw diagonal < 0.9 and mean filtered diagonal > 0.97 on all 3.
  K5  filtered normalized error <= filtered error + 0.002 on all 3 (normalizing after the filter changes little).

python examples/quantum_kernel_qg.py

Findings (pre-registration sha256 b6aa6ee5...00c8, recorded 2026-10-17 20:14 UTC; the run used that exact file,
then only this block was added; data/quantum_kernel_qg_run.log). Exact kernel: 35 / 40, alignment 0.1237.

  backend (2q, kept)       Gram error raw / raw norm / filtered / filt norm    accuracy (of 40)   diag raw / filt
  FakeBrisbane (16.0, 0.72)  0.097 / 0.016 / 0.016 / 0.015                     32 / 34 / 34 / 35   0.674 / 0.944
  FakeTorino   (13.5, 0.70)  0.101 / 0.021 / 0.017 / 0.014                     34 / 35 / 34 / 34   0.717 / 0.971
  FakeFez      (13.5, 0.89)  0.039 / 0.027 / 0.011 / 0.011                     34 / 34 / 35 / 35   0.953 / 1.000

  K1 PASS: filtered < raw normalized < raw on all 3 (on Brisbane by 0.0003 only).
  K2 FAIL: the raw kernel on Brisbane classifies 32, three below the exact 35; every other kernel is within 1.
  K3 FAIL on 2 of 3: the alignment is normalized by ||K||, so a global shrink barely moves it; raw is as close as
     filtered (Torino 0.1228 / 0.1202, Fez 0.1248 / 0.1217 against 0.1237). It is a poor measure of this damage.
  K4 FAIL: the raw diagonal of FakeFez is 0.953 (above 0.9), the filtered one of FakeBrisbane 0.944 (below 0.97).
  K5 PASS: normalizing after the filter changes the error by -0.003 to -0.0001.
  * The filter repairs the Gram matrix 2.4-6x and needs no diagonal circuits; the normalization needs 2 x 40
    extra circuits and leaves 1.0-2.5x more error than the filter. Both together are best by a hair.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import ibm  # noqa: E402
from qang import kernels as KR  # noqa: E402

N_TRAIN = N_TEST = 40
SHOTS, SEED = 2000, 161
BACKENDS = ("FakeBrisbane", "FakeTorino", "FakeFez")
KINDS = ("raw", "raw_norm", "filtered", "filtered_norm")
OUT = os.path.join(HERE, "data", "quantum_kernel_qg.json")


def data():
    from sklearn.datasets import load_breast_cancer
    from sklearn.decomposition import PCA

    X, y = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(SEED)
    tr = np.concatenate([rng.choice(np.where(y == c)[0], N_TRAIN // 2, replace=False) for c in (0, 1)])
    rest = np.setdiff1d(np.arange(len(y)), tr)
    te = np.concatenate([rng.choice(rest[y[rest] == c], N_TEST // 2, replace=False) for c in (0, 1)])
    mu, sd = X[tr].mean(0), X[tr].std(0)
    p = PCA(4).fit((X[tr] - mu) / sd)
    emb = lambda idx: KR.unary_embedding(p.transform((X[idx] - mu) / sd))  # noqa: E731
    return emb(tr), y[tr], emb(te), y[te]


def accuracy(Ktr, ytr, Kte, yte):
    from sklearn.svm import SVC

    clf = SVC(kernel="precomputed", C=1.0).fit(Ktr, ytr)
    return int((clf.predict(Kte) == yte).sum())


def run_backend(name, A, B):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    b = getattr(fp, name)()
    pairs = ([("tr", i, j) for i in range(N_TRAIN) for j in range(i + 1, N_TRAIN)]
             + [("dtr", i, i) for i in range(N_TRAIN)]
             + [("te", i, j) for i in range(N_TEST) for j in range(N_TRAIN)]
             + [("dte", i, i) for i in range(N_TEST)])
    vec = {"tr": (A, A), "dtr": (A, A), "te": (B, A), "dte": (B, B)}
    qcs = [ibm.unary_overlap_circuit(vec[k][0][i], vec[k][1][j]) for k, i, j in pairs]
    ts = transpile(qcs, b, optimization_level=1, seed_transpiler=137)
    res = AerSimulator.from_backend(b).run(ts, shots=SHOTS, seed_simulator=SEED).result()
    est = [KR.kernel_from_counts(res.get_counts(n)) for n in range(len(pairs))]
    out = {}
    for kind in ("raw", "filtered"):
        Ktr = np.eye(N_TRAIN)
        Kte = np.zeros((N_TEST, N_TRAIN))
        dtr, dte = np.zeros(N_TRAIN), np.zeros(N_TEST)
        for (k, i, j), e in zip(pairs, est):
            v = e[kind]
            if k == "tr":
                Ktr[i, j] = Ktr[j, i] = v
            elif k == "dtr":
                dtr[i] = v
            elif k == "te":
                Kte[i, j] = v
            else:
                dte[i] = v
        Ktr_d = Ktr.copy()
        np.fill_diagonal(Ktr_d, dtr)
        out[kind] = (Ktr_d, Kte, dtr, dte)
        out[kind + "_norm"] = (KR.normalize_gram(Ktr_d, dtr, dtr), KR.normalize_gram(Kte, dte, dtr), dtr, dte)
    gates = float(np.mean([sum(1 for x in t.data if x.operation.num_qubits == 2) for t in ts]))
    kept = float(np.mean([e["kept"] for e in est]))
    return out, gates, kept


def main():
    A, ytr, B, yte = data()
    Ktr0, Kte0 = KR.exact_kernel(A, A), KR.exact_kernel(B, A)
    acc0 = accuracy(Ktr0, ytr, Kte0, yte)
    s = 2 * ytr - 1
    al0 = KR.alignment(Ktr0, s)
    iu = np.triu_indices(N_TRAIN, 1)
    print(json.dumps({"exact accuracy": acc0, "exact alignment": round(al0, 4)}), flush=True)
    rows = []
    for name in BACKENDS:
        out, gates, kept = run_backend(name, A, B)
        r = {"backend": name, "two_qubit_gates": gates, "kept": kept}
        for kind in KINDS:
            Ktr, Kte, dtr, dte = out[kind]
            r[kind] = {"gram_error": float(np.abs(Ktr[iu] - Ktr0[iu]).mean()),
                       "accuracy": accuracy(Ktr, ytr, Kte, yte),
                       "alignment": KR.alignment(Ktr, s),
                       "diag_mean": float(np.mean(dtr))}
        rows.append(r)
        print(json.dumps({"backend": name, "2q": round(gates, 1), "kept": round(kept, 3),
                          **{k: {"err": round(r[k]["gram_error"], 4), "acc": r[k]["accuracy"],
                                 "align": round(r[k]["alignment"], 4), "diag": round(r[k]["diag_mean"], 3)}
                             for k in KINDS}}), flush=True)
    v = {"K1": all(r["filtered"]["gram_error"] < r["raw_norm"]["gram_error"] < r["raw"]["gram_error"] for r in rows),
         "K2": all(abs(r[k]["accuracy"] - acc0) <= 2 for r in rows for k in KINDS),
         "K3": all(abs(r["filtered"]["alignment"] - al0) < abs(r["raw"]["alignment"] - al0) for r in rows),
         "K4": all(r["raw"]["diag_mean"] < 0.9 and r["filtered"]["diag_mean"] > 0.97 for r in rows),
         "K5": all(r["filtered_norm"]["gram_error"] <= r["filtered"]["gram_error"] + 0.002 for r in rows)}
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"exact_accuracy": acc0, "exact_alignment": al0, "rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
