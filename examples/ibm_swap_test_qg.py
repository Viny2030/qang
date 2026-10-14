"""
Swap test and inversion test with the qg filter, and a fidelity classifier (§149)

How similar are two quantum states? The fidelity F = |<x|y>|^2 is a
probability, so one shot does not give it: it is estimated from many shots,
each on fresh copies of both states. Two circuits estimate it for qang's
unary (weight-1) states:

  inversion test (qang.ibm.unary_overlap_circuit): load x, undo the loader of
    y; P(e_0) = F. n qubits; every gate conserves the weight, so the filter
    keeps the weight-1 shots.
  swap test (qang.ibm.swap_test_circuit): ancilla in |+>, x and y on two
    registers, n controlled SWAPs, H; P(ancilla 0) = (1 + F) / 2. 2n + 1
    qubits. The controlled SWAPs conserve the total weight (2) of the data
    registers, so the filter keeps those shots, but the ancilla, which
    carries the answer, is outside the symmetry.

Fidelity classifier (as in Blank et al., npj Quantum Inf. 6, 41 (2020), with
class prototypes): iris versicolor against virginica, 4 features
standardized on the training set, embedded as v = (z, 1)/|(z, 1)| on n = 5
qubits; 15 training and 20 test flowers per class (seed 149); prototype of a
class = normalized mean of its training embeddings; a test flower goes to
the class whose prototype has the larger fidelity. Noiseless accuracy 37/40;
smallest noiseless margin |F_1 - F_0| = 0.048. Hypothesis test ("are these
states equal?"): the 10 first test flowers against themselves (F = 1).

Set-up (fixed before the run): 90 pairs (80 classifier + 10 identical) per
test and backend, 2000 shots, IBM Runtime's Sampler in local mode,
optimization_level 1, seed 137, on FakeBrisbane, FakeSherbrooke, FakeTorino,
FakeFez, FakeKingston, FakeOsaka. Transpiled size (structural, before the
run): inversion test 14-16 native two-qubit gates, swap test 113-119.
Fidelity error: mean |F_hat - F| over the 90 pairs.

Pre-registered predictions (committed before the run):
  S1  inversion test: filter + readout error below raw on all 6 backends.
  S2  swap test: filter + readout error below raw on all 6 backends.
  S3  the inversion test (filter + readout) beats the swap test (filter +
      readout) on all 6 backends.
  S4  the classifier with the inversion test and filter + readout gets at
      least 35 of 40 on all 6 backends (noiseless: 37).
  S5  the filter helps the swap test less, because the ancilla is outside
      the symmetry: raw error / filter + readout error is smaller for the
      swap test than for the inversion test on at least 5 of 6 backends.

python examples/ibm_swap_test_qg.py
python examples/ibm_swap_test_qg.py --backend FakeTorino   # one backend (written to data/ibm_swap_test_qg_parts/)
python examples/ibm_swap_test_qg.py --aggregate            # score the per-backend parts

Amendment (before any result was seen): the first run, all backends in one
process, stopped at the 2-hour limit of the environment (the 11-qubit swap
test takes about 20 s per circuit to simulate with noise). The run is split
into one process per backend with the same protocol, and scored by
--aggregate.

Findings:

Fidelity error mean |F_hat - F| over 90 pairs (filter + readout), the ratio
raw / filter + readout, the mean F_hat of the 10 identical pairs (exact 1),
correct classifications of 40 (noiseless 37) and the share of shots kept:

                   inversion test (12-16 two-qubit gates)  swap test (113-119)
  backend          raw    f+ro   ratio  same   acc  kept   raw    f+ro   ratio  same   acc  kept
  FakeBrisbane     0.139  0.036  3.8    0.92   37   0.74   0.267  0.162  1.7    0.63   36   0.52
  FakeSherbrooke   0.094  0.020  4.7    0.96   37   0.82   0.298  0.216  1.4    0.49   38   0.53
  FakeTorino       0.138  0.012  11.2   1.00   37   0.70   0.248  0.137  1.8    0.69   37   0.54
  FakeFez          0.047  0.011  4.3    1.00   37   0.90   0.159  0.089  1.8    0.81   37   0.71
  FakeKingston     0.094  0.010  9.6    1.00   37   0.79   0.156  0.095  1.6    0.80   37   0.71
  FakeOsaka        0.135  0.030  4.4    0.93   37   0.74   0.354  0.239  1.5    0.44   38   0.41

  S1 PASS: inversion test, filter + readout below raw on all 6 (3.8-11.2x).
  S2 PASS: swap test, filter + readout below raw on all 6 (1.4-1.8x).
  S3 PASS: the inversion test beats the swap test on all 6 (0.010-0.036
     against 0.089-0.239).
  S4 PASS: the classifier with the inversion test and filter + readout gets
     37 of 40 on all 6, the noiseless score.
  S5 PASS: the filter helps the swap test less on all 6 (1.4-1.8x against
     3.8-11.2x): the answer sits on the ancilla, outside the symmetry, and
     the 11-qubit circuit is 7-9 times deeper.
  * The classifier is robust even without the filter: raw swap tests score
    36-38 of 40. Noise shrinks both fidelities of a test flower by about the
    same factor, so their comparison survives; the absolute value does not.
  * The hypothesis test ("are these states equal?") needs the absolute
    value: for identical states the raw swap test reads F = 0.20-0.65 and
    the inversion test 0.67-0.96; with filter + readout the inversion test
    reads 0.92-1.00 and the swap test still only 0.44-0.81.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import ibm  # noqa: E402

SHOTS, SEED, CONST = 2000, 149, 1.0
BACKENDS = ("FakeBrisbane", "FakeSherbrooke", "FakeTorino", "FakeFez", "FakeKingston", "FakeOsaka")
KINDS = ("raw", "readout", "filtered", "filtered_readout")
OUT = os.path.join(HERE, "data", "ibm_swap_test_qg.json")


def dataset():
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    m = y > 0
    X, y = X[m], y[m] - 1
    rng = np.random.default_rng(SEED)
    tr, te = [], []
    for c in (0, 1):
        idx = rng.permutation(np.where(y == c)[0])
        tr += list(idx[:15])
        te += list(idx[15:35])
    mu, sd = X[tr].mean(0), X[tr].std(0)

    def emb(v):
        e = np.append((v - mu) / sd, CONST)
        return e / np.linalg.norm(e)

    E = np.array([emb(v) for v in X])
    P = [E[[i for i in tr if y[i] == c]].mean(0) for c in (0, 1)]
    P = [p / np.linalg.norm(p) for p in P]
    pairs = [(E[i], P[c]) for i in te for c in (0, 1)] + [(E[i], E[i]) for i in te[:10]]
    return pairs, y[te]


def accuracy(F, labels):
    F = np.asarray(F[:80]).reshape(-1, 2)
    return int(((F[:, 1] > F[:, 0]).astype(int) == labels).sum())


def sampler(backend):
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler
    except ImportError:  # pragma: no cover
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    return Sampler(mode=backend)


def run_backend(name, pairs, labels, exact):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile

    b = getattr(fp, name)()
    out = {"backend": name}
    for test, make, read in (("inversion", ibm.unary_overlap_circuit, ibm.overlap_fidelity),
                             ("swap", ibm.swap_test_circuit, ibm.swap_test_fidelity)):
        ts = [transpile(make(x, y), b, optimization_level=1, seed_transpiler=137) for x, y in pairs]
        res = sampler(b).run(ts, shots=SHOTS).result()
        fids = [read(r, t, b) for r, t in zip(res, ts)]
        est = {k: [f[k] for f in fids] for k in KINDS}
        out[test] = {"error": {k: float(np.nanmean(np.abs(np.array(v) - exact))) for k, v in est.items()},
                     "accuracy": {k: accuracy(v, labels) for k, v in est.items()},
                     "identical_mean": {k: float(np.nanmean(v[80:])) for k, v in est.items()},
                     "kept_raw": float(np.mean([f["kept_raw"] for f in fids])),
                     "two_qubit_gates": int(np.mean([sum(1 for i in t.data if i.operation.num_qubits == 2)
                                                     for t in ts])),
                     "estimates": est}
    return out


PARTS = os.path.join(HERE, "data", "ibm_swap_test_qg_parts")


def one(name, pairs, labels, exact):
    try:
        r = run_backend(name, pairs, labels, exact)
        print(json.dumps({"backend": name, **{t: {"err": {k: round(v, 4) for k, v in r[t]["error"].items()},
                                                  "acc": r[t]["accuracy"],
                                                  "same": {k: round(v, 3) for k, v in r[t]["identical_mean"].items()},
                                                  "kept": round(r[t]["kept_raw"], 3)}
                                              for t in ("inversion", "swap")}}), flush=True)
    except Exception as e:  # a backend that cannot run is reported, not hidden
        r = {"backend": name, "failure": f"{type(e).__name__}: {e}"[:300]}
        print(json.dumps(r), flush=True)
    return r


def main(argv=()):
    pairs, labels = dataset()
    exact = np.array([(x @ y) ** 2 for x, y in pairs])
    argv = list(argv)
    if "--backend" in argv:
        name = argv[argv.index("--backend") + 1]
        os.makedirs(PARTS, exist_ok=True)
        r = one(name, pairs, labels, exact)
        with open(os.path.join(PARTS, name + ".json"), "w", encoding="utf-8") as fh:
            json.dump(r, fh)
        return r
    print(json.dumps({"noiseless accuracy": accuracy(exact, labels)}), flush=True)
    if "--aggregate" in argv:
        rows = []
        for name in BACKENDS:
            f = os.path.join(PARTS, name + ".json")
            rows.append(json.load(open(f)) if os.path.exists(f) else {"backend": name, "failure": "not run"})
    else:
        rows = [one(name, pairs, labels, exact) for name in BACKENDS]
    ok = [r for r in rows if "inversion" in r]
    fr = "filtered_readout"
    ratio = lambda r, t: r[t]["error"]["raw"] / r[t]["error"][fr]  # noqa: E731
    v = {"S1": len(ok) == 6 and all(r["inversion"]["error"][fr] < r["inversion"]["error"]["raw"] for r in ok),
         "S2": len(ok) == 6 and all(r["swap"]["error"][fr] < r["swap"]["error"]["raw"] for r in ok),
         "S3": len(ok) == 6 and all(r["inversion"]["error"][fr] < r["swap"]["error"][fr] for r in ok),
         "S4": len(ok) == 6 and all(r["inversion"]["accuracy"][fr] >= 35 for r in ok),
         "S5": sum(ratio(r, "swap") < ratio(r, "inversion") for r in ok) >= 5}
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok_ else 'FAIL'}" for k, ok_ in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"exact": exact.tolist(), "labels": labels.tolist(), "rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main(sys.argv[1:])
