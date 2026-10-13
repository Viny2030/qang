"""
A PCA spectrum read out in the weight-1 sector, with the qg filter (§146)

Quantum PCA (Lloyd, Mohseni & Rebentrost, Nat. Phys. 10, 631 (2014)) treats
the normalized covariance of a data set as a density matrix rho and reads its
principal components from copies of rho. In the unary (weight-1) encoding of
qang, loading the data vectors x_j with weights |x_j|^2 / sum |x|^2 gives
exactly rho = C / tr C, and the RBS circuit of the PCA rotation sends
principal component k to qubit k: P(qubit k excited) = lambda_k / sum(lambda).
Every gate conserves the weight, so the qg filter applies.

This is not the exponential speed-up of quantum PCA: d features use d qubits
and the rotation is computed classically. It is a test of how well a device
reads a spectrum out of rho, with and without the filter, against an exact
answer (numpy's eigenvalues).

Set-up (fixed before the run): d = 5 features, m = 16 samples drawn from a
Gaussian with standard deviations (2.0, 1.4, 1.0, 0.6, 0.35) along random
orthogonal axes (seed 146), centred; the rotation from qang.ibm.pca_rotation;
per sample one circuit unary_loader(x_j) + unary_orthogonal(O) (4 + 10 RBS on
a chain), 2000 shots; IBM Runtime's Sampler in local mode, optimization_level
1, seed 137; the 3 backends of §137 and the 18 of §144 (21). Spectrum
estimate: qang.ibm.unary_spectrum with the four read-outs of ibm_qg.
Error: L1 = sum_k |estimate_k - lambda_k / sum(lambda)|. The exact spectrum
of these data (computed before committing, no circuit run) is (0.640, 0.203,
0.122, 0.023, 0.011): the last two eigenvalues differ by only 0.012.

Pre-registered predictions (committed before the run):
  V1  noiseless (StatevectorSampler, same shots): L1 <= 0.03.
  V2  on every backend that runs (and at least 20 of the 21 run), the
      filtered + readout L1 is below the raw L1.
  V3  the median over the backends that run of raw L1 / (filtered + readout
      L1) is at least 2.
  V4  filtered + readout puts all five eigenvalues in the right order on at
      least 18 of the 21 backends.

python examples/ibm_pca_spectrum_qg.py

Findings:

Noiseless: (0.650, 0.199, 0.116, 0.024, 0.012) against the exact (0.640,
0.203, 0.122, 0.023, 0.011), L1 = 0.020.

L1 error of the spectrum (16 circuits x 2000 shots; 26-28 native two-qubit
gates per circuit; kept = share of shots in the weight-1 sector):

  backend          raw     readout  filtered  filt+ro  order (f+ro)  kept
  FakeBrisbane     0.504   0.419    0.132     0.109    no            0.60
  FakeSherbrooke   0.266   0.192    0.062     0.055    yes           0.76
  FakeTorino       0.278   0.151    0.082     0.049    yes           0.68
  FakeAachen       0.129   0.106    0.016     0.029    yes           0.85
  FakeBerlin       0.129   0.077    0.029     0.024    yes           0.87
  FakeBoston       0.168   0.166    0.039     0.041    yes           0.85
  FakeBrussels     0.388   0.288    0.090     0.071    yes           0.66
  FakeCusco        0.454   0.440    0.230     0.227    yes           0.44
  FakeFez          0.167   0.155    0.050     0.047    yes           0.85
  FakeKawasaki     0.284   0.223    0.063     0.048    yes           0.74
  FakeKingston     0.165   0.100    0.012     0.044    yes           0.77
  FakeKyiv         0.345   0.236    0.070     0.054    yes           0.70
  FakeKyoto        1.762   1.760    0.964     1.121    no            0.15
  FakeMarrakesh    0.116   0.104    0.018     0.018    yes           0.89
  FakeMiami        0.110   0.047    0.016     0.014    yes           0.88
  FakeNighthawk    0.119   0.112    0.026     0.030    yes           0.89
  FakeOsaka        0.445   0.335    0.104     0.076    yes           0.63
  FakePittsburgh   0.090   0.081    0.027     0.025    yes           0.92
  FakeQuebec       0.502   0.475    0.180     0.167    yes           0.55
  FakeStrasbourg   0.248   0.163    0.042     0.044    yes           0.74
  FakeWashingtonV2 0.843   0.734    0.300     0.244    no            0.41

  V1 PASS: noiseless L1 = 0.020 (<= 0.03).
  V2 PASS: all 21 ran; filtered + readout is below raw on every one.
  V3 PASS: median raw / (filtered + readout) = 4.63 (range 1.6-7.7; the
     lowest is FakeKyoto, where only 15% of the shots survive).
  V4 PASS: the five eigenvalues come out in order on 18 of 21 with
     filtered + readout (raw: 13, readout: 14, filtered: 18). On
     FakeBrisbane only the last two (0.023 and 0.011) swap.
  * The raw estimate is biased upwards for every eigenvalue: shots that
    leave the sector put extra excitations on all qubits, so the raw
    "spectrum" sums to more than 1 (median L1 0.27). The filter removes
    that bias; what remains is mostly a flattening of the spectrum
    (depolarizing inside the sector) that grows as the kept share falls.
  * Median L1: raw 0.266, readout 0.166, filtered 0.062, filtered +
    readout 0.048. As in §144, readout correction after the filter helps
    on most devices but not where readout errors are small (Aachen,
    Kingston, Nighthawk).
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from qang import ibm  # noqa: E402

D, M, SHOTS, SEED = 5, 16, 2000, 146
SIGMA = (2.0, 1.4, 1.0, 0.6, 0.35)
BACKENDS = ("FakeBrisbane", "FakeSherbrooke", "FakeTorino",
            "FakeAachen", "FakeBerlin", "FakeBoston", "FakeBrussels", "FakeCusco", "FakeFez", "FakeKawasaki",
            "FakeKingston", "FakeKyiv", "FakeKyoto", "FakeMarrakesh", "FakeMiami", "FakeNighthawk", "FakeOsaka",
            "FakePittsburgh", "FakeQuebec", "FakeStrasbourg", "FakeWashingtonV2")
KINDS = ("raw", "readout", "filtered", "filtered_readout")
OUT = os.path.join(HERE, "data", "ibm_pca_spectrum_qg.json")


def data():
    rng = np.random.default_rng(SEED)
    Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    X = rng.normal(size=(M, D)) * np.array(SIGMA) @ Q.T
    return X - X.mean(axis=0)


def circuits(X):
    O, lam = ibm.pca_rotation(X, center=False)
    out = []
    for x in X:
        qc = ibm.unary_loader(x)
        qc.compose(ibm.unary_orthogonal(O), inplace=True)
        qc.measure_all()
        out.append(qc)
    w = (X ** 2).sum(axis=1)
    return out, w / w.sum(), lam / lam.sum()


def noiseless(qcs, w, exact):
    from qiskit.primitives import StatevectorSampler

    res = StatevectorSampler(seed=SEED).run(qcs, shots=SHOTS).result()
    p = np.zeros(D)
    for r, wj in zip(res, w):
        c = r.data.meas.get_counts()
        p += wj * np.array([sum(v for k, v in c.items() if k[::-1][i] == "1") for i in range(D)]) / SHOTS
    return p, float(np.abs(p - exact).sum())


def sampler(backend):
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler
    except ImportError:  # pragma: no cover
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    return Sampler(mode=backend)


def run_backend(name, qcs, w, exact):
    import qiskit_ibm_runtime.fake_provider as fp
    from qiskit import transpile

    b = getattr(fp, name)()
    ts = [transpile(qc, b, optimization_level=1, seed_transpiler=137) for qc in qcs]
    res = sampler(b).run(ts, shots=SHOTS).result()
    outs = [ibm.ibm_qg(r, t, b, weight=1) for r, t in zip(res, ts)]
    est = {k: ibm.unary_spectrum(outs, w, k)[0] for k in KINDS}
    return {"backend": name,
            "estimate": {k: [float(v) for v in e] for k, e in est.items()},
            "l1": {k: float(np.abs(e - exact).sum()) for k, e in est.items()},
            "order_ok": {k: bool(np.all(np.diff(e) < 0)) for k, e in est.items()},
            "kept_raw": float(np.dot(w, [o["kept_raw"] for o in outs])),
            "two_qubit_gates": int(np.mean([sum(1 for i in t.data if i.operation.num_qubits == 2) for t in ts]))}


def main():
    X = data()
    qcs, w, exact = circuits(X)
    p0, l1_0 = noiseless(qcs, w, exact)
    print(json.dumps({"exact": [round(float(v), 4) for v in exact], "noiseless": [round(float(v), 4) for v in p0],
                      "noiseless L1": round(l1_0, 4)}), flush=True)
    rows = []
    for name in BACKENDS:
        try:
            r = run_backend(name, qcs, w, exact)
            print(json.dumps({"backend": name, **{k: round(v, 4) for k, v in r["l1"].items()},
                              "order ok (f+ro)": r["order_ok"]["filtered_readout"], "kept": round(r["kept_raw"], 3)}),
                  flush=True)
        except Exception as e:  # a backend that cannot run is reported, not hidden
            r = {"backend": name, "failure": f"{type(e).__name__}: {e}"[:300]}
            print(json.dumps(r), flush=True)
        rows.append(r)
    ok = [r for r in rows if "l1" in r]
    ratio = [r["l1"]["raw"] / r["l1"]["filtered_readout"] for r in ok]
    v = {"V1": l1_0 <= 0.03,
         "V2": len(ok) >= 20 and all(r["l1"]["filtered_readout"] < r["l1"]["raw"] for r in ok),
         "V3": len(ok) > 0 and float(np.median(ratio)) >= 2,
         "V4": sum(r["order_ok"]["filtered_readout"] for r in ok) >= 18}
    summary = {"backends run": len(ok), "median raw / filtered_readout": round(float(np.median(ratio)), 2),
               "order ok": {k: sum(r["order_ok"][k] for r in ok) for k in KINDS},
               "median L1": {k: round(float(np.median([r["l1"][k] for r in ok])), 4) for k in KINDS}}
    print(json.dumps(summary))
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok_ else 'FAIL'}" for k, ok_ in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"exact": exact.tolist(), "noiseless": p0.tolist(), "rows": rows, "summary": summary,
                   "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
