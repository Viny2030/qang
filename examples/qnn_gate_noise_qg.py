"""
Gate errors in qang's simulator: calibrated training with T1, T2 and the
published two-qubit gate error (§124)

Section 119 calibrated training from T1 and T2 only and found nothing to
gain, because gate errors dominate the IBM device noise models. Here the
simulator also gets a gate error: after every RBS gate, a two-qubit
depolarizing channel on its pair, with probability

    p = 4/3 * (1 - (1 - e)^m),

where e is the mean published error of the native two-qubit gate (ECR or CZ)
on the coupled pairs of the chosen layout and m the number of those gates
per RBS in the transpiled circuit (the 4/3 converts an average gate error
into a two-qubit depolarizing probability). The simulator (GateNoiseQNN) is a
batched density-matrix simulation that equals WeightQNN.probs when p = 0
(tested).

Models (weight 1, 5 qubits; iris, cancer, wine, digits; 189 test inputs; the
fixed layout and IBM fake backends brisbane, sherbrooke, torino of §119):
  A   trained noiselessly
  C1  trained with qang under T1/T2 (§119)
  C2  trained with qang under T1/T2 + gate error
  D2  trained without qang under T1/T2 + gate error
On each backend (Aer with its noise model, 1000 shots per input) they are read
raw and with the filter. The simulator's prediction for A (accuracy with the
filter, and kept fraction) is compared with the device.

Pre-registered predictions (committed before any noisy run; the simulator is
checked in the tests):
  R1  pooled over the three backends, C2 with qang is at least as accurate
      as A with qang.
  R2  pooled, C2 with qang is at least as accurate as C1 with qang.
  R3  pooled, D2 raw is at least 1 point more accurate than A raw.
  R4  the simulator with the full calibration predicts the kept fraction of
      A on the device within 0.05, averaged over datasets, on every backend.
  R5  pooled, C2 with qang is within 0.5 points of A's noiseless accuracy.

python examples/qnn_gate_noise_qg.py --mode fake --out fake.json

Findings:

FINDINGS_PLACEHOLDER
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
import qnn_full_pipeline_qg as P  # noqa: E402
import qnn_hardware_qg as H  # noqa: E402
import qnn_ionq_datasets_qg as DS  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

N = 5
EPOCHS = 120
ONE = [1 << (N - 1 - q) for q in range(N)]
KEYS = ("noiseless", "A raw", "A qang", "C1 qang", "C2 qang", "D2 raw")


class GateNoiseQNN(WeightQNN):
    """WeightQNN simulated with the batched density matrix, with a two-qubit
    depolarizing error of probability ``gate_error`` after every RBS gate.
    With gate_error = 0 it equals WeightQNN.probs."""

    gate_error = 0.0

    def _tables(self):
        if not hasattr(self, "_perm"):
            idx = np.arange(2 ** self.n)
            self._perm = [idx ^ self._bit(q) for q in range(self.n)]
            self._b = [((idx & self._bit(q)) != 0) for q in range(self.n)]
            self._ham = np.array([[bin(i ^ j).count("1") for j in idx] for i in idx])
            self._pair_idx = {}

    def _depol_pair(self, r, a, b):
        """I/4 (x) Tr_ab(rho) on the qubits a, b."""
        key = (a, b)
        if key not in self._pair_idx:
            idx = np.arange(2 ** self.n)
            self._pair_idx[key] = [idx[(self._b[a] == x) & (self._b[b] == y)] for x in (0, 1) for y in (0, 1)]
        sel = self._pair_idx[key]
        T = sum(r[:, s][:, :, s] for s in sel) / 4
        out = np.zeros_like(r)
        for s in sel:
            out[:, s[:, None], s[None, :]] = T
        return out

    def probs(self, theta, psi, gamma=None, dephasing=0.0):
        self._tables()
        n = self.n
        gam = None if gamma is None else np.broadcast_to(np.asarray(gamma, float), (n,))
        rho = np.einsum("si,sj->sij", psi, psi)
        k = 0
        for _ in range(self.layers):
            for pairs in self.sublayers:
                for a, b in pairs:
                    U = self._rbs(a, b, theta[k])
                    k += 1
                    rho = U[None] @ rho @ U.T[None]
                    if self.gate_error > 0:
                        rho = (1 - self.gate_error) * rho + self.gate_error * self._depol_pair(rho, a, b)
                if dephasing:  # phase flips on all qubits commute with each other and with T1
                    rho = rho * ((1 - 2 * dephasing) ** self._ham)[None]
                for q in range(n):
                    g = 0.0 if gam is None else gam[q]
                    if g:
                        bq = self._b[q]
                        f = np.where(bq, math.sqrt(1 - g), 1.0)
                        nb = (~bq).astype(float)
                        jump = g * rho[:, self._perm[q]][:, :, self._perm[q]] * (nb[:, None] * nb[None, :])[None]
                        rho = rho * (f[:, None] * f[None, :])[None] + jump
        return np.einsum("sii->si", rho).copy()


def gate_calibration(backend, circuit, layout):
    """Two-qubit depolarizing probability per RBS from the published gate errors."""
    from qiskit import transpile

    tc = transpile(circuit, backend=backend, optimization_level=1, seed_transpiler=119, initial_layout=layout)
    name = next(g for g in ("ecr", "cz", "cx") if g in backend.target.operation_names)
    errs = [p.error for (q0, q1), p in backend.target[name].items()
            if q0 in layout and q1 in layout and p is not None and p.error is not None]
    e = float(np.mean(errs))
    m = sum(1 for i in tc.data if i.operation.name == name) / 19  # 4 loader + 15 trained RBS gates
    return {"gate": name, "mean error": e, "per RBS": m, "depolarizing": min(4 / 3 * (1 - (1 - e) ** m), 1.0)}


def gate_model(p_dep, **kw):
    m = GateNoiseQNN(N, 1, **kw)
    m.gate_error = p_dep
    return m


def run_backend(mode, name, epochs=EPOCHS, datasets=Q.DATASETS):
    backend = H.get_backend("fake", name or "fake_brisbane")
    out = {"backend": name if mode == "fake" else "local", "per dataset": {}}
    layout = cal = gcal = None
    for ds in datasets:
        Xtr, Xte, ytr, yte = P.data(ds)
        A = WeightQNN(N, 1).fit(Xtr, ytr, epochs, seed=DS.TRAIN_SEED)
        if layout is None:
            layout, cal = P.calibration(backend, H.build_circuit(A, Xte[0]))
            gcal = gate_calibration(backend, H.build_circuit(A, Xte[0]), layout)
            out["calibration"] = {"gamma": cal["gamma"].tolist(), "dephasing": cal["dephasing"], **gcal}
        noise = {"gamma": cal["gamma"], "dephasing": cal["dephasing"]}
        C1 = WeightQNN(N, 1).fit(Xtr, ytr, epochs, qang=True, seed=DS.TRAIN_SEED, **noise)
        C2 = gate_model(gcal["depolarizing"]).fit(Xtr, ytr, epochs, qang=True, seed=DS.TRAIN_SEED, **noise)
        D2 = gate_model(gcal["depolarizing"]).fit(Xtr, ytr, epochs, qang=False, seed=DS.TRAIN_SEED, **noise)
        n = len(Xte)
        circ = [H.build_circuit(m, x) for m in (A, C1, C2, D2) for x in Xte]
        probs = [H.counts_to_probs(c) for c in P.counts_for(mode, backend, circ, [P.SHOTS] * len(circ), layout)]
        pa, p1, p2, pd = (probs[i * n:(i + 1) * n] for i in range(4))
        def acc(m, pr, q):
            F = np.array([WeightQNN.qg_z(m, p, qang=q)[0] for p in pr])
            d = F @ m.params_[m.n_theta:m.n_theta + m.n_head] + m.params_[-1]
            return int(np.sum((d > 0).astype(int) == yte))

        sim = gate_model(gcal["depolarizing"])
        sim.params_ = A.params_
        pred = sim.probs(A.params_[: A.n_theta], A.encode(Xte), **noise)
        r = {"inputs": n, "noiseless": int(np.sum(A.predict(Xte) == yte)),
             "A raw": acc(A, pa, False), "A qang": acc(A, pa, True), "C1 qang": acc(C1, p1, True),
             "C2 qang": acc(C2, p2, True), "D2 raw": acc(D2, pd, False),
             "A predicted qang": acc(A, pred, True),
             "kept device": float(np.mean([sum(p[i] for i in ONE) for p in pa])),
             "kept predicted": float(np.mean([sum(p[i] for i in ONE) for p in pred]))}
        out["per dataset"][ds] = r
    tot = sum(r["inputs"] for r in out["per dataset"].values())
    out["pooled"] = {k: sum(r[k] for r in out["per dataset"].values()) / tot for k in KEYS + ("A predicted qang",)}
    out["pooled"]["inputs"] = tot
    out["kept"] = {k: float(np.mean([r[k] for r in out["per dataset"].values()])) for k in ("kept device", "kept predicted")}
    return out


def verdict(results):
    Pp = [r["pooled"] for r in results]
    mean = lambda k: float(np.mean([p[k] for p in Pp]))  # noqa: E731
    return {
        "R1": mean("C2 qang") >= mean("A qang"),
        "R2": mean("C2 qang") >= mean("C1 qang"),
        "R3": mean("D2 raw") - mean("A raw") >= 0.01,
        "R4": all(abs(r["kept"]["kept device"] - r["kept"]["kept predicted"]) <= 0.05 for r in results),
        "R5": mean("noiseless") - mean("C2 qang") <= 0.005,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "fake"], default="fake")
    ap.add_argument("--backends", nargs="*", default=list(P.BACKENDS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    names = args.backends if args.mode == "fake" else [None]
    rs = []
    for nm in names:
        r = run_backend(args.mode, nm)
        rs.append(r)
        print(json.dumps({"backend": r["backend"], "pooled": r["pooled"], "kept": r["kept"], "calibration": r["calibration"]}), flush=True)
    if args.mode == "fake" and len(rs) == 3:
        print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in verdict(rs).items()))
    if args.out:
        json.dump(rs, open(args.out, "w"), indent=1)
    return rs


if __name__ == "__main__":
    main()
