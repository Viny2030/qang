"""
What removes the missing weight-1 shots on torino? A channel-by-channel
comparison of Aer's device noise model and qang's simulator (§132)

§129 closed the kept-fraction gap of §124 on the two ECR devices by adding
readout error to qang's QNN simulator (within 0.006), but on fake torino (CZ)
the simulator still predicts 0.783 against 0.724 measured. The circuits are
weight-preserving, so only channels that change the Hamming weight can lower
the kept fraction: two-qubit gate errors, single-qubit gate errors, T1
relaxation and readout. qang's simulator models the first, the third and the
fourth, but not single-qubit gate errors; its T1 is applied over the whole
circuit duration (idle time included), while Aer's device model applies
thermal relaxation only during gates. On torino the gates are fast (CZ 68 ns,
total circuit 7.4 us against 35-39 us on the ECR devices), so an idle-time T1
that may have compensated the missing single-qubit errors on the ECR devices
cannot do so there.

Here the model A circuits of §124 (weight 1, iris, cancer, wine, digits; 189
test inputs; same layout) are run with Aer under the full device noise model
and under each channel alone, and qang's simulator is decomposed the same way:

  channel            Aer                                   qang simulator
  full               NoiseModel.from_backend               §129 (gate + T1/T2 + readout)
  2q gates           only the CZ / ECR depolarizing errors  RBS depolarizing (§124)
  1q gates           only the sx / x / id errors            (absent; added below)
  thermal            only thermal relaxation (gates)         T1/T2 per sublayer (§119)
  readout            only readout error                      readout confusion (§129)

and a single-qubit depolarizing channel is added to qang's simulator: after
each RBS, on each of its two qubits, with probability 1 - (1 - 2 e1)^m1, where
e1 is the mean published sx error on the layout and m1 the number of sx/x
gates per qubit per RBS in the transpiled circuit.

Calibration data were inspected before writing the predictions (no noisy run):
per RBS 3.6 two-qubit gates on every backend; sx/x gates per circuit 210 /
234 / 193 with mean sx error 2.8e-4 / 1.4e-4 / 2.2e-4 (brisbane / sherbrooke
/ torino); circuit duration 39.0 / 35.1 / 7.4 us.

Sanity check (not a prediction): Aer with the full model reproduces the
§124 device kept fraction within 0.01 on every backend.

Pre-registered predictions (committed before any run):
  T1  on torino, Aer's single-qubit gate errors alone lose at least 0.02 of
      the weight-1 fraction (kept <= 0.98).
  T2  on torino, qang's two-qubit channel alone predicts Aer's two-qubit
      channel alone within 0.01 (the two-qubit model itself is right).
  T3  on brisbane, qang's T1/T2 channel alone loses at least 0.02 more of the
      weight-1 fraction than Aer's thermal channel alone (idle-time
      relaxation stands in for the missing errors on the ECR devices).
  T4  with the single-qubit channel added, qang's full prediction is within
      0.05 of the §124 device kept fraction on all three backends.

python examples/qnn_torino_residual_qg.py

Findings:

FINDINGS_PLACEHOLDER
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qnn_classifier_qg as Q  # noqa: E402
import qnn_full_pipeline_qg as P  # noqa: E402
import qnn_gate_noise_qg as G  # noqa: E402
import qnn_hardware_qg as H  # noqa: E402
import qnn_ionq_datasets_qg as DS  # noqa: E402
import qnn_readout_error_qg as RO  # noqa: E402
from qang.qml import WeightQNN  # noqa: E402

N = G.N
ONE = G.ONE
SHOTS = 2000
CHANNELS = ("full", "2q gates", "1q gates", "thermal", "readout")
OUT = os.path.join(HERE, "data", "qnn_torino_residual.json")


class GateNoise1QNN(G.GateNoiseQNN):
    """GateNoiseQNN plus a single-qubit depolarizing channel of probability ``one_qubit_error`` on each qubit
    of every RBS (after it). With one_qubit_error = 0 it equals GateNoiseQNN."""

    one_qubit_error = 0.0

    def _depol_one(self, r, q):
        """I/2 (x) Tr_q(rho) on qubit q."""
        b, perm = self._b[q], self._perm[q]
        same = (b[:, None] == b[None, :]).astype(float)       # entries diagonal in qubit q
        diag = r * same[None]
        return 0.5 * (diag + diag[:, perm][:, :, perm])

    def probs(self, theta, psi, gamma=None, dephasing=0.0):
        if not self.one_qubit_error:
            return super().probs(theta, psi, gamma, dephasing)
        # same loop as GateNoiseQNN.probs with the 1q channel after each RBS (and its 2q channel)
        import math

        self._tables()
        n = self.n
        gam = None if gamma is None else np.broadcast_to(np.asarray(gamma, float), (n,))
        rho = np.einsum("si,sj->sij", psi, psi)
        k = 0
        p1 = self.one_qubit_error
        for _ in range(self.layers):
            for pairs in self.sublayers:
                for a, b in pairs:
                    U = self._rbs(a, b, theta[k])
                    k += 1
                    rho = U[None] @ rho @ U.T[None]
                    if self.gate_error > 0:
                        rho = (1 - self.gate_error) * rho + self.gate_error * self._depol_pair(rho, a, b)
                    for q in (a, b):
                        rho = (1 - p1) * rho + p1 * self._depol_one(rho, q)
                if dephasing:
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


def one_qubit_calibration(backend, circuit, layout):
    """Single-qubit depolarizing probability per qubit per RBS from the published sx errors."""
    from qiskit import transpile

    tc = transpile(circuit, backend=backend, optimization_level=1, seed_transpiler=119, initial_layout=layout)
    t = backend.target
    e1 = float(np.mean([t["sx"][(q,)].error for q in layout]))
    n1 = sum(1 for i in tc.data if i.operation.name in ("sx", "x"))
    m1 = n1 / 19 / 2
    return {"mean sx error": e1, "1q gates": n1, "per qubit per RBS": m1, "depolarizing": 1 - (1 - 2 * e1) ** m1}


def aer_models(backend):
    from qiskit_aer.noise import NoiseModel

    full = NoiseModel.from_backend(backend)
    gates = NoiseModel.from_backend(backend, readout_error=False, thermal_relaxation=False)
    two = NoiseModel(basis_gates=gates.basis_gates)
    one = NoiseModel(basis_gates=gates.basis_gates)
    for name, per_q in gates._local_quantum_errors.items():
        for qubits, err in per_q.items():
            (two if len(qubits) == 2 else one).add_quantum_error(err, name, qubits)
    return {
        "full": full,
        "2q gates": two,
        "1q gates": one,
        "thermal": NoiseModel.from_backend(backend, gate_error=False, readout_error=False),
        "readout": NoiseModel.from_backend(backend, gate_error=False, thermal_relaxation=False),
    }


def kept_of_counts(counts):
    n = sum(counts.values())
    return sum(c for k, c in counts.items() if k.replace(" ", "").count("1") == 1) / n


def run_backend(name, datasets=Q.DATASETS, shots=SHOTS, aer=True):
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    backend = H.get_backend("fake", name)
    layout = None
    models = aer_models(backend) if aer else {}
    aer_kept = {c: [] for c in models}
    sim_kept = {c: [] for c in ("full", "2q gates", "1q gates", "thermal", "readout", "full + 1q")}
    for d, ds in enumerate(datasets):
        Xtr, Xte, ytr, yte = P.data(ds)
        A = WeightQNN(N, 1).fit(Xtr, ytr, G.EPOCHS, seed=DS.TRAIN_SEED)
        if layout is None:
            c0 = H.build_circuit(A, Xte[0])
            layout, cal = P.calibration(backend, c0)
            gcal = G.gate_calibration(backend, c0, layout)
            ocal = one_qubit_calibration(backend, c0, layout)
            noise = {"gamma": cal["gamma"], "dephasing": cal["dephasing"]}
            ro = RO.readout_errors(backend, layout)
        th, psi = A.params_[: A.n_theta], A.encode(Xte)

        def sim(p2=0.0, p1=0.0, thermal=False, readout=False):
            m = GateNoise1QNN(N, 1)
            m.gate_error, m.one_qubit_error = p2, p1
            pr = m.probs(th, psi, **(noise if thermal else {}))
            if readout:
                pr = RO.apply_readout(pr, ro)
            return float(np.mean(pr[:, ONE].sum(axis=1)))

        p2, p1 = gcal["depolarizing"], ocal["depolarizing"]
        sim_kept["full"].append(sim(p2, 0.0, True, True))
        sim_kept["2q gates"].append(sim(p2))
        sim_kept["1q gates"].append(sim(0.0, p1))
        sim_kept["thermal"].append(sim(thermal=True))
        sim_kept["readout"].append(sim(readout=True))
        sim_kept["full + 1q"].append(sim(p2, p1, True, True))
        if aer:
            circs = [H.build_circuit(A, x) for x in Xte]
            for c, nm in models.items():
                s = AerSimulator(noise_model=nm)
                tc = transpile(circs, backend=backend, optimization_level=1, seed_transpiler=119, initial_layout=layout)
                res = s.run(tc, shots=shots, seed_simulator=1320 + d).result()
                aer_kept[c].append(float(np.mean([kept_of_counts(res.get_counts(i)) for i in range(len(tc))])))
    # the datasets are averaged with equal weight, as in §124 and §129
    avg = lambda v: float(np.mean(v))  # noqa: E731
    return {"backend": name, "layout": layout, "1q calibration": ocal, "2q calibration": gcal,
            "aer": {c: avg(v) for c, v in aer_kept.items()}, "sim": {c: avg(v) for c, v in sim_kept.items()}}


def verdict(rows, dev):
    by = {r["backend"]: r for r in rows}
    t = by["fake_torino"]
    out = {
        "T1": t["aer"]["1q gates"] <= 0.98,
        "T2": abs(t["sim"]["2q gates"] - t["aer"]["2q gates"]) <= 0.01,
    }
    b = by["fake_brisbane"]
    out["T3"] = (1 - b["sim"]["thermal"]) - (1 - b["aer"]["thermal"]) >= 0.02
    out["T4"] = all(abs(by[k]["sim"]["full + 1q"] - dev[k]["kept device"]) <= 0.05 for k in by)
    return out


def main():
    dev = RO.device_values()
    rows = []
    for name in P.BACKENDS:
        r = run_backend(name)
        r["kept device 124"] = dev[name]["kept device"]
        print(json.dumps({"backend": name, "device 124": round(dev[name]["kept device"], 4),
                          "aer": {k: round(v, 4) for k, v in r["aer"].items()},
                          "sim": {k: round(v, 4) for k, v in r["sim"].items()},
                          "p1": round(r["1q calibration"]["depolarizing"], 5)}), flush=True)
        rows.append(r)
    v = verdict(rows, dev)
    print("predictions: " + ", ".join(f"{k} {'PASS' if ok else 'FAIL'}" for k, ok in v.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdict": v}, fh, indent=1)
    return rows, v


if __name__ == "__main__":
    main()
